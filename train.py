import copy
import enum
import os
import pdb
import pickle
import random
import time
import warnings
from re import X, template
from readline import set_pre_input_hook

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# import clip
from clip import clip
from matplotlib.pyplot import phase_spectrum
from numpy.core.fromnumeric import cumprod
from pytz import NonExistentTimeError
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from classes import CLASSES, CUSTOM_TEMPLATES, GENERIC_PROMPT_COLLECTIONS
from diffgrad import diffgrad
from logger import Logger
from utils import *


def load_clip_to_cpu(visual_backbone):
    backbone_name = visual_backbone
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url, os.path.expanduser("~/.cache/clip"))

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        self.token_embedding = clip_model.token_embedding

    def forward(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x


class model:
    def __init__(self, config, data, test=False):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.training_opt = self.config["training_opt"]
        self.model_opt = self.config["model"]
        self.data = data
        self.test_mode = test
        self.num_gpus = torch.cuda.device_count()
        self.do_shuffle = config["shuffle"] if "shuffle" in config else False
        self.clip_model = load_clip_to_cpu(
            self.model_opt["clip"]["params"]["visual_backbone"]
        )
        self.writer = SummaryWriter(log_dir="./runs/" + self.training_opt["tf_folder"])

        # Setup logger
        self.logger = Logger(self.training_opt["log_dir"])

        self.optimizer_variant = (
            config["optimizer_variant"] if "optimizer_variant" in config else None
        )

        # Initialize model
        self.init_models()

        # Under training mode, initialize training steps, optimizers, schedulers, criterions, and centroids
        if not self.test_mode:

            print("Using steps for training.")
            self.training_data_num = len(self.data["train"].dataset)
            self.epoch_steps = int(
                self.training_data_num / self.training_opt["batch_size"]
            )

            # Initialize model optimizer and scheduler
            print("Initializing model optimizer.")
            self.scheduler_params = self.training_opt["scheduler_params"]
            self.model_optimizer, self.model_optimizer_scheduler = self.init_optimizers(
                self.model_optim_params_list
            )
            self.init_criterions()

            # Set up log file
            self.log_file = os.path.join(self.training_opt["log_dir"], "log.txt")
            if os.path.isfile(self.log_file):
                os.remove(self.log_file)
            self.logger.log_cfg(self.config)
        else:
            self.log_file = None

    def init_models(self, optimizer=True):
        # import pdb; pdb.set_trace()

        self.model_optim_params_list = []
        self.model_optim_params_list_LBFGS = []

        print("Using", torch.cuda.device_count(), "GPUs.")

        # Initializing CLIP visual and Text models
        self.visual_model = torch.nn.DataParallel(self.clip_model.visual).cuda()
        text_model = TextEncoder(self.clip_model)
        self.text_model = torch.nn.DataParallel(text_model).cuda()

        in_dim = self.model_opt["fusion"]["params"]["in_dim"]
        out_dim = self.model_opt["fusion"]["params"]["out_dim"]
        self.fusion = torch.nn.DataParallel(
            nn.Linear(in_dim, out_dim, bias=False)
        ).cuda()

        # feat_dim = self.model_opt["adapter"]["params"]["feat_dim"]
        # # self.load_model(self.config['model_dir'])
        # self.adapter = torch.nn.DataParallel(
        #     nn.Linear(feat_dim, feat_dim, bias=False)
        # ).cuda()

        # if self.training_opt['phaseA'] is not True:
        #     self.load_model(self.config['model_dir'])

        if "model_dir" in self.config:
            print("Loading model weights from ", self.config["model_dir"])
            self.load_model(self.config["model_dir"])

        self.load_model()

        if self.training_opt["image_encoder_frozen"] is True:
            for param_name, param in self.visual_model.named_parameters():
                param.requires_grad = False

        if self.training_opt["text_encoder_frozen"] is True:
            for param_name, param in self.text_model.named_parameters():
                param.requires_grad = False

        # optim_params_adapter = self.model_opt["adapter"]["optim_params"]
        # self.model_optim_params_list.append(
        #     {
        #         "params": self.adapter.parameters(),
        #         "lr": optim_params_adapter["lr"],
        #         "momentum": optim_params_adapter["momentum"],
        #         "weight_decay": optim_params_adapter["weight_decay"],
        #     }
        # )

        optim_params_clip = self.model_opt["clip"]["optim_params"]
        self.model_optim_params_list.append(
            {
                "params": self.visual_model.parameters(),
                "lr": optim_params_clip["lr"],
                "momentum": optim_params_clip["momentum"],
                "weight_decay": optim_params_clip["weight_decay"],
            }
        )
        self.model_optim_params_list_LBFGS.extend(self.visual_model.parameters())

        self.model_optim_params_list.append(
            {
                "params": self.text_model.parameters(),
                "lr": optim_params_clip["lr"],
                "momentum": optim_params_clip["momentum"],
                "weight_decay": optim_params_clip["weight_decay"],
            }
        )
        self.model_optim_params_list_LBFGS.extend(self.text_model.parameters())

        optim_params_fusion = self.model_opt["fusion"]["optim_params"]
        self.model_optim_params_list.append(
            {
                "params": self.fusion.parameters(),
                "lr": optim_params_fusion["lr"],
                "momentum": optim_params_fusion["momentum"],
                "weight_decay": optim_params_fusion["weight_decay"],
            }
        )
        self.model_optim_params_list_LBFGS.extend(self.fusion.parameters())

    def init_criterions(self):
        criterion_defs = self.config["criterions"]
        self.criterions = {}
        self.criterion_weights = {}

        for key, val in criterion_defs.items():
            def_file = val["def_file"]
            loss_args = list(val["loss_params"].values())

            self.criterions[key] = (
                source_import(def_file).create_loss(*loss_args).cuda()
            )
            self.criterion_weights[key] = val["weight"]

            if val["optim_params"]:
                print("Initializing criterion optimizer.")
                optim_params = val["optim_params"]
                optim_params = [
                    {
                        "params": self.criterions[key].parameters(),
                        "lr": optim_params["lr"],
                        "momentum": optim_params["momentum"],
                        "weight_decay": optim_params["weight_decay"],
                    }
                ]
                # Initialize criterion optimizer and scheduler
                (
                    self.criterion_optimizer,
                    self.criterion_optimizer_scheduler,
                ) = self.init_optimizers(optim_params)
            else:
                self.criterion_optimizer = None

    def init_optimizers(self, optim_params):

        if self.optimizer_variant == "LBFGS":
            optim_params = self.config["optimizer_args"]
            optimizer = optim.LBFGS(
                self.fusion.parameters(),
                history_size=int(optim_params["history_size"]),
                max_iter=int(optim_params["max_iter"]),
                lr=float(optim_params["lr"]),
            )
        elif self.optimizer_variant == "Adam":
            optimizer = optim.Adam(optim_params)
        elif self.optimizer_variant == "diffgrad":
            optimizer = diffgrad(optim_params)
        else:
            optimizer = optim.SGD(optim_params)
        if self.config["coslr"]:
            print("===> Using coslr eta_min={}".format(self.config["endlr"]))
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, self.training_opt["num_epochs"], eta_min=self.config["endlr"]
            )
        else:
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.scheduler_params["step_size"],
                gamma=self.scheduler_params["gamma"],
            )
        return optimizer, scheduler

    def batch_forward(self, inputs, labels, phase="train", paths=None):
        """
        This is a general single batch running function.
        """

        variation = self.training_opt["variation"]

        if variation == "clip":
            self.regular_clip(inputs, labels, phase=phase)
        elif variation == "clip_with_linear_layer":
            self.clip_with_linear_layer(inputs, labels, phase=phase)
        elif variation == "add_embs_then_linear1":
            self.add_embs_then_linear1(inputs, labels, phase=phase)
        elif variation == "image_linear_probe":
            self.image_linear_probe(inputs, labels, phase=phase)
        elif variation == "image_plus_text":
            self.image_plus_text(inputs, labels, phase=phase)
        elif variation == "image_dot_text":
            self.image_dot_text(inputs, labels, phase=phase)
        elif variation == "image_linear_probe_boosted":
            self.image_linear_probe_boosted(inputs, labels, phase=phase, paths=paths)
        elif variation == "image_concat_text":
            self.image_concat_text(inputs, labels, phase=phase)

    ###########################################################################
    ### Forward pass variations
    ###########################################################################

    def regular_clip(self, inputs, labels, phase="train"):

        classnames = CLASSES
        templates = CUSTOM_TEMPLATES["ImageNet"]

        texts = torch.cat([clip.tokenize(templates.format(c)) for c in classnames])
        texts = texts.cuda()
        zeroshot_weights = self.text_model(texts).float()
        zeroshot_weights = zeroshot_weights / zeroshot_weights.norm(
            dim=-1, keepdim=True
        )

        y = self.visual_model(inputs.half()).float()
        x = y
        x = x / x.norm(dim=-1, keepdim=True)

        self.logits = 100.0 * x @ zeroshot_weights.t()

    def clip_with_linear_layer(self, inputs, labels, phase="train"):

        classnames = CLASSES
        templates = CUSTOM_TEMPLATES["ImageNet"]

        texts = torch.cat([clip.tokenize(templates.format(c)) for c in classnames])
        texts = texts.cuda()
        zeroshot_weights = self.text_model(texts).float()
        zeroshot_weights = zeroshot_weights / zeroshot_weights.norm(
            dim=-1, keepdim=True
        )

        y = self.visual_model(inputs.half()).float()
        x = y
        x = x / x.norm(dim=-1, keepdim=True)

        logits = 100.0 * x @ zeroshot_weights.t()
        self.logits = self.fusion(logits)

    def add_embs_then_linear1(self, inputs, labels, phase="train"):

        classnames = CLASSES
        templates = CUSTOM_TEMPLATES["ImageNet"]

        texts = torch.cat([clip.tokenize(templates.format(c)) for c in classnames])
        texts = texts.cuda()
        zeroshot_weights = self.text_model(texts).float()
        zeroshot_weights = zeroshot_weights / zeroshot_weights.norm(
            dim=-1, keepdim=True
        )

        x = self.visual_model(inputs.half()).float()
        x = x / x.norm(dim=-1, keepdim=True)

        fused = x.unsqueeze(1) + zeroshot_weights.unsqueeze(
            0
        )  # [100, 1, 1024] + [1, 1000, 1024]

        self.logits = self.fusion(fused).squeeze(-1)

    ###########################################################################

    def image_linear_probe(self, inputs, labels, phase="train"):
        """
        Linear layer of dimension 1024x1000 on top of image embeddings,
        no text used.
        """

        x = self.visual_model(inputs.half()).float()
        x = x / x.norm(dim=-1, keepdim=True)

        self.logits = self.fusion(x)

    def image_linear_probe_boosted(self, inputs, labels, phase="train", paths=None):
        """
        Linear layer of dimension 1024x1000 on top of image embeddings,
        no text used.  To work in tandem with boosted dataset.
        """

        classnames = np.array(CLASSES)
        templates = np.array(GENERIC_PROMPT_COLLECTIONS["ImageNet"])

        if phase == "train" and paths is not None:

            classnames_for_labels = classnames[labels.cpu()]
            prompts_for_labels = templates[paths.cpu()]

            tokens = []
            for p, c in zip(prompts_for_labels, classnames_for_labels):
                tokens.append(clip.tokenize(p.format(c)))
            texts = torch.cat(tokens)

            zeroshot_weights = self.text_model(texts).float()
            zeroshot_weights = zeroshot_weights / zeroshot_weights.norm(
                dim=-1, keepdim=True
            )
            x = self.visual_model(inputs.half()).float()
            x = x / x.norm(dim=-1, keepdim=True)

            idx_im_is_boosted = np.where(paths != 1)
            idx_im_is_not_boosted = np.where(paths == 1)

            zeroshot_weights[idx_im_is_not_boosted] = 0.0
            x[idx_im_is_boosted] = 0.0

            self.logits = self.fusion(x + zeroshot_weights)

        else:
            x = self.visual_model(inputs.half()).float()
            x = x / x.norm(dim=-1, keepdim=True)

            self.logits = self.fusion(x)

    def image_linear_probe_boosted_wiki(
        self, inputs, labels, phase="train", paths=None
    ):
        """
        Linear layer of dimension 1024x1000 on top of image embeddings,
        no text used.  To work in tandem with boosted dataset.
        """

        classnames = np.array(CLASSES)
        templates = np.array(GENERIC_PROMPT_COLLECTIONS["ImageNet"])

        if phase == "train" and paths is not None:

            classnames_for_labels = classnames[labels.cpu()]
            prompts_for_labels = templates[paths.cpu()]

            tokens = []
            for p, c in zip(prompts_for_labels, classnames_for_labels):
                tokens.append(clip.tokenize(p.format(c)))
            texts = torch.cat(tokens)

            zeroshot_weights = self.text_model(texts).float()
            zeroshot_weights = zeroshot_weights / zeroshot_weights.norm(
                dim=-1, keepdim=True
            )
            x = self.visual_model(inputs.half()).float()
            x = x / x.norm(dim=-1, keepdim=True)

            idx_im_is_boosted = np.where(paths != 1)
            idx_im_is_not_boosted = np.where(paths == 1)

            zeroshot_weights[idx_im_is_not_boosted] = 0.0
            x[idx_im_is_boosted] = 0.0

            self.logits = self.fusion(x + zeroshot_weights)

        else:
            x = self.visual_model(inputs.half()).float()
            x = x / x.norm(dim=-1, keepdim=True)

            self.logits = self.fusion(x)

    def image_plus_text(self, inputs, labels, phase="train", lam=None):
        """
        Add every image embedding to the text embedding corresponding to "a photo of a".
        This is the simplest phrase and thus we refer to it as the default.

        """

        if phase == "train":

            only_med_and_few = check_config(self.training_opt, "only_med_and_few")

            lam = float(self.training_opt["image_emb_weight"]) if lam == None else lam
            classnames = np.array(CLASSES)
            templates = CUSTOM_TEMPLATES["ImageNet"]

            classnames_for_labels = classnames[labels.cpu()]

            texts = torch.cat(
                [clip.tokenize(templates.format(c)) for c in classnames_for_labels]
            )
            texts = texts.cuda()
            zeroshot_weights = self.text_model(texts).float()
            zeroshot_weights = zeroshot_weights / zeroshot_weights.norm(
                dim=-1, keepdim=True
            )

            x = self.visual_model(inputs.half()).float()
            x = x / x.norm(dim=-1, keepdim=True)

            if only_med_and_few:
                mask = torch.isin(
                    labels,
                    torch.tensor(self.data["label_categorization"]["many"]).cuda(),
                )
                indices = torch.argwhere(mask)
                zeroshot_weights[indices] = x[indices]

            fused = ((lam) * x) + ((1 - lam) * zeroshot_weights)

            self.logits = self.fusion(fused)
        else:
            # self.training_opt["eval_type"] = "image_and_text"
            if self.training_opt["eval_type"] == "image_and_text":

                m = nn.Softmax(dim=-1)
                lam = (
                    float(self.training_opt["image_emb_weight"]) if lam == None else lam
                )
                batch_size = inputs.shape[0]
                classnames = np.array(CLASSES)
                templates = CUSTOM_TEMPLATES["ImageNet"]
                texts = torch.cat(
                    [clip.tokenize(templates.format(c)) for c in classnames]
                ).cuda()
                zeroshot_weights = self.text_model(texts).float()
                zeroshot_weights = zeroshot_weights / zeroshot_weights.norm(
                    dim=-1, keepdim=True
                )  # 1000 x 1024

                x = self.visual_model(inputs.half()).float()
                x = x / x.norm(dim=-1, keepdim=True)  # batch_size x 1024

                # import pdb; pdb.set_trace()
                y = x.unsqueeze(-1).repeat(1, 1, 1000)  # batch_size x 1024 x 1000
                z = zeroshot_weights.T.unsqueeze(0).repeat(
                    batch_size, 1, 1
                )  # batch_size x 1024 x 1000

                y = y.permute(0, 2, 1)  # batch_size x 1000 x 1024
                z = z.permute(0, 2, 1)  # batch_size x 1000 x 1024
                fused = ((lam) * y) + ((1 - lam) * z)  # batch_size x 1024 x 1000
                # fused = fused.reshape(batch_size*1000, 1024)  # batch_size*1000 x 2048
                out = m(self.fusion(fused))  # batch_size*1000 x 1000
                # out = out.reshape(batch_size, 1000, 1000) # batch_size x 1000 x 1000
                # out = out * torch.eye(1000,1000).cuda() # batch_size x 1000 x 1000
                out = out.sum(dim=1)  # batch_size x 1000
                self.logits = out
            else:
                x = self.visual_model(inputs.half()).float()
                x = x / x.norm(dim=-1, keepdim=True)

                self.logits = self.fusion(x)

    def image_dot_text(self, inputs, labels, phase="train"):
        """
        Multiply every image embedding to the text embedding to "a photo of a".
        This is the simplest phrase and thus we refer to it as the default.
        """

        if phase == "train":
            classnames = np.array(CLASSES)
            templates = CUSTOM_TEMPLATES["ImageNet"]

            classnames_for_labels = classnames[labels.cpu()]

            texts = torch.cat(
                [clip.tokenize(templates.format(c)) for c in classnames_for_labels]
            )
            texts = texts.cuda()
            zeroshot_weights = self.text_model(texts).float()
            zeroshot_weights = zeroshot_weights / zeroshot_weights.norm(
                dim=-1, keepdim=True
            )

            x = self.visual_model(inputs.half()).float()
            x = x / x.norm(dim=-1, keepdim=True)

            fused = x * zeroshot_weights

            self.logits = self.fusion(fused)
        else:

            if self.training_opt["eval_type"] == "image_and_text":
                m = nn.Softmax(dim=-1)
                batch_size = inputs.shape[0]
                classnames = np.array(CLASSES)
                templates = CUSTOM_TEMPLATES["ImageNet"]
                texts = torch.cat(
                    [clip.tokenize(templates.format(c)) for c in classnames]
                ).cuda()
                zeroshot_weights = self.text_model(texts).float()
                zeroshot_weights = zeroshot_weights / zeroshot_weights.norm(
                    dim=-1, keepdim=True
                )  # 1000 x 1024

                x = self.visual_model(inputs.half()).float()
                x = x / x.norm(dim=-1, keepdim=True)  # batch_size x 1024

                y = x.unsqueeze(-1).repeat(1, 1, 1000)  # batch_size x 1024 x 1000
                z = zeroshot_weights.T.unsqueeze(0).repeat(
                    batch_size, 1, 1
                )  # batch_size x 1024 x 1000
                fused = y * z  # batch_size x 1000 x 2048
                fused = fused.reshape(batch_size * 1000, 1024)  # batch_size*1000 x 2048
                out = self.fusion(fused)  # batch_size*1000 x 1000
                out = out.reshape(batch_size, 1000, 1000)  # batch_size x 1000 x 1000
                out = out * torch.eye(1000, 1000).cuda()  # batch_size x 1000 x 1000
                out = m(out).sum(dim=1)  # batch_size x 1000
                self.logits = out

            else:

                x = self.visual_model(inputs.half()).float()
                x = x / x.norm(dim=-1, keepdim=True)

                self.logits = self.fusion(x)

    def image_concat_text(self, inputs, labels, phase="train"):
        """
        Multiply every image embedding to the text embedding to "a photo of a".
        This is the simplest phrase and thus we refer to it as the default.
        """

        if phase == "train":
            classnames = np.array(CLASSES)
            templates = CUSTOM_TEMPLATES["ImageNet"]

            classnames_for_labels = classnames[labels.cpu()]

            texts = torch.cat(
                [clip.tokenize(templates.format(c)) for c in classnames_for_labels]
            )
            texts = texts.cuda()
            zeroshot_weights = self.text_model(texts).float()
            zeroshot_weights = zeroshot_weights / zeroshot_weights.norm(
                dim=-1, keepdim=True
            )

            x = self.visual_model(inputs.half()).float()
            x = x / x.norm(dim=-1, keepdim=True)

            fused = torch.cat([x, zeroshot_weights], dim=-1)

            self.logits = self.fusion(fused)
        else:
            m = nn.Softmax(dim=-1)
            batch_size = inputs.shape[0]
            classnames = np.array(CLASSES)
            templates = CUSTOM_TEMPLATES["ImageNet"]
            texts = torch.cat(
                [clip.tokenize(templates.format(c)) for c in classnames]
            ).cuda()
            zeroshot_weights = self.text_model(texts).float()
            zeroshot_weights = zeroshot_weights / zeroshot_weights.norm(
                dim=-1, keepdim=True
            )  # 1000 x 1024

            x = self.visual_model(inputs.half()).float()
            x = x / x.norm(dim=-1, keepdim=True)  # batch_size x 1024

            # import pdb; pdb.set_trace()
            y = x.unsqueeze(-1).repeat(1, 1, 1000)  # batch_size x 1024 x 1000
            z = zeroshot_weights.T.unsqueeze(0).repeat(
                batch_size, 1, 1
            )  # batch_size x 1024 x 1000

            y = y.permute(0, 2, 1)  # batch_size x 1000 x 1024
            z = z.permute(0, 2, 1)  # batch_size x 1000 x 1024
            fused = torch.cat([y, z], dim=-1)  # batch_size x 1024 x 1000
            out = m(self.fusion(fused))  # batch_size*1000 x 1000
            # out = out * torch.eye(1000,1000).cuda() # batch_size x 1000 x 1000
            out = out.sum(dim=1)  # batch_size x 1000
            self.logits = out

    ###########################################################################
    ###########################################################################

    def batch_backward(self):
        # Zero out optimizer gradients
        self.model_optimizer.zero_grad()
        if self.criterion_optimizer:
            self.criterion_optimizer.zero_grad()
        # Back-propagation from loss outputs
        self.loss.backward()
        # Step optimizers

        self.model_optimizer.step()
        if self.criterion_optimizer:
            self.criterion_optimizer.step()

    def batch_loss(self, labels):
        self.loss = 0

        # First, apply performance loss
        if "PerformanceLoss" in self.criterions.keys():
            self.loss_perf = self.criterions["PerformanceLoss"](self.logits, labels)
            self.loss_perf *= self.criterion_weights["PerformanceLoss"]
            self.loss += self.loss_perf

        # Apply loss on features if set up
        if "FeatureLoss" in self.criterions.keys():
            self.loss_feat = self.criterions["FeatureLoss"](self.features, labels)
            self.loss_feat = self.loss_feat * self.criterion_weights["FeatureLoss"]
            # Add feature loss to total loss
            self.loss += self.loss_feat

    def shuffle_batch(self, x, y):
        index = torch.randperm(x.size(0))
        x = x[index]
        y = y[index]
        return x, y

    def train_step(self, step, epoch, inputs, labels, indexes, t):

        # Break when step equal to epoch step
        if step == self.epoch_steps:
            return
        if self.do_shuffle:
            inputs, labels = self.shuffle_batch(inputs, labels)
        inputs, labels = inputs.cuda(), labels.cuda()

        # If on training phase, enable gradients

        with torch.set_grad_enabled(True):

            if self.training_opt["variation"] == "image_text_all_operations":
                self.image_plus_text(inputs, labels, phase="train", lam=0)
                self.batch_loss(labels)
                self.batch_backward()

                self.image_plus_text(inputs, labels, phase="train", lam=1)
                self.batch_loss(labels)
                self.batch_backward()

                self.image_plus_text(inputs, labels, phase="train", lam=0.5)
                self.batch_loss(labels)
                self.batch_backward()

                self.image_dot_text(inputs, labels, phase="train")
                self.batch_loss(labels)
                self.batch_backward()

            else:
                # If training, forward with loss, and no top 5 accuracy calculation
                self.batch_forward(inputs, labels, phase="train", paths=t)
                self.batch_loss(labels)
                self.batch_backward()

            # Tracking predictions
            _, preds = torch.max(self.logits, 1)
            self.total_preds.append(torch2numpy(preds))
            self.total_labels.append(torch2numpy(labels))

            # TODO uncomment
            self.minibatch_training_results(step, epoch, preds, labels)

    def train_step_LBFGS(self, step, epoch, inputs, labels, indexes, t):

        # Break when step equal to epoch step
        if step == self.epoch_steps:
            return
        if self.do_shuffle:
            inputs, labels = self.shuffle_batch(inputs, labels)
        inputs, labels = inputs.cuda(), labels.cuda()

        # If on training phase, enable gradients

        def closure():
            self.model_optimizer.zero_grad()
            if self.criterion_optimizer:
                self.criterion_optimizer.zero_grad()

            self.batch_forward(inputs, labels, phase="train", paths=t)
            self.batch_loss(labels)
            # Back-propagation from loss outputs
            if self.loss.requires_grad:
                self.loss.backward()
            return self.loss

        self.model_optimizer.step(closure)
        if self.criterion_optimizer:
            self.criterion_optimizer.step(closure)

        # Tracking predictions
        _, preds = torch.max(self.logits, 1)
        self.total_preds.append(torch2numpy(preds))
        self.total_labels.append(torch2numpy(labels))

        # TODO uncomment
        self.minibatch_training_results(step, epoch, preds, labels)

    def minibatch_training_results(self, step, epoch, preds, labels):

        # Output minibatch training results
        if step % self.training_opt["display_step"] == 0:

            minibatch_loss_feat = (
                self.loss_feat.item()
                if "FeatureLoss" in self.criterions.keys()
                else None
            )
            minibatch_loss_perf = (
                self.loss_perf.item() if "PerformanceLoss" in self.criterions else None
            )
            minibatch_loss_total = self.loss.item()
            minibatch_acc = mic_acc_cal(preds, labels)

            print_str = [
                "Config {}".format(self.training_opt["tf_folder"]),
                "Epoch: [%d/%d]" % (epoch, self.training_opt["num_epochs"]),
                "Step: %5d" % (step),
                "Minibatch_loss_feature: %.3f" % (minibatch_loss_feat)
                if minibatch_loss_feat
                else "",
                "Minibatch_loss_performance: %.3f" % (minibatch_loss_perf)
                if minibatch_loss_perf
                else "",
                "Minibatch_accuracy_micro: %.3f" % (minibatch_acc),
            ]
            print_write(print_str, self.log_file)

            loss_info = {
                "Epoch": epoch,
                "Step": step,
                "Total": minibatch_loss_total,
                "CE": minibatch_loss_perf,
                "feat": minibatch_loss_feat,
            }

            self.logger.log_loss(loss_info)
            self.epoch_loss += minibatch_loss_total

    def train_epoch(self, epoch):

        torch.cuda.empty_cache()

        # Set model modes and set scheduler
        # In training, step optimizer scheduler and set model to train()
        self.model_optimizer_scheduler.step()
        if self.criterion_optimizer:
            self.criterion_optimizer_scheduler.step()

        # Iterate over dataset
        self.total_preds = []
        self.total_labels = []

        self.epoch_loss = 0.0

        def call_train_step(step, epoch, inputs, labels, indexes, t):
            if self.optimizer_variant == "LBFGS":
                self.train_step_LBFGS(step, epoch, inputs, labels, indexes, t)
            else:
                self.train_step(step, epoch, inputs, labels, indexes, t)

        for step, (inputs, labels, indexes, t) in enumerate(self.data["train"]):

            if self.training_opt["variation"] == "image_text_all_operations":
                self.training_opt["variation"] = "image_plus_text"
                self.training_opt["image_emb_weight"] = 0
                call_train_step(step, epoch, inputs, labels, indexes, t)
                self.training_opt["image_emb_weight"] = 1
                call_train_step(step, epoch, inputs, labels, indexes, t)
                self.training_opt["image_emb_weight"] = 0.5
                call_train_step(step, epoch, inputs, labels, indexes, t)

            else:
                call_train_step(step, epoch, inputs, labels, indexes, t)

            # Update priority weights if using PrioritizedSampler
            # if self.training_opt['sampler'] and \
            #    self.training_opt['sampler']['type'] == 'PrioritizedSampler':
            if hasattr(self.data["train"].sampler, "update_weights"):
                if hasattr(self.data["train"].sampler, "ptype"):
                    ptype = self.data["train"].sampler.ptype
                else:
                    ptype = "score"
                ws = get_priority(ptype, self.logits.detach(), labels)
                # ws = logits2score(self.logits.detach(), labels)
                inlist = [indexes.cpu().numpy(), ws]
                if self.training_opt["sampler"]["type"] == "ClassPrioritySampler":
                    inlist.append(labels.cpu().numpy())
                self.data["train"].sampler.update_weights(*inlist)
                # self.data['train'].sampler.update_weights(indexes.cpu().numpy(), ws)

        if hasattr(self.data["train"].sampler, "get_weights"):
            self.logger.log_ws(epoch, self.data["train"].sampler.get_weights())
        if hasattr(self.data["train"].sampler, "reset_weights"):
            self.data["train"].sampler.reset_weights(epoch)

        # After every epoch, validation
        rsls = {"epoch": epoch}
        rsls_train = self.eval_with_preds(self.total_preds, self.total_labels)
        # rsls_train = self.eval(phase="train")
        rsls_eval = self.eval(phase="val")
        rsls.update(rsls_train)
        rsls.update(rsls_eval)

        # Reset class weights for sampling if pri_mode is valid
        if hasattr(self.data["train"].sampler, "reset_priority"):
            ws = get_priority(
                self.data["train"].sampler.ptype,
                self.total_logits.detach(),
                self.total_labels,
            )
            self.data["train"].sampler.reset_priority(
                ws, self.total_labels.cpu().numpy()
            )

        # Log results
        self.logger.log_acc(rsls)

        # Under validation, the best model need to be updated
        if self.eval_acc_mic_top1 > self.best_acc:
            self.best_epoch = epoch
            self.best_acc = self.eval_acc_mic_top1
            # best_centroids = self.centroids
            self.best_model_weights["visual_model"] = copy.deepcopy(
                self.visual_model.state_dict()
            )
            self.best_model_weights["text_model"] = copy.deepcopy(
                self.text_model.state_dict()
            )
            # if self.training_opt["phaseA"] is not True:
            self.best_model_weights["fusion"] = copy.deepcopy(self.fusion.state_dict())

        print("===> Saving checkpoint")

        self.writer.add_scalar("Loss/train", self.epoch_loss / step, epoch)
        self.writer.add_scalar("Acc/train", rsls_train["train_all"], epoch)
        self.writer.add_scalar("Acc/train", rsls_train["train_many"], epoch)
        self.writer.add_scalar("Acc/train", rsls_train["train_median"], epoch)
        self.writer.add_scalar("Acc/train", rsls_train["train_low"], epoch)

        self.writer.add_scalar("Acc/val_all", rsls_eval["val_all"], epoch)
        self.writer.add_scalar("Acc/val_many", rsls_eval["val_many"], epoch)
        self.writer.add_scalar("Acc/val_median", rsls_eval["val_median"], epoch)
        self.writer.add_scalar("Acc/val_low", rsls_eval["val_low"], epoch)

        self.save_latest(epoch)
        self.save_model(epoch, self.best_epoch, self.best_model_weights, self.best_acc)

    def train(self):
        # When training the network
        print_str = ["Phase: train"]
        print_write(print_str, self.log_file)
        time.sleep(0.25)

        print_write(["Do shuffle??? --- ", self.do_shuffle], self.log_file)

        # Initialize best model
        self.best_model_weights = {}
        self.best_model_weights["visual_model"] = copy.deepcopy(
            self.visual_model.state_dict()
        )
        self.best_model_weights["text_model"] = copy.deepcopy(
            self.text_model.state_dict()
        )
        # if self.training_opt["phaseA"] is not True:
        self.best_model_weights["fusion"] = copy.deepcopy(self.fusion.state_dict())
        self.best_acc = 0.0
        self.best_epoch = 0
        # best_centroids = self.centroids

        end_epoch = self.training_opt["num_epochs"]

        # Loop over epochs
        for epoch in range(1, end_epoch + 1):

            self.train_epoch(epoch)

            if epoch % 10 == 0:
                rsls_eval_test = self.eval("test" if "test" in self.data else "val")

                self.writer.add_scalar(
                    "Acc/test_all", rsls_eval_test["test_all"], epoch
                )
                self.writer.add_scalar(
                    "Acc/test_many", rsls_eval_test["test_many"], epoch
                )
                self.writer.add_scalar(
                    "Acc/test_median", rsls_eval_test["test_median"], epoch
                )
                self.writer.add_scalar(
                    "Acc/test_low", rsls_eval_test["test_low"], epoch
                )

        print()
        print("Training Complete.")

        print_str = [
            "Best validation accuracy is %.3f at epoch %d"
            % (self.best_acc, self.best_epoch)
        ]
        print_write(print_str, self.log_file)
        # Save the best model and best centroids if calculated
        self.save_model(epoch, self.best_epoch, self.best_model_weights, self.best_acc)

        # Test on the test set
        # self.reset_model(best_model_weights)
        rsls_eval_test = self.eval("test" if "test" in self.data else "val")

        self.writer.add_scalar("Acc/test_all", rsls_eval_test["test_all"], epoch)
        self.writer.add_scalar("Acc/test_many", rsls_eval_test["test_many"], epoch)
        self.writer.add_scalar("Acc/test_median", rsls_eval_test["test_median"], epoch)
        self.writer.add_scalar("Acc/test_low", rsls_eval_test["test_low"], epoch)

        print("Done")

    def eval_with_preds(self, preds, labels):
        # Count the number of examples
        n_total = sum([len(p) for p in preds])

        # Split the examples into normal and mixup
        normal_preds, normal_labels = [], []
        mixup_preds, mixup_labels1, mixup_labels2, mixup_ws = [], [], [], []
        for p, l in zip(preds, labels):
            if isinstance(l, tuple):
                mixup_preds.append(p)
                mixup_labels1.append(l[0])
                mixup_labels2.append(l[1])
                mixup_ws.append(l[2] * np.ones_like(l[0]))
            else:
                normal_preds.append(p)
                normal_labels.append(l)

        # Calculate normal prediction accuracy
        rsl = {
            "train_all": 0.0,
            "train_many": 0.0,
            "train_median": 0.0,
            "train_low": 0.0,
        }
        if len(normal_preds) > 0:
            normal_preds, normal_labels = list(
                map(np.concatenate, [normal_preds, normal_labels])
            )
            n_top1 = mic_acc_cal(normal_preds, normal_labels)
            (
                n_top1_many,
                n_top1_median,
                n_top1_low,
            ) = shot_acc(normal_preds, normal_labels, self.data["train_ltcount"])
            rsl["train_all"] += len(normal_preds) / n_total * n_top1
            rsl["train_many"] += len(normal_preds) / n_total * n_top1_many
            rsl["train_median"] += len(normal_preds) / n_total * n_top1_median
            rsl["train_low"] += len(normal_preds) / n_total * n_top1_low

        # Top-1 accuracy and additional string
        print_str = [
            "\n Training acc Top1: %.3f \n" % (rsl["train_all"]),
            "Many_top1: %.3f" % (rsl["train_many"]),
            "Median_top1: %.3f" % (rsl["train_median"]),
            "Low_top1: %.3f" % (rsl["train_low"]),
            "\n",
        ]
        print_write(print_str, self.log_file)

        return rsl

    def eval(self, phase="val", openset=False, save_feat=False):

        print_str = ["Phase: %s" % (phase)]
        print_write(print_str, self.log_file)
        time.sleep(0.25)

        if openset:
            print(
                "Under openset test mode. Open threshold is %.1f"
                % self.training_opt["open_threshold"]
            )

        torch.cuda.empty_cache()

        self.total_logits = torch.empty((0, self.training_opt["num_classes"])).cuda()
        self.total_labels = torch.empty(0, dtype=torch.long).cuda()
        self.total_paths = np.empty(0)

        get_feat_only = save_feat
        feats_all, labels_all, idxs_all, logits_all = [], [], [], []
        featmaps_all = []
        # Iterate over dataset
        for inputs, labels, indexes, paths in tqdm(self.data[phase]):
            inputs, labels = inputs.cuda(), labels.cuda()

            # If on training phase, enable gradients
            with torch.set_grad_enabled(False):

                # In validation or testing
                self.batch_forward(inputs, labels, phase=phase)
                if not get_feat_only:
                    self.total_logits = torch.cat((self.total_logits, self.logits))
                    self.total_labels = torch.cat((self.total_labels, labels))
                    self.total_paths = np.concatenate((self.total_paths, paths))

                if get_feat_only:
                    logits_all.append(self.logits.cpu().numpy())
                    feats_all.append(self.features.cpu().numpy())
                    labels_all.append(labels.cpu().numpy())
                    idxs_all.append(paths.numpy())

        if get_feat_only:
            typ = "feat"
            if phase == "train_plain":
                name = "train{}_all.pkl".format(typ)
            elif phase == "test":
                name = "test{}_all.pkl".format(typ)
            elif phase == "val":
                name = "val{}_all.pkl".format(typ)

            fname = os.path.join(self.training_opt["log_dir"], name)
            print("===> Saving feats to " + fname)
            with open(fname, "wb") as f:
                pickle.dump(
                    {
                        "feats": np.concatenate(feats_all),
                        "labels": np.concatenate(labels_all),
                        "idxs": np.concatenate(idxs_all),
                    },
                    f,
                    protocol=4,
                )
            return
        probs, preds = F.softmax(self.total_logits.detach(), dim=1).max(dim=1)

        if openset:
            preds[probs < self.training_opt["open_threshold"]] = -1
            self.openset_acc = mic_acc_cal(
                preds[self.total_labels == -1],
                self.total_labels[self.total_labels == -1],
            )
            print("\n\nOpenset Accuracy: %.3f" % self.openset_acc)

        # Calculate the overall accuracy and F measurement
        self.eval_acc_mic_top1 = mic_acc_cal(
            preds[self.total_labels != -1], self.total_labels[self.total_labels != -1]
        )
        self.precision, self.recall, self.eval_f_measure = F_measure(
            preds,
            self.total_labels,
            openset=openset,
            theta=self.training_opt["open_threshold"],
        )
        (
            self.many_acc_top1,
            self.median_acc_top1,
            self.low_acc_top1,
            self.cls_accs,
            self.cls_accs_avg,
        ) = shot_acc(
            preds[self.total_labels != -1],
            self.total_labels[self.total_labels != -1],
            self.data["train_ltcount"],
            acc_per_cls=True,
        )
        # Top-1 accuracy and additional string
        print_str = [
            "\n\n",
            "Phase: %s" % (phase),
            "\n\n",
            "Evaluation_accuracy_micro_top1: %.5f" % (self.eval_acc_mic_top1),
            "\n",
            "Evaluation_accuracy_micro_top1_avg_class: %.5f" % (self.cls_accs_avg),
            "\n",
            "Averaged Precision: %.5f" % (self.precision),
            "\n",
            "Averaged Recall: %.5f" % (self.recall),
            "\n",
            "Averaged F-measure: %.5f" % (self.eval_f_measure),
            "\n",
            "Many_shot_accuracy_top1: %.5f" % (self.many_acc_top1),
            "Median_shot_accuracy_top1: %.5f" % (self.median_acc_top1),
            "Low_shot_accuracy_top1: %.5f" % (self.low_acc_top1),
            "\n",
        ]

        rsl = {
            phase + "_all": self.eval_acc_mic_top1,
            phase + "_many": self.many_acc_top1,
            phase + "_median": self.median_acc_top1,
            phase + "_low": self.low_acc_top1,
            phase + "_fscore": self.eval_f_measure,
        }

        if phase == "val":
            print_write(print_str, self.log_file)
        else:
            acc_str = [
                "{:.3f} \t {:.3f} \t {:.3f} \t {:.3f}".format(
                    self.many_acc_top1 * 100,
                    self.median_acc_top1 * 100,
                    self.low_acc_top1 * 100,
                    self.eval_acc_mic_top1 * 100,
                )
            ]
            if self.log_file is not None and os.path.exists(self.log_file):
                print_write(print_str, self.log_file)
                print_write(acc_str, self.log_file)
            else:
                print(*print_str)
                print(*acc_str)

        if phase == "test":
            with open(
                os.path.join(self.training_opt["log_dir"], "cls_accs.pkl"), "wb"
            ) as f:
                pickle.dump(self.cls_accs, f)
        return rsl

    def load_model(self, model_dir=None):
        model_dir = self.training_opt["log_dir"] if model_dir is None else model_dir

        if os.path.isfile(model_dir + "/final_model_checkpoint.pth"):

            model_dir += "/final_model_checkpoint.pth"
            checkpoint = torch.load(model_dir, map_location="cpu")
            model_state = checkpoint["state_dict_best"]
            epoch = checkpoint["epoch"]
            print(f"Loading best model which was trained for {epoch} epochs")

        elif os.path.isfile(model_dir + "/latest_model_checkpoint.pth"):

            model_dir += "/latest_model_checkpoint.pth"
            checkpoint = torch.load(model_dir, map_location="cpu")
            model_state = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]
            print(f"Training hasn't finished, loading model trained for {epoch} epochs")
        else:
            print("No pretrained model")
            return

        print("Loading model from %s" % (model_dir))

        # checkpoint = torch.load(model_dir, map_location="cpu")
        # model_state = checkpoint["state_dict_best"]

        self.visual_model.load_state_dict(model_state["visual_model"])
        self.text_model.load_state_dict(model_state["text_model"])
        self.fusion.load_state_dict(model_state["fusion"])

        # if self.test_mode is True:
        #     self.adapter.load_state_dict(model_state["classifier"])

    def save_latest(self, epoch):
        model_weights = {}
        model_weights["visual_model"] = copy.deepcopy(self.visual_model.state_dict())
        model_weights["text_model"] = copy.deepcopy(self.text_model.state_dict())
        # if self.training_opt["phaseA"] is not True:
        model_weights["fusion"] = copy.deepcopy(self.fusion.state_dict())

        model_states = {"epoch": epoch, "state_dict": model_weights}

        model_dir = os.path.join(
            self.training_opt["log_dir"], "latest_model_checkpoint.pth"
        )
        torch.save(model_states, model_dir)

    def save_model(
        self, epoch, best_epoch, best_model_weights, best_acc, centroids=None
    ):

        model_states = {
            "epoch": epoch,
            "best_epoch": best_epoch,
            "state_dict_best": best_model_weights,
            "best_acc": best_acc,
            "centroids": centroids,
        }

        model_dir = os.path.join(
            self.training_opt["log_dir"], "final_model_checkpoint.pth"
        )

        torch.save(model_states, model_dir)

    def output_logits(self, openset=False):
        filename = os.path.join(
            self.training_opt["log_dir"], "logits_%s" % ("open" if openset else "close")
        )
        print("Saving total logits to: %s.npz" % filename)
        np.savez(
            filename,
            logits=self.total_logits.detach().cpu().numpy(),
            labels=self.total_labels.detach().cpu().numpy(),
            paths=self.total_paths,
        )


def check_config(conf, field):
    if field not in conf:
        return NonExistentTimeError
    return conf[field]
