import enum
import os
import copy
import pickle
from re import template
from numpy.core.fromnumeric import cumprod
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from utils import *
from logger import Logger
import time
import numpy as np
import warnings
import pdb

# import clip
from clip import clip
from classes import CLASSES, CUSTOM_TEMPLATES

import torch
from torch.utils.tensorboard import SummaryWriter


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
        self.writer = SummaryWriter(log_dir="./runs/" + self.training_opt["tf_folder"])

        # Setup logger
        self.logger = Logger(self.training_opt["log_dir"])

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
        self.model_optim_params_list = []

        print("Using", torch.cuda.device_count(), "GPUs.")

        in_dim = self.model_opt["fusion"]["params"]["in_dim"]
        out_dim = self.model_opt["fusion"]["params"]["out_dim"]
        self.fusion = torch.nn.DataParallel(
            nn.Linear(in_dim, out_dim, bias=False)
        ).cuda()

        optim_params_fusion = self.model_opt["fusion"]["optim_params"]
        self.model_optim_params_list.append(
            {
                "params": self.fusion.parameters(),
                "lr": optim_params_fusion["lr"],
                "momentum": optim_params_fusion["momentum"],
                "weight_decay": optim_params_fusion["weight_decay"],
            }
        )

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

    def batch_forward(self, inputs, text_embeddings, phase="train", variation="0"):
        """
        This is a general single batch running function.
        """
        if variation == "0":
            self.image_linear_probe(inputs, text_embeddings, phase)
        elif variation == "1":
            self.image_plus_text_default(inputs, text_embeddings, phase)
        elif variation == "2":
            self.image_dot_text_default(inputs, text_embeddings, phase)
        elif variation == "3":
            self.image_plus_text_default_weighted(inputs, text_embeddings, phase)

    ###########################################################################
    ### Forward pass variations
    ###########################################################################

    def image_linear_probe(self, inputs, text_embeddings, phase="train"):
        """
        Linear layer of dimension 1024x1000 on top of image embeddings,
        no text used.

        inputs: batch_size x 1 x 1024
        text_embeddings: batch_size x 1 x 1024
        """

        inputs = inputs / inputs.norm(dim=-1, keepdim=True)

        if phase == "train":
            self.logits = self.fusion(inputs).squeeze(1).squeeze(-1)
        else:
            self.logits = self.fusion(inputs).squeeze(1).squeeze(-1)

    def image_plus_text_default(self, inputs, text_embeddings, phase="train"):
        """
        Add every image embedding to the text embedding that occurs at index 1,
        which happens to be "a photo of a". This is the simplest phrase and thus we
        refer to it as the default.

        inputs: batch_size x 1 x 1024
        text_embeddings: batch_size x 82 x 1024 (82 prompts)
        """

        inputs = inputs / inputs.norm(dim=-1, keepdim=True)
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)

        if phase == "train":
            combined_embedding = inputs.squeeze(1) + text_embeddings[:, 1, :]
            self.logits = self.fusion(combined_embedding).squeeze(-1)
        else:
            self.logits = self.fusion(inputs).squeeze(1).squeeze(-1)

    def image_dot_text_default(self, inputs, text_embeddings, phase="train"):
        """
        Multiply every image embedding to the text embedding that occurs at index 1,
        which happens to be "a photo of a". This is the simplest phrase and thus we
        refer to it as the default.

        inputs: batch_size x 1 x 1024
        text_embeddings: batch_size x 82 x 1024 (82 prompts)
        """

        inputs = inputs / inputs.norm(dim=-1, keepdim=True)
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)

        if phase == "train":
            combined_embedding = inputs.squeeze(1) * text_embeddings[:, 1, :]
            self.logits = self.fusion(combined_embedding).squeeze(-1)
        else:
            self.logits = self.fusion(inputs).squeeze(1).squeeze(-1)

    def image_plus_text_default_weighted(self, inputs, text_embeddings, phase="train"):
        """
        Add every image embedding * image_emb_weight to the text embedding * (1-image_emb_weight)
        that occurs at index 1, which happens to be "a photo of a".
        This is the simplest phrase and thus we refer to it as the default.

        inputs: batch_size x 1 x 1024
        text_embeddings: batch_size x 82 x 1024 (82 prompts)
        """

        inputs = inputs / inputs.norm(dim=-1, keepdim=True)
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)

        if phase == "train":
            combined_embedding = (
                float(self.training_opt["image_emb_weight"]) * inputs.squeeze(1)
            ) + (
                (1 - float(self.training_opt["image_emb_weight"]))
                * text_embeddings[:, 1, :]
            )
            self.logits = self.fusion(combined_embedding).squeeze(-1)
        else:
            self.logits = self.fusion(inputs).squeeze(1).squeeze(-1)

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

    def shuffle_batch(self, x, y, z):
        index = torch.randperm(x.size(0))
        # import pdb; pdb.set_trace()
        x = x[index]
        y = y[index]
        z = z[index]
        return x, y, z

    def train(self):

        # import pdb; pdb.set_trace()
        # When training the network
        print_str = ["Phase: train"]
        print_write(print_str, self.log_file)
        time.sleep(0.25)

        print_write(["Do shuffle??? --- ", self.do_shuffle], self.log_file)

        # Initialize best model
        self.best_model_weights = {}
        self.best_acc = 0.0
        self.best_epoch = 0
        # best_centroids = self.centroids

        end_epoch = self.training_opt["num_epochs"]

        # Loop over epochs
        for epoch in range(1, end_epoch + 1):

            self.train_epoch(epoch)

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

    def train_epoch(self, epoch):

        # import pdb; pdb.set_trace()

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
        self.curr_epoch_steps = 0
        for step, (
            inputs,
            labels,
            indexes,
            label_names,
            text_embeddings,
        ) in enumerate(self.data["train"]):

            self.train_step(
                step, epoch, inputs, labels, indexes, label_names, text_embeddings
            )

        # After every epoch, validation
        rsls = {"epoch": epoch}

        # TODO(bdevnani) Fix the logging for rsls
        rsls_train = self.eval_with_preds(self.total_preds, self.total_labels)
        # rsls_train = self.eval(phase="train")
        rsls_eval = self.eval(phase="val")
        rsls.update(rsls_train)
        rsls.update(rsls_eval)

        # Log results
        self.logger.log_acc(rsls)

        # Under validation, the best model need to be updated
        if self.eval_acc_mic_top1 > self.best_acc:
            self.best_epoch = epoch
            self.best_acc = self.eval_acc_mic_top1
            self.best_model_weights["fusion"] = copy.deepcopy(self.fusion.state_dict())

        print("===> Saving checkpoint")

        self.writer.add_scalar(
            "Loss/train", self.epoch_loss / self.curr_epoch_steps, epoch
        )
        self.writer.add_scalar("Acc/train", rsls_train["train_all"], epoch)

        self.writer.add_scalar("Acc/val_all", rsls_eval["val_all"], epoch)
        self.writer.add_scalar("Acc/val_many", rsls_eval["val_many"], epoch)
        self.writer.add_scalar("Acc/val_median", rsls_eval["val_median"], epoch)
        self.writer.add_scalar("Acc/val_low", rsls_eval["val_low"], epoch)

        self.save_latest(epoch)

    def train_step(
        self, step, epoch, inputs, labels, indexes, label_names, text_embeddings
    ):

        # import pdb; pdb.set_trace()
        # Break when step equal to epoch step
        if step >= self.epoch_steps:
            return
        if self.do_shuffle:
            inputs, labels, text_embeddings = self.shuffle_batch(
                inputs, labels, text_embeddings
            )
        inputs, labels, text_embeddings = (
            inputs.cuda(),
            labels.cuda(),
            text_embeddings.cuda(),
        )

        # If on training phase, enable gradients
        with torch.set_grad_enabled(True):

            # If training, forward with loss, and no top 5 accuracy calculation
            self.batch_forward(
                inputs,
                text_embeddings,
                phase="train",
                variation=self.training_opt["variation"],
            )

            # self.logits
            self.batch_loss(labels)

            self.batch_backward()

            # Tracking predictions
            _, preds = torch.max(self.logits, 1)
            self.total_preds.append(torch2numpy(preds))
            self.total_labels.append(torch2numpy(labels))

            # Output minibatch training results
            if step % self.training_opt["display_step"] == 0:

                minibatch_loss_feat = (
                    self.loss_feat.item()
                    if "FeatureLoss" in self.criterions.keys()
                    else None
                )
                minibatch_loss_perf = (
                    self.loss_perf.item()
                    if "PerformanceLoss" in self.criterions
                    else None
                )
                minibatch_loss_total = self.loss.item()
                minibatch_acc = mic_acc_cal(preds, labels)

                print_str = [
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
                self.curr_epoch_steps += 1

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

        import pdb

        pdb.set_trace()

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
            ) = shot_acc(normal_preds, normal_labels, self.data["train"])
            rsl["train_all"] += len(normal_preds) / n_total * n_top1
            rsl["train_many"] += len(normal_preds) / n_total * n_top1_many
            rsl["train_median"] += len(normal_preds) / n_total * n_top1_median
            rsl["train_low"] += len(normal_preds) / n_total * n_top1_low

        # Calculate mixup prediction accuracy
        if len(mixup_preds) > 0:
            mixup_preds, mixup_labels, mixup_ws = list(
                map(
                    np.concatenate,
                    [mixup_preds * 2, mixup_labels1 + mixup_labels2, mixup_ws],
                )
            )
            mixup_ws = np.concatenate([mixup_ws, 1 - mixup_ws])
            n_top1 = weighted_mic_acc_cal(mixup_preds, mixup_labels, mixup_ws)
            n_top1_many, n_top1_median, n_top1_low, = weighted_shot_acc(
                mixup_preds, mixup_labels, mixup_ws, self.data["train"]
            )
            rsl["train_all"] += len(mixup_preds) / 2 / n_total * n_top1
            rsl["train_many"] += len(mixup_preds) / 2 / n_total * n_top1_many
            rsl["train_median"] += len(mixup_preds) / 2 / n_total * n_top1_median
            rsl["train_low"] += len(mixup_preds) / 2 / n_total * n_top1_low

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

        # Iterate over dataset
        for inputs, labels, indexes, label_names, text_embeddings in tqdm(
            self.data[phase]
        ):
            inputs, labels, text_embeddings = (
                inputs.cuda(),
                labels.cuda(),
                text_embeddings.cuda(),
            )

            # If on training phase, enable gradients
            with torch.set_grad_enabled(False):

                # In validation or testing
                self.batch_forward(inputs, text_embeddings, phase=phase)
                self.total_logits = torch.cat((self.total_logits, self.logits))
                self.total_labels = torch.cat((self.total_labels, labels))

        probs, preds = F.softmax(self.total_logits.detach(), dim=1).max(dim=1)

        # Calculate the overall accuracy and F measurement
        self.eval_acc_mic_top1 = mic_acc_cal(
            preds[self.total_labels != -1], self.total_labels[self.total_labels != -1]
        )
        self.eval_f_measure = F_measure(
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
        ) = shot_acc(
            preds[self.total_labels != -1],
            self.total_labels[self.total_labels != -1],
            self.data["train"],
            acc_per_cls=True,
        )
        # Top-1 accuracy and additional string
        print_str = [
            "\n\n",
            "Phase: %s" % (phase),
            "\n\n",
            "Evaluation_accuracy_micro_top1: %.3f" % (self.eval_acc_mic_top1),
            "\n",
            "Averaged F-measure: %.3f" % (self.eval_f_measure),
            "\n",
            "Many_shot_accuracy_top1: %.3f" % (self.many_acc_top1),
            "Median_shot_accuracy_top1: %.3f" % (self.median_acc_top1),
            "Low_shot_accuracy_top1: %.3f" % (self.low_acc_top1),
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
                "{:.1f} \t {:.1f} \t {:.1f} \t {:.1f}".format(
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
        if not model_dir.endswith(".pth"):
            print("No pretrained model")

        print("Validation on the best model.")
        print("Loading model from %s" % (model_dir))

        checkpoint = torch.load(model_dir, map_location="cpu")
        model_state = checkpoint["state_dict_best"]

        self.fusion.load_state_dict(model_state["fusion"])

    def save_latest(self, epoch):
        model_weights = {}
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
        )
