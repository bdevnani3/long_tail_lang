import argparse
import os
import pprint
import warnings
from datetime import datetime

import yaml

from data_loader import dataloaders as dataloader
from train import model
from utils import *

##change your data root here
data_root = {"ImageNet": "./datasets/ImageNet/", "Places": "./datasets/Places/"}

parser = argparse.ArgumentParser()
parser.add_argument("--cfg", default=None, type=str)
parser.add_argument("--test", default=False, action="store_true")
parser.add_argument("--batch_size", type=int, default=None)
parser.add_argument("--test_open", default=False, action="store_true")
parser.add_argument("--output_logits", default=False)
parser.add_argument("--model_dir", type=str, default=None)
parser.add_argument("--save_feat", type=str, default="")

# KNN testing parameters
parser.add_argument("--knn", default=False, action="store_true")
parser.add_argument("--feat_type", type=str, default="cl2n")
parser.add_argument("--dist_type", type=str, default="l2")

# Learnable tau
parser.add_argument("--val_as_train", default=False, action="store_true")

args = parser.parse_args()


def update(config, args):
    # Change parameters
    config["training_opt"]["batch_size"] = get_value(
        config["training_opt"]["batch_size"], args.batch_size
    )

    # Testing with KNN
    if args.knn and args.test:
        training_opt = config["training_opt"]
        classifier_param = {
            "feat_dim": training_opt["feature_dim"],
            "num_classes": training_opt["num_classes"],
            "feat_type": args.feat_type,
            "dist_type": args.dist_type,
            "log_dir": training_opt["log_dir"],
        }
        classifier = {
            "def_file": "./models/KNNClassifier.py",
            "params": classifier_param,
            "optim_params": config["networks"]["classifier"]["optim_params"],
        }
        config["networks"]["classifier"] = classifier

    config["model_dir"] = args.model_dir
    if "eval_type" not in config["training_opt"]:
        config["training_opt"]["eval_type"] = "image"

    return config


# ============================================================================
# LOAD CONFIGURATIONS


def get_log_and_tf_dir(cfg):

    dateTimeObj = datetime.now()
    datetimestr = dateTimeObj.strftime("%d-%b-%Y-%H-%M-%S")
    cfg = cfg.split(".yaml")[0]
    cfg = cfg.split("configs/")[1]
    cfg = "_".join(cfg.split("/"))
    log_cfg = (
        f"/nethome/bdevnani3/flash1/long_tail_lang/results/config_"
        + cfg
        + "/"
        + datetimestr
    )

    tf_cfg = f"config_" + cfg + "--" + datetimestr

    return log_cfg, tf_cfg


with open(args.cfg) as f:
    config = yaml.safe_load(f)
config = update(config, args)

test_mode = args.test
test_open = args.test_open
if test_open:
    test_mode = True
output_logits = args.output_logits
training_opt = config["training_opt"]
dataset = training_opt["dataset"]

if "log_dir" not in training_opt:
    training_opt["log_dir"], training_opt["tf_folder"] = get_log_and_tf_dir(args.cfg)
    print("Saving results at: {}".format(training_opt["log_dir"]))

    if not os.path.isdir(training_opt["log_dir"]):
        os.makedirs(training_opt["log_dir"])

    copy_current_codebase_to_path(training_opt["log_dir"] + "/src")

print("Loading dataset from: %s" % data_root[dataset.rstrip("_LT")])
pprint.pprint(config)


def split2phase(split):
    if split == "train" and args.val_as_train:
        return "train_val"
    else:
        return split


if not test_mode:

    sampler_defs = training_opt["sampler"]
    if sampler_defs:
        if sampler_defs["type"] == "ClassAwareSampler":
            sampler_dic = {
                "sampler": source_import(sampler_defs["def_file"]).get_sampler(),
                "params": {"num_samples_cls": sampler_defs["num_samples_cls"]},
            }
        elif sampler_defs["type"] in [
            "MixedPrioritizedSampler",
            "ClassPrioritySampler",
        ]:
            sampler_dic = {
                "sampler": source_import(sampler_defs["def_file"]).get_sampler(),
                "params": {
                    k: v
                    for k, v in sampler_defs.items()
                    if k not in ["type", "def_file"]
                },
            }
    else:
        sampler_dic = None

    # Because of weird ImageNet set up
    splits = ["train", "train_plain", "val"]
    if dataset not in ["ImageNet"]:
        splits.append("test")

    data = {}

    for x in splits:
        d = dataloader.load_data(
            data_root=data_root[dataset.rstrip("_LT")],
            dataset=dataset,
            phase=split2phase(x),
            batch_size=training_opt["batch_size"],
            sampler_dic=sampler_dic,
            num_workers=training_opt["num_workers"],
            boosted=config["dataset_variant"] == "boosted",
        )
        data[x] = d[0]
        if x == "train":
            data[x + "_ltcount"] = d[1]

    # CLIP dataloader
    # data = {
    #     x: dataloader.load_data(
    #         data_root=f"/nethome/bdevnani3/flash1/long_tail_lang/datasets/ImageNet_emb/RN50",
    #         phase=x,
    #         batch_size=training_opt["batch_size"],
    #         num_workers=training_opt["num_workers"],
    #     )
    #     for x in splits
    # }
    many_shot_thr = 100
    low_shot_thr = 20
    data["label_categorization"] = {"few": [], "many": [], "medium": []}
    for i in data["train_ltcount"]:
        if data["train_ltcount"][i] > many_shot_thr:
            data["label_categorization"]["many"].append(i)
        elif data["train_ltcount"][i] < low_shot_thr:
            data["label_categorization"]["few"].append(i)
        else:
            data["label_categorization"]["medium"].append(i)

    print(
        "Label categorization: \n Few: {} \n Medium: {} \n Many: {}".format(
            len(data["label_categorization"]["few"]),
            len(data["label_categorization"]["medium"]),
            len(data["label_categorization"]["many"]),
        )
    )

    training_model = model(config, data, test=False)

    training_model.train()

else:

    warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

    print(
        "Under testing phase, we load training data simply to calculate \
           training data number for each class."
    )

    splits = ["train", "val", "test"]
    test_split = "test"

    # Because of weird ImageNet set up
    if "ImageNet" == training_opt["dataset"]:
        splits = ["train", "val"]
        test_split = "val"

    if args.knn or True:
        splits.append("train_plain")

    data = {}

    for x in splits:
        d = dataloader.load_data(
            data_root=data_root[dataset.rstrip("_LT")],
            dataset=dataset,
            phase=x,
            batch_size=training_opt["batch_size"],
            sampler_dic=None,
            test_open=test_open,
            num_workers=training_opt["num_workers"],
            shuffle=False,
        )
        data[x] = d[0]
        if x == "train":
            data[x + "_ltcount"] = d[1]

    training_model = model(config, data, test=True)
    # training_model.load_model()
    # training_model.load_model(args.model_dir)
    if args.save_feat in ["train_plain", "val", "test"]:
        saveit = True
        test_split = args.save_feat
    else:
        saveit = False

    training_model.eval(phase=test_split, openset=test_open, save_feat=saveit)

    if output_logits:
        training_model.output_logits(openset=test_open)

print("ALL COMPLETED.")
