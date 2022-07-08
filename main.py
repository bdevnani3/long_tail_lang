import os
import argparse
import pprint
from train0 import model
from torch import int16
from data_loader import clip_dataloaders as dataloader
import warnings
import yaml
from utils import *
from datetime import datetime

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

    return config


# ============================================================================
# LOAD CONFIGURATIONS
with open(args.cfg) as f:
    config = yaml.safe_load(f)
config = update(config, args)


test_mode = args.test
output_logits = args.output_logits
training_opt = config["training_opt"]
dataset = training_opt["dataset"]

dateTimeObj = datetime.now()
datetimestr = dateTimeObj.strftime("%d-%b-%Y-%H-%M-%S")
training_opt["log_dir"] = (
    "/nethome/bdevnani3/flash1/long_tail_lang/results/config_"
    + args.cfg.split("/")[-1].split(".yaml")[0]
    + "/"
    + datetimestr
)
print("Saving results at: {}".format(training_opt["log_dir"]))

training_opt["tf_folder"] = (
    "config_" + args.cfg.split("/")[-1].split(".yaml")[0] + "--" + datetimestr
)

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

    splits = ["train", "val", "test"]

    data = {
        x: dataloader.load_data(
            data_root=f"/nethome/bdevnani3/flash1/long_tail_lang/datasets/ImageNet_emb/RN50",
            phase=x,
            batch_size=training_opt["batch_size"],
            num_workers=training_opt["num_workers"],
        )
        for x in splits
    }

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

    data = {
        x: dataloader.load_data(
            data_root=f"/nethome/bdevnani3/flash1/long_tail_lang/datasets/ImageNet_emb/RN50",
            phase=x,
            batch_size=training_opt["batch_size"],
            num_workers=training_opt["num_workers"],
        )
        for x in splits
    }

    training_model = model(config, data, test=True)
    if args.save_feat in ["train", "val", "test"]:
        saveit = True
        test_split = args.save_feat
    else:
        saveit = False

    training_model.eval(phase=test_split, save_feat=saveit)

    if output_logits:
        training_model.output_logits()

print("ALL COMPLETED.")
