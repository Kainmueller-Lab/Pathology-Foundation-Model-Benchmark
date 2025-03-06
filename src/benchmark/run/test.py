import argparse
import os
from datetime import datetime
from time import time

import numpy as np
import torch
import wandb
from omegaconf import OmegaConf

from benchmark.heads.dpt import DPT
from benchmark.heads.unetr import UnetR
from benchmark.models.hovernext import HoverNext
from benchmark.models.simple_segmentation_model import (
    MockModel,
    SimpleSegmentationModel,
)
from benchmark.run.eval import Eval
from benchmark.run.train import create_dataloaders, initialize_dist, make_log_dirs
from benchmark.utils.utils import prep_datasets

os.environ["OMP_NUM_THREADS"] = "1"
torch.backends.cudnn.benchmark = True
MAX_EARLY_STOPPING = 5000


def print_ds_stats(train_dset, val_dset, test_dset, label_dict):
    print("-------- printing dataset stats -------")
    print(f"Number of classes: {len(label_dict.keys())}")
    print(f"Unique samples: {len(label_dict.keys())}")

    def print_ds(ds):
        print(f"Number of samples: {len(ds)}")

    print("Train")
    print_ds(train_dset)
    print("Val")
    print_ds(val_dset)
    print("Test")
    print_ds(test_dset)


def train(cfg):
    log_dir, _, snap_dir = make_log_dirs(cfg, split="test")
    best_model_path = cfg.best_model_path
    train_dset, val_dset, test_dset, label_dict = prep_datasets(cfg)
    print_ds_stats(train_dset, val_dset, test_dset, label_dict)
    model_wrapper = eval(cfg.model.model_wrapper)
    model = model_wrapper(
        model_name=cfg.model.backbone,
        num_classes=len(label_dict),
        do_ms_aug=getattr(cfg.model, "do_ms_aug", False),  # Default to False if not specified
    )

    evaluater = Eval(
        label_dict,
        instance_level=True,
        pixel_level=False,
        save_dir=os.path.join(log_dir, "validation_results"),
        fname="validation_metrics.csv",
    )
    model, device, WORLD_SIZE, LOCAL_RANK = initialize_dist(model)

    # log only on rank 0
    if "RANK" in os.environ and LOCAL_RANK == 0 or "mock" not in cfg.experiment.lower():
        logging = True
        wandb.init(
            name=cfg.experiment,
            project=cfg.project,
            entity="kainmueller-lab",
            config=OmegaConf.to_container(cfg),
            dir=cfg.writer_dir,
            mode=cfg.get("wandb_mode", "online"),
        )
    else:
        logging = False

    _, _, test_dataloader = create_dataloaders(cfg, train_dset, val_dset, test_dset, train_sampler=None)
    np.random.seed(cfg.seed if hasattr(cfg, "seed") else time())

    # Testing
    if hasattr(cfg, "primary_metric"):
        model.load_state_dict(torch.load(best_model_path))
        evaluater.save_dir = os.path.join(log_dir, "test_results")
        evaluater.fname = "test_metrics_best_model.csv"
        logging_dict, classwise_dict = evaluater.compute_metrics(
            model, test_dataloader, device, save_preds=True, snap_dir=snap_dir
        )

    if logging:
        logging_dict = {f"test/{k}": v for k, v in logging_dict.items()}
        classwise_dict = {k + "_test": v for k, v in classwise_dict.items()}
        wandb.log(logging_dict)
        wandb.log(classwise_dict)

    wandb.finish()
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--job_id", type=int, default=0)
    parser.add_argument("--model_name", type=str, default="uni")

    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    cfg.model.backbone = args.model_name
    cfg.experiment = f"Test_{cfg.experiment}_{cfg.model.backbone}_{datetime.now().strftime('%d%m_%H%M')}_{cfg.dataset.name}_{args.job_id}"
    print(f"Experiment name: {cfg.experiment}")
    if not hasattr(cfg, "early_stopping"):
        cfg.early_stopping = MAX_EARLY_STOPPING
    if not hasattr(cfg, "save_snapshots"):
        cfg.save_snapshots = False
    if not hasattr(cfg, "save_all_ckpts"):
        cfg.save_all_ckpts = False
    if not hasattr(cfg.model, "do_ms_aug"):
        cfg.model.do_ms_aug = False
    train(cfg)
