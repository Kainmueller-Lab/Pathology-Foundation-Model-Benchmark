import argparse
import os
from datetime import datetime
from time import time

import kornia
import numpy as np
import torch
import wandb
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from benchmark.augmentations.augmentations import Augmenter
from benchmark.eval import Eval
from benchmark.heads.dpt import DPT
from benchmark.heads.unetr import UnetR
from benchmark.models.hovernext import HoverNext
from benchmark.models.simple_segmentation_model import (
    MockModel,
    SimpleSegmentationModel,
)
from benchmark.utils.init_dist import init_distributed
from benchmark.utils.utils import (
    EMAInverseClassFrequencyLoss,
    ExcludeClassLossWrapper,
    get_weighted_sampler,
    maybe_resize,
    prep_datasets,
    save_imgs_for_debug,
)

os.environ["OMP_NUM_THREADS"] = "1"
torch.backends.cudnn.benchmark = True
MAX_EARLY_STOPPING = 5000


def make_log_dirs(cfg, split="train"):
    """Prepare directories for writing training infos."""
    log_dir = os.path.join(cfg.experiment_path, cfg.experiment, split)
    checkpoint_path = os.path.join(log_dir, "checkpoints")
    snap_dir = os.path.join(log_dir, "snaps")
    os.makedirs(snap_dir, exist_ok=True)
    os.makedirs(checkpoint_path, exist_ok=True)
    cfg.writer_dir = os.path.join(log_dir, "summary", str(time()))
    os.makedirs(cfg.writer_dir, exist_ok=True)

    print(f"Saving logs at {log_dir}")
    return log_dir, checkpoint_path, snap_dir


def create_dataloaders(cfg, train_dset, val_dset, test_dset, train_sampler=None):
    train_dataloader = DataLoader(
        train_dset,
        batch_size=cfg.dataset.batch_size,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        prefetch_factor=8 if cfg.multiprocessing else None,
        num_workers=cfg.num_workers - 1 if cfg.multiprocessing else 0,
        sampler=train_sampler if cfg.dataset.uniform_class_sampling else None,
        shuffle=not cfg.dataset.uniform_class_sampling,
    )
    val_dataloader = DataLoader(val_dset, batch_size=cfg.dataset.batch_size, pin_memory=True, num_workers=0)
    test_dataloader = DataLoader(
        test_dset,
        batch_size=cfg.dataset.batch_size,
        pin_memory=True,
        num_workers=0,
        shuffle=False,
    )
    return train_dataloader, val_dataloader, test_dataloader


def build_lr_scheduler(cfg, optimizer):
    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        [  # linear warmup
            torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=cfg.optimizer.warmup_steps,
            ),
            getattr(torch.optim.lr_scheduler, cfg.scheduler.name)(optimizer, **cfg.scheduler.params),
        ],
        milestones=[cfg.optimizer.warmup_steps],
    )
    return lr_scheduler


def initialize_dist(model):
    if "RANK" in os.environ:
        LOCAL_RANK, LOCAL_WORLD_SIZE, RANK, WORLD_SIZE = init_distributed()
        device = torch.device(f"cuda:{LOCAL_RANK}")
        model = model.cuda()
        model = DDP(
            model,
            device_ids=[LOCAL_RANK],
            output_device=LOCAL_RANK,
            find_unused_parameters=True,
        )
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        WORLD_SIZE = 1
        model = model.to(device)
        LOCAL_RANK = 0
    print(f"Device: {device}")
    return model, device, WORLD_SIZE, LOCAL_RANK


def create_loss_fn(cfg, label_dict):
    if hasattr(torch.nn, cfg.loss_fn.name):
        loss_fn = getattr(torch.nn, cfg.loss_fn.name)(**cfg.loss_fn.params)
        if hasattr(cfg.loss_fn, "exclude_classes"):
            print(f"Excluding classes from loss calculation: {cfg.loss_fn.exclude_classes}")
            loss_fn = EMAInverseClassFrequencyLoss(
                loss_fn=loss_fn,
                num_classes=len(label_dict),
                exclude_class=(cfg.loss_fn.exclude_classes if hasattr(cfg.loss_fn, "exclude_classes") else None),
                class_weighting=(cfg.loss_fn.class_weighting if hasattr(cfg.loss_fn, "class_weighting") else False),
            )
    elif hasattr(kornia.losses, cfg.loss_fn.name):
        loss_fn = getattr(kornia.losses, cfg.loss_fn.name)(**cfg.loss_fn.params)
    return loss_fn


def save_config(cfg):
    with open(os.path.join(cfg.experiment_path, cfg.experiment, "config.yaml"), "w") as f:
        OmegaConf.save(config=cfg, f=f)


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


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
    log_dir, checkpoint_path, snap_dir = make_log_dirs(cfg)
    best_model_path = os.path.join(checkpoint_path, "best_model.pth")
    train_dset, val_dset, test_dset, label_dict = prep_datasets(cfg)
    print_ds_stats(train_dset, val_dset, test_dset, label_dict)
    model_wrapper = eval(cfg.model.model_wrapper)
    model = model_wrapper(
        model_name=cfg.model.backbone,
        num_classes=len(label_dict),
        do_ms_aug=getattr(cfg.model, "do_ms_aug", False),  # Default to False if not specified
    )
    if cfg.model.unfreeze_backbone:
        model.unfreeze_model()

    evaluater = Eval(
        label_dict,
        instance_level=True,
        pixel_level=False,
        save_dir=os.path.join(log_dir, "validation_results"),
        fname="validation_metrics.csv",
    )
    if hasattr(cfg, "augmentations"):
        augment_fn = Augmenter(cfg.augmentations, data_keys=["input", "mask", "mask"])
    else:

        def augment_fn(img, mask, instance_mask):
            return img, mask, instance_mask

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

    scaler = torch.amp.GradScaler(device.type)

    if cfg.dataset.uniform_class_sampling:
        if cfg.dataset.sample_excluded_classes:
            classes = [int(k) for k in label_dict.keys() if int(k) not in cfg.loss_fn.exclude_classes]
        else:
            classes = [int(k) for k in label_dict.keys()]
        train_sampler = get_weighted_sampler(train_dset, classes=classes)

    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(
        cfg, train_dset, val_dset, test_dset, train_sampler=train_sampler
    )

    loss_fn = create_loss_fn(cfg, label_dict)

    # this maybe needs to change depending on how the model_wrapper is implemented
    optimizer = getattr(torch.optim, cfg.optimizer.name)(list(model.parameters()), **cfg.optimizer.params)

    lr_scheduler = build_lr_scheduler(cfg, optimizer)

    save_config(cfg)

    # training pipeline
    step = 0
    steps_since_last_best = 0
    loss_history = []
    primary_metric_history = []
    print("Start training")
    np.random.seed(cfg.seed if hasattr(cfg, "seed") else time())
    loss_tmp = []
    while step < cfg.training_steps and steps_since_last_best < cfg.early_stopping:
        model.train()  # maybe change depending on how the model_wrapper is implemented
        for sample_dict in train_dataloader:
            img = sample_dict["image"].float()
            semantic_mask = sample_dict["semantic_mask"]
            instance_mask = sample_dict.get("instance_mask", None)
            img = img.to(device)
            semantic_mask = semantic_mask.to(device)
            instance_mask = instance_mask.to(device)
            img_aug, semantic_mask_aug, instance_mask_aug = augment_fn(img, semantic_mask, instance_mask)
            semantic_mask_aug = semantic_mask_aug.squeeze(1)  # Remove the extra dimension
            instance_mask_aug = instance_mask_aug.squeeze(1)  # Remove the extra dimension
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                pred_mask = model(img_aug)
                pred_mask = maybe_resize(pred_mask, semantic_mask_aug)

                loss = loss_fn(pred_mask, semantic_mask_aug.long())
            if not torch.isnan(loss):
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                scaler.step(optimizer)
                scaler.update()

                lr_scheduler.step()
                optimizer.zero_grad()
                loss_tmp.append(loss.cpu().detach().numpy())
                if logging:
                    wandb.log(
                        {
                            "train_loss": loss_tmp[-1] / float(WORLD_SIZE),
                            "lr": optimizer.param_groups[0]["lr"],
                        },
                        step=step,
                    )
                if step % cfg.log_interval == 0:
                    print(f"Step {step}, loss {np.mean(loss_tmp) / float(WORLD_SIZE)}")

            # log at cfg.val_interval or at the end of training
            if step > 0 and (step % cfg.val_interval == 0 or step == cfg.training_steps - 1):
                # validating
                evaluater.save_dir = os.path.join(log_dir, "validation_results")
                evaluater.fname = f"validation_metrics_step_{step}.csv"
                logging_dict, classwise_dict = evaluater.compute_metrics(model, val_dataloader, device)
                logging_dict = {"validation/" + k: v for k, v in logging_dict.items()}
                classwise_dict = {k + "_val": v for k, v in classwise_dict.items()}
                logging_dict["train_loss"] = np.mean(loss_tmp) / float(WORLD_SIZE)
                logging_dict["lr"] = optimizer.param_groups[0]["lr"]
                loss_history.append(logging_dict["train_loss"])
                if logging:
                    wandb.log(logging_dict, step=step)
                    wandb.log(classwise_dict, step=step)
                if logging or "RANK" not in os.environ:
                    if cfg.save_all_ckpts:
                        model_path = os.path.join(checkpoint_path, f"checkpoint_step_{step}.pth")
                        torch.save(model.state_dict(), model_path)
                    if cfg.save_snapshots and step % (10 * cfg.val_interval) == 0:
                        save_imgs_for_debug(
                            snap_dir,
                            step,
                            img,
                            pred_mask,
                            semantic_mask,
                            img_aug,
                            semantic_mask_aug,
                            instance_mask=None,
                            instance_mask_aug=None,
                        )
                if hasattr(cfg, "primary_metric"):
                    primary_metric_history.append(logging_dict["validation/" + cfg.primary_metric])
                    steps_since_last_best += cfg.val_interval
                if hasattr(cfg, "primary_metric") and (logging or "RANK" not in os.environ):
                    if max(primary_metric_history) == logging_dict["validation/" + cfg.primary_metric]:
                        torch.save(model.state_dict(), best_model_path)
                        best_checkpoint_step = step
                        steps_since_last_best = 0
                        print(f"Found new BEST model at step {step}, loss {np.mean(loss_tmp) / float(WORLD_SIZE)}")

                loss_tmp = []
                model.train()
            step += 1

    # Testing
    if hasattr(cfg, "primary_metric"):
        model.load_state_dict(torch.load(best_model_path))
        evaluater.save_dir = os.path.join(log_dir, "test_results")
        evaluater.fname = "test_metrics_best_model.csv"
        logging_dict, classwise_dict = evaluater.compute_metrics(model, test_dataloader, device)
    if logging:
        logging_dict["best_checkpoint_step"] = best_checkpoint_step
        logging_dict = {f"test/{k}": v for k, v in logging_dict.items()}
        classwise_dict = {k + "_test": v for k, v in classwise_dict.items()}
        wandb.log(logging_dict)
        wandb.log(classwise_dict)
    wandb.finish()
    return loss_history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--job_id", type=int, default=0)
    parser.add_argument("--model_name", type=str, default="uni")

    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    cfg.model.backbone = args.model_name
    cfg.experiment = (
        f"{cfg.experiment}_{cfg.model.backbone}_{datetime.now().strftime('%d%m_%H%M')}_{cfg.dataset.name}_{args.job_id}"
    )
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
