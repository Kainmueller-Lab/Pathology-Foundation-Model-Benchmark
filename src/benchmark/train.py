import os
import torch
from torch.utils.data import DataLoader
import numpy as np
from time import time
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import wandb
from benchmark.init_dist import init_distributed
from benchmark.utils import prep_datasets
from benchmark.eval_utils import evaluate_model
from benchmark.model_wrapper import *
import argparse

os.environ["OMP_NUM_THREADS"] = "1"
torch.backends.cudnn.benchmark = True


def train(cfg):
    # prepare directories for writing training infos
    log_dir = os.path.join(cfg.experiment_path, cfg.experiment,'train')
    checkpoint_path = os.path.join(log_dir, "checkpoints")
    snap_dir = os.path.join(log_dir,'snaps')
    os.makedirs(snap_dir,exist_ok=True)
    os.makedirs(checkpoint_path,exist_ok=True)
    cfg.writer_dir = os.path.join(log_dir,'summary',str(time()))
    os.makedirs(cfg.writer_dir,exist_ok=True)

    train_dset, val_dset, test_dset, label_dict = prep_datasets(cfg)
    model_wrapper = eval(cfg.model.model_wrapper)
    model = model_wrapper(
        model_name=cfg.model.backbone, num_classes=len(label_dict))
    model = model.cuda()

    # initialize dist
    if 'RANK' in os.environ:
        LOCAL_RANK, LOCAL_WORLD_SIZE, RANK, WORLD_SIZE = init_distributed()
        device = torch.device(f'cuda:{LOCAL_RANK}')
        model = DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK, find_unused_parameters=True)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        WORLD_SIZE = 1

    # log only on rank 0
    if 'RANK' in os.environ and LOCAL_RANK==0 or 'RANK' not in os.environ:
        logging = True
        wandb.init(
            name=cfg.experiment,
            project=cfg.project,
            entity="kainmueller-lab",
            config=OmegaConf.to_container(cfg),
            dir=cfg.writer_dir,
            mode="offline" if "mock" in cfg.experiment.lower() else "online",
            )
    else:
        logging = False

    scaler = torch.cuda.amp.GradScaler()

    def worker_init_fn(worker_id):                                                          
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    train_dataloader = DataLoader(
        train_dset, batch_size=cfg.dataset.batch_size, pin_memory=True,
        worker_init_fn=worker_init_fn,
        prefetch_factor=8 if cfg.multiprocessing else None,
        num_workers=cfg.num_workers-1 if cfg.multiprocessing else 0,
    )
    val_dataloader = DataLoader(val_dset, batch_size=cfg.dataset.batch_size, pin_memory=True, num_workers=0)
    test_dataloader = DataLoader(test_dset, batch_size=cfg.dataset.batch_size, pin_memory=True, num_workers=4)
    loss_fn = getattr(torch.nn, cfg.loss_fn.name)(**cfg.loss_fn.params)

    # this maybe needs to change depending on how the model_wrapper is implemented
    optimizer = getattr(torch.optim, cfg.optimizer.name)(
        list(model.parameters()),**cfg.optimizer.params
    )

    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, [ # linear warmup
            torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.1, end_factor=1.0,
                total_iters=cfg.optimizer.warmup_steps
            ),
        getattr(torch.optim.lr_scheduler, cfg.scheduler.name)(
            optimizer, **cfg.scheduler.params
        )], milestones=[cfg.optimizer.warmup_steps]
    )
    # save parameters
    with open(os.path.join(cfg.experiment_path, cfg.experiment, 'config.yaml'), 'w') as f:
        OmegaConf.save(config=cfg, f=f)

    # training pipeline
    validation_loss = []

    step = 0
    loss_history = []
    print("Start training")
    np.random.seed(cfg.seed if hasattr(cfg, "seed") else time())
    loss_tmp = []
    while step < cfg.training_steps:
        model.train() # maybe change depending on how the model_wrapper is implemented
        for sample_dict in train_dataloader:
            img = sample_dict["image"].float()
            semantic_mask = sample_dict["semantic_mask"]
            instance_mask = sample_dict.get("instance_mask", None)
            step += 1
            img = img.cuda()
            semantic_mask = semantic_mask.cuda()
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                pred_mask = model(img)
                loss = loss_fn(pred_mask, semantic_mask.long())
            if not torch.isnan(loss):
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step()
                optimizer.zero_grad()
                loss_tmp.append(loss.cpu().detach().numpy())
                if logging:
                    wandb.log({
                        "train_loss": loss_tmp[-1] / float(WORLD_SIZE),
                        "lr": optimizer.param_groups[0]["lr"],
                    }, step=step)
                print(f"Step {step}, loss {np.mean(loss_tmp) / float(WORLD_SIZE)}")
            if step % cfg.log_interval == 0:
                # validation
                logging_dict = evaluate_model(model, val_dataloader, label_dict)
                logging_dict["train_loss"] = np.mean(loss_tmp) / float(WORLD_SIZE)
                logging_dict["lr"] = optimizer.param_groups[0]["lr"]
                loss_history.append(logging_dict["train_loss"])
                loss_tmp = []
                if logging:
                    wandb.log(logging_dict, step=step)
                    model_path = os.path.join(
                        checkpoint_path, f"checkpoint_step_{step}.pth"
                    )
                    torch.save(model.state_dict(), model_path)
    wandb.finish()
    return loss_history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    train(cfg)
