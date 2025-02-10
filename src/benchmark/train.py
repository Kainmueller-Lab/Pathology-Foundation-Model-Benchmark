import os
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
from time import time
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import wandb
from benchmark.augmentations import Augmenter
from benchmark.init_dist import init_distributed
from benchmark.utils import prep_datasets, ExcludeClassLossWrapper, EMAInverseClassFrequencyLoss
from benchmark.eval import Eval
from benchmark.simple_segmentation_model import *
from benchmark.mask2former import *
import argparse
import h5py

from datetime import datetime
from matplotlib import pyplot as plt

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
    model = model_wrapper(model_name=cfg.model.backbone, num_classes=len(label_dict))
    if cfg.model.unfreeze_backbone:
        model.unfreeze_model()
    
    evaluater = Eval(
        label_dict, instance_level=True, pixel_level=False,
        save_dir=os.path.join(log_dir, "validation_results"),
        fname="validation_metrics.csv"
    )
    if hasattr(cfg, "augmentations"):
        augment_fn = Augmenter(cfg.augmentations, data_keys=["input", "mask", "mask"])
    else:
        def augment_fn(img, mask, instance_mask):
            return img, mask, instance_mask
    # metric_names = ["precision_macro", "recall_macro", "f1_score_macro", "accuracy_macro",
    #     "precision_micro", "recall_micro", "f1_score_micro", "accuracy_micro", "classwise_metrics"]

    # initialize dist
    if 'RANK' in os.environ:
        LOCAL_RANK, LOCAL_WORLD_SIZE, RANK, WORLD_SIZE = init_distributed()
        device = torch.device(f'cuda:{LOCAL_RANK}')
        model = model.cuda()
        model = DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK, find_unused_parameters=True)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        WORLD_SIZE = 1
        model = model.to(device)
    print(f"Device: {device}")

    # log only on rank 0
    if 'RANK' in os.environ and LOCAL_RANK==0 or "mock" not in cfg.experiment.lower():
        logging = True
        wandb.init(
            name=cfg.experiment,
            project=cfg.project,
            entity="kainmueller-lab",
            config=OmegaConf.to_container(cfg),
            dir=cfg.writer_dir,
            mode=cfg.get('wandb_mode', 'online'),
            )
    else:
        logging = False

    scaler = torch.amp.GradScaler(device.type)

    def worker_init_fn(worker_id):                                                          
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    train_dataloader = DataLoader(
        train_dset, batch_size=cfg.dataset.batch_size, pin_memory=True,
        worker_init_fn=worker_init_fn,
        prefetch_factor=8 if cfg.multiprocessing else None,
        num_workers=cfg.num_workers-1 if cfg.multiprocessing else 0,
        shuffle=True,
    )
    val_dataloader = DataLoader(
        val_dset, batch_size=cfg.dataset.batch_size, pin_memory=True, num_workers=0
    )
    test_dataloader = DataLoader(
        test_dset, batch_size=cfg.dataset.batch_size, pin_memory=True, num_workers=0
    )
    loss_fn = getattr(torch.nn, cfg.loss_fn.name)(**cfg.loss_fn.params)
    if hasattr(cfg.loss_fn, "exclude_classes"):
        print(f"Excluding classes from loss calculation: {cfg.loss_fn.exclude_classes}")
        loss_fn = EMAInverseClassFrequencyLoss(
            loss_fn=loss_fn, num_classes=len(label_dict),
            exclude_class=cfg.loss_fn.exclude_classes if hasattr(cfg.loss_fn, "exclude_classes") else None,
            class_weighting=cfg.loss_fn.class_weighting if hasattr(cfg.loss_fn, "class_weighting") else False
        )

    for name, param in model.named_parameters():
        if not param.requires_grad:
            print(f"Parameter {name} does not require gradients!")

    print('MODEL PARAMS', sum(1 for p in model.parameters()))
    print('MODEL PARAMS REQUIRE GRAD', sum(p.requires_grad for p in model.parameters()))

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
    step = 0
    loss_history = []
    primary_metric_history = []
    print("Start training")
    np.random.seed(cfg.seed if hasattr(cfg, "seed") else time())
    loss_tmp = []

    outdir = "outputs/test_m2f_hf_{date}".format(date=datetime.now().strftime("%Y%m%d%H%M"))
    os.makedirs(outdir, exist_ok=True)

    while step < cfg.training_steps:
        model.train() # maybe change depending on how the model_wrapper is implemented
        for sample_dict in train_dataloader:
            img = sample_dict["image"].float()
            semantic_mask = sample_dict["semantic_mask"]
            instance_mask = sample_dict.get("instance_mask", None)
            img = img.to(device)
            semantic_mask = semantic_mask.to(device)
            instance_mask = instance_mask.to(device)
            
            img_aug, semantic_mask_aug, instance_mask_aug = augment_fn(
                 img, semantic_mask, instance_mask
            )

            with torch.autocast(device_type=device.type, dtype=torch.float16):
                if cfg.model.model_wrapper == "Mask2FormerModel":
                    # make tensors a lists of images
                    img_list = list(img_aug)
                    semantic_mask_list = list(semantic_mask_aug)
                    input_dict = model.image_processor(img_list, semantic_mask_list, return_tensors="pt")

                    print('PIX VAL SHAPE:', input_dict["pixel_values"].shape)
                    print('PIX MASK SHAPE:', input_dict["pixel_mask"].shape)
                    print('MASK LABELS SHAPE:', [x.shape for x in input_dict["mask_labels"]])
                    print('CLASS LABELS SHAPE:', [x.shape for x in input_dict["class_labels"]])

                    # make plot with subfigures for pixel values, pixel mask and all mask labels
                    f, a = plt.subplots(2, 10)
                    f.set_size_inches(25, 5)
                    a[0,0].imshow(input_dict["pixel_values"][0].permute(1, 2, 0).cpu().numpy())
                    a[0,1].imshow(input_dict["pixel_mask"][0].cpu().numpy())
                    for i in range(len(input_dict["mask_labels"])):
                        a[1,i].imshow(input_dict["mask_labels"][0][i].cpu().numpy()) 
                    plt.show()
                    plt.savefig(os.path.join(outdir, f"inputs_{step}.png"), bbox_inches="tight")

                    input_dict = {
                        "pixel_values": input_dict["pixel_values"].to(device),
                        "pixel_mask": input_dict["pixel_mask"].to(device),
                        "mask_labels": [x.to(device) for x in input_dict["mask_labels"]],
                        "class_labels": [x.to(device) for x in input_dict["class_labels"]]
                    }

                    print(f"IMG requires_grad:", input_dict["pixel_values"].requires_grad)
                    input_dict["pixel_values"].to(device)
                    pred_mask = model(input_dict)
                    pred_mask = torch.stack(pred_mask)
                    print('pred_mask requires_grad', pred_mask.requires_grad) # False --> needs to be True

                    # reorganize predicted mask to be tensor of shape [batch_size, num_classes, height, width]
                    pred_mask = F.one_hot(pred_mask, num_classes=len(label_dict))
                    pred_mask = pred_mask.permute(0, 3, 1, 2).contiguous()
                    pred_mask = pred_mask.float()
                    pred_mask.requires_grad_() # maybe doesn't make sense to do it here, if before it is not attached to the graph?
                    print('pred_mask requires_grad', pred_mask.requires_grad)
                else:
                    pred_mask = model(img_aug)

                print('PRED MASK SHAPE', pred_mask.shape) # torch.Size([bs, num_classes, 224, 224])
                print('SEMANTIC MASK SHAPE', semantic_mask_aug.shape) # torch.Size([2, 224, 224])
                print('PRED MASK DTYPE', pred_mask.dtype) # torch.float32
                print('SEMANTIC MASK DTYPE', semantic_mask_aug.dtype) # torch.uint8

                loss = loss_fn(pred_mask, semantic_mask_aug.long())
                print('Loss', loss)
                print(f"Loss requires_grad: {loss.requires_grad}")
            if not torch.isnan(loss):
                scaler.scale(loss).backward()
                for param in model.parameters():
                    if param.grad is None:
                        # print(f"Param {param} has no gradient after backward")
                        # Result: no params have grad!!!
                        pass
                    else:
                        print(f"Param {param} has gradient after backward")
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            print(f"NaN or Inf found in gradients for {param}")
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
            # log at cfg.log_interval or at the end of training
            if (step % cfg.log_interval == 0) or (step == cfg.training_steps-1):
                # validation
                evaluater.save_dir = os.path.join(log_dir, "validation_results")
                evaluater.fname = f"validation_metrics_step_{step}.csv"
                logging_dict, classwise_dict = evaluater.compute_metrics(
                    model, val_dataloader, device
                )
                logging_dict = {"validation/"+k: v for k, v in logging_dict.items()}
                classwise_dict = {k+"_val": v for k, v in classwise_dict.items()}
                logging_dict["train_loss"] = np.mean(loss_tmp) / float(WORLD_SIZE)
                logging_dict["lr"] = optimizer.param_groups[0]["lr"]
                loss_history.append(logging_dict["train_loss"])
                loss_tmp = []
                model.train()
                if logging:
                    wandb.log(logging_dict, step=step)
                    wandb.log(classwise_dict, step=step)
                if logging or 'RANK' not in os.environ:
                    model_path = os.path.join(
                        checkpoint_path, f"checkpoint_step_{step}.pth"
                    )
                    torch.save(model.state_dict(), model_path)
                    # save img, pred_mask, semantic_mask, instance_mask to hdf
                    with h5py.File(os.path.join(snap_dir, f"snapshot_step_{step}.hdf"), "w") as f:
                        f.create_dataset("img", data=img.cpu().detach().numpy())
                        f.create_dataset("pred_mask", data=pred_mask.softmax(1).cpu().detach().numpy())
                        f.create_dataset(
                            "semantic_mask", data=semantic_mask.unsqueeze(1).cpu().detach().numpy()
                        )
                        f.create_dataset(
                            "img_aug", data=img_aug.cpu().detach().numpy()
                        )
                        f.create_dataset(
                            "semantic_mask_aug",
                            data=semantic_mask_aug.unsqueeze(1).cpu().detach().numpy()
                        )
                        if instance_mask is not None:
                            f.create_dataset(
                                "instance_mask",
                                data=instance_mask.unsqueeze(1).cpu().detach().numpy().astype(np.uint8)
                            )
                            f.create_dataset(
                                "instance_mask_aug",
                                data=instance_mask_aug.unsqueeze(1).cpu().detach().numpy().astype(np.uint8)
                            )
                if hasattr(cfg, "primary_metric"):
                    primary_metric_history.append(logging_dict["validation/"+cfg.primary_metric])
                if hasattr(cfg, "primary_metric") and (logging or 'RANK' not in os.environ):
                    if max(primary_metric_history) == logging_dict["validation/"+cfg.primary_metric]:
                        model_path = os.path.join(
                            checkpoint_path, f"best_model.pth"
                        )
                        torch.save(model.state_dict(), model_path)
                        best_checkpoint_step = step
            step += 1

    if hasattr(cfg, "primary_metric"):
        model.load_state_dict(torch.load(model_path))
        evaluater.save_dir = os.path.join(log_dir, "test_results")
        evaluater.fname = "test_metrics_best_model.csv"
        logging_dict, classwise_dict = evaluater.compute_metrics(model, test_dataloader, device)
    if logging:
        logging_dict["best_checkpoint_step"] = best_checkpoint_step
        logging_dict = {f"test/{k}": v for k, v in logging_dict.items()}
        classwise_dict = {k+"_test": v for k, v in classwise_dict.items()}
        wandb.log(logging_dict)
        wandb.log(classwise_dict)
    wandb.finish()
    return loss_history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    train(cfg)