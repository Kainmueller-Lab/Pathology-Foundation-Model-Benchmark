import os
import torch
from torch.utils.data import DataLoader
from time import time
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from benchmark.init_dist import init_distributed
from benchmark.eval import Eval
from benchmark.simple_segmentation_model import *
import argparse

os.environ["OMP_NUM_THREADS"] = "1"
torch.backends.cudnn.benchmark = True


def inference(cfg):

    # prepare path for writing inference results
    log_dir = os.path.join(cfg.experiment_path, cfg.experiment)
    cfg.writer_dir = os.path.join(log_dir,'summary', str(time()))
    os.makedirs(cfg.writer_dir,exist_ok=True)

    # load dataset and model
    train_dset, val_dset, test_dset, label_dict = prep_datasets(cfg)
    if cfg.split == "train":
        eval_dset = train_dset
    elif cfg.split == "test":
        eval_dset = test_dset
    elif cfg.split == "val":
        eval_dset = val_dset
    eval_dataloader = DataLoader(
            eval_dset, batch_size=cfg.dataset.batch_size, pin_memory=True, num_workers=0)
    model_wrapper = eval(cfg.model.model_wrapper)
    model = model_wrapper(
        model_name=cfg.model.backbone, num_classes=len(label_dict), return_embeddings=cfg.save_embeddings)
    if cfg.pretrained_path is not None:
        model.load_state_dict(torch.load(cfg.pretrained_path))

    # prepare evaluator
    if cfg.compute_metrics:
        evaluater = Eval(
            label_dict, instance_level=True, pixel_level=False,
            save_dir=os.path.join(log_dir, "validation_results"),
            fname="validation_metrics.csv")

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

    # save parameters
    with open(os.path.join(cfg.experiment_path, cfg.experiment, 'config.yaml'), 'w') as f:
        OmegaConf.save(config=cfg, f=f)

    # inference
    print("Start Inference")
    model.eval()
    evaluater.save_dir = os.path.join(log_dir, "validation_results")
    evaluater.fname = f"metrics.csv"
    _ = evaluater.compute_metrics(model, eval_dataloader, device)

    print("Inference done")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/inference_config.yaml")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    inference(cfg)
