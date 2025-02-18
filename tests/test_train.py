import os
import tempfile
from pathlib import Path

from omegaconf import OmegaConf

from benchmark.train import train


def test_train():
    # load config
    root = Path(__file__).parents[1]
    cfg_path = root / "configs" / "config.yaml"
    cfg = OmegaConf.load(cfg_path)
    with tempfile.TemporaryDirectory() as tmp_dir:
        cfg.experiment = "mock_experiment"
        cfg.experiment_path = tmp_dir
        cfg.model.backbone = "MOCK"
        cfg.early_stopping = 20
        cfg.save_snapshots = False
        cfg.save_all_ckpts = False
        loss_history = train(cfg)
        log_dir = os.path.join(cfg.experiment_path, cfg.experiment)
        # check if files and folders got created
        assert "train" in os.listdir(log_dir)
        assert "config.yaml" in os.listdir(log_dir)
        assert "snaps" in os.listdir(os.path.join(log_dir, "train"))
        assert "checkpoints" in os.listdir(os.path.join(log_dir, "train"))
        assert "summary" in os.listdir(os.path.join(log_dir, "train"))
        assert "best_model.pth" in os.listdir(os.path.join(log_dir, "train", "checkpoints"))
        # # test hovernext
        cfg_path = root / "configs" / "hovernext_config.yaml"
        cfg = OmegaConf.load(cfg_path)
        cfg.experiment = "mock_experiment"
        cfg.experiment_path = tmp_dir
        loss_history = train(cfg)
        log_dir = os.path.join(cfg.experiment_path, cfg.experiment)
        # check if files and folders got created
        assert "train" in os.listdir(log_dir)
        assert "config.yaml" in os.listdir(log_dir)
        assert "snaps" in os.listdir(os.path.join(log_dir, "train"))
        assert "checkpoints" in os.listdir(os.path.join(log_dir, "train"))
        assert "summary" in os.listdir(os.path.join(log_dir, "train"))
        assert "best_model.pth" in os.listdir(os.path.join(log_dir, "train", "checkpoints"))
