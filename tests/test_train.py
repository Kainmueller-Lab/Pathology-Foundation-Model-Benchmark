from benchmark.train import train
import tempfile
from omegaconf import OmegaConf
from pathlib import Path
import os


def test_train():
    # load config
    root = Path(__file__).parents[1]
    cfg_path = root / "configs" / "config.yaml"
    cfg = OmegaConf.load(cfg_path)
    with tempfile.TemporaryDirectory() as tmp_dir:
        cfg.experiment = "mock_experiment"
        cfg.experiment_path = tmp_dir
        cfg.model.backbone = "MOCK"
        loss_history = train(cfg)
        log_dir = os.path.join(cfg.experiment_path, cfg.experiment)
        # check if files and folders got created
        assert "train" in os.listdir(log_dir)
        assert "config.yaml" in os.listdir(log_dir)
        assert "snaps" in os.listdir(os.path.join(log_dir, "train"))
        assert "checkpoints" in os.listdir(os.path.join(log_dir, "train"))
        assert "summary" in os.listdir(os.path.join(log_dir, "train"))
        for step in [100,200]:
            assert f"checkpoint_step_{step}.pth" in \
                os.listdir(os.path.join(log_dir, "train", "checkpoints"))
        wandb_path = os.path.join(log_dir, "train", "summary", 
            os.listdir(os.path.join(log_dir, "train", "summary"))[0], 
            "wandb"
        )
        assert os.path.exists(wandb_path)
        # check if loss decreases
        assert loss_history[0] > loss_history[-1]
