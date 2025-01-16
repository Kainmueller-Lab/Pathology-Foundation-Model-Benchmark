import os
import tempfile
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from benchmark.eval import Eval, extract_numbers_from_string


class MockDataset(Dataset):
    def __init__(self, num_samples=10, num_classes=3):
        self.num_samples = num_samples
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            "image": torch.rand(3, 224, 224),
            "semantic_mask": np.random.randint(0, self.num_classes, (224, 224)),
            "instance_mask": np.random.randint(0, 10, (224, 224)),
            "sample_name": f"sample_{idx}"
        }


class MockModel(torch.nn.Module):
    def forward(self, x):
        return torch.rand(x.size(0), 3, x.size(2), x.size(3))


def test_extract_numbers_from_string():
    assert extract_numbers_from_string("abc123def456") == ["123", "456"]
    assert extract_numbers_from_string("no numbers") == []
    assert extract_numbers_from_string("1.23 and 4.56") == ["1.23", "4.56"]


def test_eval_instance_level():
    label_dict = {0: "class_0", 1: "class_1", 2: "class_2"}
    eval = Eval(label_dict, instance_level=True, pixel_level=False)

    model = MockModel()
    dataset = MockDataset(num_samples=5, num_classes=3)
    dataloader = DataLoader(dataset, batch_size=2)

    device = torch.device("cpu")
    metrics = eval.compute_metrics(model, dataloader, device)

    assert "precision_macro" in metrics
    assert "recall_macro" in metrics
    assert "f1_score_macro" in metrics
    assert "accuracy_macro" in metrics
    assert "precision_micro" in metrics
    assert "recall_micro" in metrics
    assert "f1_score_micro" in metrics
    assert "accuracy_micro" in metrics
    assert "confusion_matrix" in metrics


def test_eval_save_predictions():
    label_dict = {0: "class_0", 1: "class_1", 2: "class_2"}
    with tempfile.TemporaryDirectory() as tmp_dir:
        eval = Eval(label_dict, instance_level=True, pixel_level=False, save_dir=tmp_dir)

        model = MockModel()
        dataset = MockDataset(num_samples=5, num_classes=3)
        dataloader = DataLoader(dataset, batch_size=2)

        device = torch.device("cpu")
        eval.compute_metrics(model, dataloader, device)

        assert os.path.exists(os.path.join(tmp_dir, "predictions.csv"))


def test_eval_save_metrics():
    label_dict = {0: "class_0", 1: "class_1", 2: "class_2"}
    with tempfile.TemporaryDirectory() as tmp_dir:
        eval = Eval(label_dict, instance_level=True, pixel_level=False, save_dir=tmp_dir)

        model = MockModel()
        dataset = MockDataset(num_samples=5, num_classes=3)
        dataloader = DataLoader(dataset, batch_size=2)

        device = torch.device("cpu")
        eval.compute_metrics(model, dataloader, device)

        assert os.path.exists(os.path.join(tmp_dir, "evaluation_results.csv"))


def test_get_instance_level_labels():
    label_dict = {0: "class_0", 1: "class_1", 2: "class_2"}
    eval = Eval(label_dict, instance_level=True, pixel_level=False)

    semantic_pred = np.random.rand(3, 224, 224)
    semantic_gt = np.random.randint(0, 3, (224, 224))
    instance_gt = np.random.randint(0, 10, (224, 224))

    pred_df = eval._get_instance_level_labels(semantic_pred, semantic_gt, instance_gt)

    assert "label" in pred_df.columns
    assert "groundtruth" in pred_df.columns
    assert "pred_class" in pred_df.columns
