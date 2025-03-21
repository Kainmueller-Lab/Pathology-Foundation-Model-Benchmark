import os
import re

import numpy as np
import pandas as pd
import torch
from skimage.measure import regionprops_table
from tqdm import tqdm

from benchmark.utils.metric_utils import (
    accuracy,
    confusion_matrix_func,
    f1_score_class,
    precision,
    recall,
)
from benchmark.utils.utils import maybe_resize, save_imgs_for_debug


def extract_numbers_from_string(s):
    """
    Extracts all numbers (integer or decimal) from a string.

    Args:
        s (str): The input string.

    Returns
    -------
        list: A list of numbers as strings. Convert to integers or floats if needed.
    """
    return re.findall(r"\d+\.?\d*", s)


class Eval:
    def __init__(
        self,
        label_dict,
        instance_level=True,
        pixel_level=False,
        save_dir=None,
        fname=None,
    ):
        """
        Initializes the evaluation pipeline.

        Args:
            label_dict (dict): Dictionary mapping class ids to class names.
            instance_level (bool): Whether to compute instance-level metrics.
            pixel_level (bool): Whether to compute pixel-level metrics.
            save_dir (str): Directory to save the evaluation results and predictions.
            fname (str): Filename to save the evaluation results.
        """
        self.instance_level = instance_level
        self.pixel_level = pixel_level
        self.label_dict = {int(k): v for k, v in label_dict.items()}
        self.label_dict_rev = {v: int(k) for k, v in label_dict.items()}
        self.save_dir = save_dir
        if fname:
            self.fname = fname
        else:
            self.fname = "evaluation_results.csv"

    def compute_metrics(self, model, dataloader, device, save_preds=False, snap_dir=None):
        """
        Computes the metrics over the entire dataset.

        Args:
            model (torch.nn.Module): The model to evaluate.
            dataloader (torch.utils.data.DataLoader): DataLoader for the dataset.
            device (torch.device): Device to run the model on.

        Returns
        -------
            dict: Computed metrics.
            dict: Instance-level predictions (empty if instance_level=False).
        """
        print("Running inference for computing metrics...")
        if save_preds:
            print(f"Saving predictions at {snap_dir}")
        pred_dfs = []
        model.eval()
        with torch.no_grad():
            for i, sample_dict in tqdm(enumerate(dataloader)):
                img = sample_dict["image"].float().to(device)
                semantic_mask = sample_dict["semantic_mask"]
                instance_mask = sample_dict.get("instance_mask", None)
                sample_name = sample_dict.get("sample_name", None)

                # Run model to get predictions
                pred_logits = model(img)
                pred_logits = maybe_resize(pred_logits, semantic_mask)

                if save_preds:
                    save_imgs_for_debug(
                        snap_dir,
                        i,
                        img,
                        pred_logits,
                        semantic_mask,
                        img_aug=None,
                        semantic_mask_aug=None,
                        instance_mask=instance_mask,
                        instance_mask_aug=None,
                    )

                # Assuming pred_logits is (batch_size, num_classes, H, W)
                pred_probs = torch.softmax(pred_logits, dim=1)
                pred_probs = pred_probs.cpu().numpy()
                semantic_mask = semantic_mask.numpy()
                instance_mask = instance_mask.numpy()

                batch_size = img.shape[0]
                for i in range(batch_size):
                    if self.instance_level:
                        # Get per-cell predictions and ground truths
                        pred_df = self._get_instance_level_labels(
                            pred_probs[i],
                            semantic_mask[i],
                            instance_mask[i],
                        )
                        pred_df["sample_name"] = sample_name[i]
                        pred_dfs.append(pred_df)
                    if self.pixel_level:
                        # Pixel-level evaluation (not implemented)
                        pass
        pred_df = pd.concat(pred_dfs, ignore_index=True)
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
            pred_df.to_csv(os.path.join(self.save_dir, "predictions.csv"), index=False)
        # Compute metrics
        if self.instance_level:
            metrics = self.compute_metrics_instance_level(
                y_pred=pred_df["pred_class"].values,
                y_true=pred_df["groundtruth"].values,
            )
            return metrics
        else:
            return {}

    def _get_instance_level_labels(self, semantic_pred, semantic_gt, instance_gt):
        """Extracts per-instance true and predicted class labels using regionprops_table.

        Args:
            semantic_pred (np.ndarray): Predicted semantic segmentation mask (C, H, W).
            semantic_gt (np.ndarray): Ground truth semantic segmentation mask (H, W).
            instance_gt (np.ndarray): Ground truth instance segmentation mask (H, W).

        Returns
        -------
                data_frame: Dataframe containing instance-level predictions amd ground truths in
                the following format:
                label | intensity_mean_gt | class_0 | class_1 | ... | class_n | pred_class

        """
        props_gt = regionprops_table(
            instance_gt,
            intensity_image=semantic_gt,
            properties=("label", "intensity_mean"),
        )
        props_gt = pd.DataFrame(props_gt)
        props_gt.rename(columns={"intensity_mean": "groundtruth"}, inplace=True)
        props_gt["groundtruth"] = props_gt["groundtruth"].astype(int)

        semantic_pred = np.moveaxis(semantic_pred, 0, -1)
        props_pred = regionprops_table(
            instance_gt,
            intensity_image=semantic_pred,
            properties=("label", "intensity_mean"),
        )
        props_pred = pd.DataFrame(props_pred)

        # extract numbers from props_pred.columns if number is there
        rename_dict = {
            col: self.label_dict[int(extract_numbers_from_string(col)[0])]
            for col in props_pred.columns
            if col != "label"
        }
        props_pred.rename(columns=rename_dict, inplace=True)
        # get maximum intensity class
        props_pred["pred_class_name"] = props_pred.drop("label", axis=1).idxmax(axis=1)
        props_pred["pred_class"] = props_pred["pred_class_name"].map(self.label_dict_rev)

        # Merge the two dataframes
        pred_df = pd.merge(props_gt, props_pred, on="label", how="inner")
        return pred_df

    def compute_metrics_instance_level(self, y_pred, y_true):
        """
        Computes instance-level metrics.

        Args:
            y_pred (np.ndarray): Predicted class labels per instance.
            y_true (np.ndarray): True class labels per instance.

        Returns
        -------
            dict: Computed metrics.
        """
        metrics = {}

        # Get unique class labels
        unique_classes = np.unique(np.concatenate([y_true, y_pred]))
        # exclude background class
        unique_classes = unique_classes[unique_classes != 0]

        # Calculate precision, recall, and F1 score per class
        precision_scores = {}
        recall_scores = {}
        f1_scores = {}
        accuracy_scores = {}
        classwise_metrics = {}
        for cls in unique_classes:
            class_name = self.label_dict[cls]
            precision_scores[class_name] = precision(y_true, y_pred, class_id=cls)
            recall_scores[class_name] = recall(y_true, y_pred, class_id=cls)
            f1_scores[class_name] = f1_score_class(y_true, y_pred, class_id=cls)
            accuracy_scores[class_name] = accuracy(y_true, y_pred, class_id=cls)
            classwise_metrics[f"{class_name}/precision"] = precision_scores[class_name]
            classwise_metrics[f"{class_name}/recall"] = recall_scores[class_name]
            classwise_metrics[f"{class_name}/f1_score"] = f1_scores[class_name]
            classwise_metrics[f"{class_name}/accuracy"] = accuracy_scores[class_name]

        # Calculate macro averages
        metrics["precision_macro"] = np.mean(list(precision_scores.values()))
        metrics["recall_macro"] = np.mean(list(recall_scores.values()))
        metrics["f1_score_macro"] = np.mean(list(f1_scores.values()))
        metrics["accuracy_macro"] = np.mean(list(accuracy_scores.values()))

        # Calculate micro averages
        metrics["precision_micro"] = precision(y_true, y_pred, class_id=None)
        metrics["recall_micro"] = recall(y_true, y_pred, class_id=None)
        metrics["f1_score_micro"] = f1_score_class(y_true, y_pred, class_id=None)
        metrics["accuracy_micro"] = accuracy(y_true, y_pred, class_id=None)

        # # Calculate confusion matrix
        # labels = [self.label_dict[cls] for cls in unique_classes]
        # conf_mat = confusion_matrix_func(y_true, y_pred, labels=unique_classes)
        # # rename index and columns
        # metrics["confusion_matrix"] = pd.DataFrame(
        #     conf_mat,
        #     index=labels,
        #     columns=labels
        # )

        if self.save_dir:
            # save metrics to csv and exclude all per class metrics before
            cols = [
                "precision_macro",
                "recall_macro",
                "f1_score_macro",
                "accuracy_macro",
                "precision_micro",
                "recall_micro",
                "f1_score_micro",
                "accuracy_micro",
            ]
            metrics_df = pd.DataFrame({k: metrics[k] for k in cols}, index=[0])
            classwise_metrics_df = pd.DataFrame(classwise_metrics, index=[0])
            metrics_df = pd.concat([metrics_df, classwise_metrics_df], axis=1)
            metrics_df.to_csv(os.path.join(self.save_dir, self.fname), index=False)
        return metrics, classwise_metrics
