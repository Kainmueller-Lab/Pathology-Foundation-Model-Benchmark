import os
import re
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
from skimage.measure import regionprops_table
import h5py
from benchmark.metric_utils import (
    accuracy,
    precision,
    recall,
    f1_score_class,
    confusion_matrix_func,
)


def extract_numbers_from_string(s):
    """
    Extracts all numbers (integer or decimal) from a string.

    Args:
        s (str): The input string.

    Returns:
        list: A list of numbers as strings. Convert to integers or floats if needed.
    """
    return re.findall(r'\d+\.?\d*', s)


class Eval:
    def __init__(
            self, label_dict, instance_level=True, pixel_level=False, save_dir=None, fname=None,
            save_samples=False, save_embeddings=False):
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
        self.save_samples = save_samples
        self.save_embeddings = save_embeddings
        self.label_dict = {int(k): v for k, v in label_dict.items()}
        self.label_dict_rev = {v: int(k) for k, v in label_dict.items()}
        self.save_dir = save_dir
        if fname:
            self.fname = fname
        else:
            self.fname = "evaluation_results.csv"

    def compute_metrics(self, model, dataloader, device):
        """
        Computes the metrics over the entire dataset.

        Args:
            model (torch.nn.Module): The model to evaluate.
            dataloader (torch.utils.data.DataLoader): DataLoader for the dataset.
            device (torch.device): Device to run the model on.

        Returns:
            dict: Computed metrics.
            dict: Instance-level predictions (empty if instance_level=False).
        """
        print("Running inference for computing metrics...")
        pred_dfs = []
        if self.save_samples:
            snap_dir = os.path.join(self.save_dir, 'samples')
            os.makedirs(snap_dir, exist_ok=True)
        if hasattr(model, "return_embeddings"):
            if model.return_embeddings and self.save_embeddings:
                save_embeddings = True
        model.eval()
        sample_idx = 0
        with torch.no_grad():
            for sample_dict in tqdm(dataloader):
                img = sample_dict["image"].float().to(device)
                semantic_mask = sample_dict["semantic_mask"].numpy()
                instance_mask = sample_dict.get("instance_mask", None)
                sample_name = sample_dict.get("sample_name", None)

                # Run model to get predictions
                if save_embeddings:
                    pred_logits, patch_embeddings = model(img)
                else:
                    pred_logits = model(img)
                # Assuming pred_logits is (batch_size, num_classes, H, W)
                pred_probs = torch.softmax(pred_logits, dim=1).cpu().numpy()

                batch_size = img.shape[0]
                for i in range(batch_size):
                    if self.instance_level:
                        # Get per-cell predictions and ground truths
                        pred_df = self._get_instance_level_labels(
                            pred_probs[i],
                            semantic_mask[i],
                            instance_mask[i].numpy(),
                        )
                        pred_df["sample_name"] = sample_name[i]
                        pred_dfs.append(pred_df)
                    if self.pixel_level:
                        # Pixel-level evaluation (not implemented)
                        pass
                    if self.save_samples:
                        with h5py.File(os.path.join(snap_dir, f"sample_{sample_idx}_{sample_name[i]}.hdf"), "w") as f:
                            f.create_dataset("img", data=img[i].cpu().detach().numpy())
                            f.create_dataset("pred", data=pred_probs[i].cpu().detach().numpy())
                            f.create_dataset("semantic_mask", data=semantic_mask[i].cpu().detach().numpy())
                            if instance_mask is not None:
                                f.create_dataset("instance_mask", data=instance_mask[i].cpu().detach().numpy())
                            if save_embeddings:
                                f.create_dataset("patch_embeddings", data=patch_embeddings.cpu().detach().numpy())
                    sample_idx += 1

        pred_df = pd.concat(pred_dfs, ignore_index=True)
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
            pred_df.to_csv(os.path.join(self.save_dir, "predictions.csv"), index=False)
        # Compute metrics
        if self.instance_level:
            metrics = self.compute_metrics_instance_level(
                y_pred=pred_df["pred_class"].values,
                y_true=pred_df["groundtruth"].values
            )
            return metrics
        else:
            return {}

    def _get_instance_level_labels(self, semantic_pred, semantic_gt, instance_gt):
        """
        Extracts per-instance true and predicted class labels using regionprops_table.

        Args:
            semantic_pred (np.ndarray): Predicted semantic segmentation mask (C, H, W).
            semantic_gt (np.ndarray): Ground truth semantic segmentation mask (H, W).
            instance_gt (np.ndarray): Ground truth instance segmentation mask (H, W).
        Returns:
                data_frame: Dataframe containing instance-level predictions amd ground truths in
                the following format:
                label | intensity_mean_gt | class_0 | class_1 | ... | class_n | pred_class

        """

        props_gt = regionprops_table(
            instance_gt,
            intensity_image=semantic_gt,
            properties=('label', 'intensity_mean')
        )
        props_gt = pd.DataFrame(props_gt)
        props_gt.rename(columns={'intensity_mean': 'groundtruth'}, inplace=True)
        props_gt['groundtruth'] = props_gt['groundtruth'].astype(int)

        semantic_pred = np.moveaxis(semantic_pred, 0, -1)
        props_pred = regionprops_table(
            instance_gt,
            intensity_image=semantic_pred,
            properties=('label', 'intensity_mean')
        )
        props_pred = pd.DataFrame(props_pred)

        # extract numbers from props_pred.columns if number is there
        rename_dict = {
            col: self.label_dict[int(extract_numbers_from_string(col)[0])] for col in props_pred.columns if col != 'label'
        }
        props_pred.rename(columns=rename_dict, inplace=True)
        # get maximum intensity class
        props_pred['pred_class_name'] = props_pred.drop('label', axis=1).idxmax(axis=1)
        props_pred['pred_class'] = props_pred['pred_class_name'].map(self.label_dict_rev)

        # Merge the two dataframes
        pred_df = pd.merge(props_gt, props_pred, on='label', how='inner')
        return pred_df

    def compute_metrics_instance_level(self, y_pred, y_true):
        """
        Computes instance-level metrics.

        Args:
            y_pred (np.ndarray): Predicted class labels per instance.
            y_true (np.ndarray): True class labels per instance.

        Returns:
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

        for cls in unique_classes:
            class_name = self.label_dict[cls]
            precision_scores[class_name] = precision(y_true, y_pred, class_id=cls)
            recall_scores[class_name] = recall(y_true, y_pred, class_id=cls)
            f1_scores[class_name] = f1_score_class(y_true, y_pred, class_id=cls)
            accuracy_scores[class_name] = accuracy(y_true, y_pred, class_id=cls)

        metrics["precision_per_class"] = precision_scores
        metrics["recall_per_class"] = recall_scores
        metrics["f1_score_per_class"] = f1_scores
        metrics["accuracy_per_class"] = accuracy_scores

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

        # Calculate confusion matrix
        labels = [self.label_dict[cls] for cls in unique_classes]
        conf_mat = confusion_matrix_func(y_true, y_pred, labels=unique_classes)
        # rename index and columns
        metrics["confusion_matrix"] = pd.DataFrame(
            conf_mat,
            index=labels,
            columns=labels
        )

        if self.save_dir:
            # save metrics to csv and exclude all per class metrics before
            cols = ["precision_macro", "recall_macro", "f1_score_macro", "accuracy_macro",
                    "precision_micro", "recall_micro", "f1_score_micro", "accuracy_micro"]
            metrics_df = pd.DataFrame({k: metrics[k] for k in cols}, index=[0])    
            metrics_df.to_csv(os.path.join(self.save_dir, self.fname), index=False)
        return metrics


class Inference:
    '''
    Class for saving all interesting outputs (for further analysis)
    from a full model pass to a given Dataset, specifically for these
    segmentation models. (Note: can be very memory-inefficient.)
    '''
    def __init__(
        self, model, result_path, input_data=True,
        label=True, prediction=True, embeddings=True, inferred_masks=True):
        '''
        Initializes the class and sets up which outputs to store.
        '''
        self.model = model  # e.g., an instance of SimpleSegmentationModel
        self.store_input = input_data
        self.store_label = label
        self.store_prediction = prediction
        self.store_embeddings = embeddings
        self.store_inferred_masks = inferred_masks
        self.softmax = torch.nn.Softmax(dim=1)

        if self.store_embeddings:
            if model.return_embeddings is False:
                raise ValueError("Model must return embeddings to store them.")

        self.result_path = result_path
        os.makedirs(self.result_path, exist_ok=True)

    @torch.no_grad()
    def forward(self, images):
        if self.model.return_embeddings:
            segmentation_logits, embeddings = self.model(images)
            embeddings = embeddings.cpu().numpy()
        else:
            segmentation_logits = self.model(images)
            embeddings = None
        segmentation_pred = self.softmax(segmentation_logits).cpu().numpy()
        return segmentation_pred, embeddings

    def predict_batch(self, images):
        '''
        Perform forward pass on a single batch and return a dictionary
        of the stored items (input, label, embedding, prediction, etc.).

        Expects batch = (images, labels) or a dict with keys 'image', 'label'.
        Adapt as needed for your dataset structure.
        '''
        self.model.eval()
        B, C, H, W = images.shape

        # 2) forward pass
        predictions, embeddings = self.forward(images)

        # 3) optional post-processing (e.g. argmax for segmentation)
        if self.store_inferred_masks: 
            inferred_masks = predictions.argmax(dim=1)  # shape [B, H, W] or [B, p, p]
        else:
            inferred_masks = None

        # 4) Build output dictionary
        output_dict = {}
        if self.store_input:
            output_dict['input'] = images.cpu()
        if self.store_prediction:
            output_dict['prediction'] = predictions.cpu()
        if self.store_embeddings and (embeddings is not None):
            output_dict['embeddings'] = embeddings.cpu()
        if self.store_inferred_masks and (inferred_masks is not None):
            output_dict['inferred_masks'] = inferred_masks.cpu()

        return output_dict

    # def predict(self, dataloader):
    #     '''
    #     Predict and store outputs for the entire dataset or DataLoader.
    #     This method saves the outputs to disk batch by batch.
    #     '''
    #     for idx, batch in enumerate(tqdm(data_loader, desc="Predicting")):
    #         images, labels = ...
    #         batch_out = self.predict_batch(batch)
    #         save_path = os.path.join(self.result_path, f"batch_{idx}.pt")
    #         torch.save(batch_out, save_path)


def fetch_embeddings_and_patches(patch_embeddings, img, mask=None, patch_size=16):
    """
    reformates the patch embeddings and fetches the 
    corresponding patches from the input and the mask

    Args: 
        patch_embeddings (Tensor): Embeddings from model (B, D, P_H, P_W), P_H: N horizontal patches
        img (Tensor): The original image of shape (B, C, H, W)
        mask (Tensor): The mask of shape (B, H, W)
        patch_size (int): The patch size (default 16)

    Returns:
        patch_embeddings (Tensor): (B, num_patches, D)
        patches_img (Tensor): (B, num_patches, C, patch_size, patch_size)
        patches_mask (Tensor): (B, num_patches, 1, patch_size, patch_size)
    """
    patch_embeddings_flat = patch_embeddings.flatten(2, 3).transpose(1, 2)  # (B, P_H * P_W, D) 

    # extract image patches
    unfold = torch.nn.Unfold(kernel_size=patch_size, stride=patch_size)
    unfolded_img = unfold(img)  # (B, C*patch_size*patch_size, P_H*P_W)
    B, _, num_patches = unfolded_img.shape
    patches_img = (
        unfolded_img
        .transpose(1, 2)  # (B, num_patches, C*patch_size*patch_size)
        .reshape(B, num_patches, img.shape[1], patch_size, patch_size))

    # extract patched masks
    if mask is not None:
        mask = mask.unsqueeze(1)
        unfolded_mask = unfold(mask)  # (B, 1*patch_size*patch_size, num_patches)
        patches_mask = (unfolded_mask
            .transpose(1, 2)
            .reshape(B, num_patches, 1, patch_size, patch_size))
        return patch_embeddings_flat, patches_img, patches_mask
        
    return patch_embeddings_flat, patches_img
