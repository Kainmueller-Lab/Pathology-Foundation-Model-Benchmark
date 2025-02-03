import os
from pathlib import Path

import numpy as np
import torch

from torchmetrics.classification import MulticlassJaccardIndex

from benchmark.m2f_hf_wrapper import M2F_HF_SegmentationModel, BACKBONE_ARCHS
from bio_image_datasets.lizard_dataset import LizardDataset
from bio_image_datasets.pannuke import PanNukeDataset
from bio_image_datasets.consep_dataset import ConSePDataset
from bio_image_datasets.schuerch_dataset import SchuerchDataset
from bio_image_datasets.segpath_dataset import SegPath
from dotenv import load_dotenv
from benchmark import utils
from benchmark.lmdb_dataset import LMDBDataset

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # load the environment variables
    dotenv_path = Path(__file__).parents[1] / ".env"
    load_dotenv(dotenv_path=dotenv_path)
    num_classes = 8

    backbone_size = "small"
    model_names = ["provgigapath", "phikonv2", "virchow2", "uni", "titan"]

    model_name = "provgigapath"
    assert model_name in model_names

    print(f"Instantiating model {model_name}...")
    model = M2F_HF_SegmentationModel(
        backbone_size=backbone_size,
        model_name=model_name,
        num_classes=num_classes,
        cfg_str=f"./configs/m2f_config.yaml",
    )
    dataset = LMDBDataset(
        path="/fast/AG_Kainmueller/data/patho_foundation_model_bench_data/lizard_dataset/lizard_tiled_lmdb/",
    )
    sample = dataset[111]
    image = torch.Tensor(sample["image"]).cuda()
    ground_truth = sample["semantic_mask"]

    print("doing_inference...")
    processed_image = model.image_processor(image, return_tensors="pt")
    print(processed_image["pixel_values"].shape, processed_image["pixel_values"].device)
    segmentation_logits = model(processed_image)
    segmentation_logits_cpu = segmentation_logits[0, :, :, :].detach().cpu().numpy()
    print("segmentation_logits", segmentation_logits_cpu.shape)
    segmentation_classes = np.argmax(segmentation_logits_cpu, axis=0)

    print("segmentation_classes", segmentation_classes.shape, segmentation_classes.min(), segmentation_classes.max())
    num_classes = segmentation_logits_cpu.shape[0]
    segmented_image = utils.render_segmentation(segmentation_classes, num_classes=num_classes)

    outdir = "../../outputs"
    os.makedirs(outdir, exist_ok=True)

    print("shapes:", image.shape, segmentation_logits_cpu.shape, segmented_image.shape, ground_truth.shape)

    min_count = 10
    unique, counts = np.unique(segmentation_logits_cpu, return_counts=True)
    vals_count_seg = {el: count for el, count in zip(unique, counts) if count > min_count}

    unique, counts = np.unique(ground_truth, return_counts=True)
    vals_count_gt = {el: count for el, count in zip(unique, counts) if count > min_count}

    unique, counts = np.unique(segmentation_classes, return_counts=True)
    vals_count_segcls = {el: count for el, count in zip(unique, counts) if count > min_count}

    print("GT", vals_count_gt)
    print("segmentation_classes", vals_count_segcls, segmentation_classes.size)

    def normalize(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-4)

    f, a = plt.subplots(1, 4)
    f.set_size_inches(20, 5)
    a[0].imshow(normalize(image.detach().cpu().numpy().transpose(1, 2, 0)))
    a[1].imshow(normalize(segmentation_logits_cpu[:3, :, :].transpose(1, 2, 0)))
    a[2].imshow(normalize(segmented_image))
    a[3].imshow(normalize(ground_truth))

    plt.savefig(os.path.join(outdir, "segmented_image.png"), bbox_inches="tight")

    # Assuming `segmentation_logits` is the predicted segmentation map and `ground_truth` is the actual label
    score = MulticlassJaccardIndex(num_classes=num_classes, average="macro")
    print(segmentation_logits.shape, ground_truth.shape)
    score_value = score(segmentation_logits.cpu(), torch.tensor(ground_truth))
    print("Jaccard Score:", score_value)
