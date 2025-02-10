import os
from pathlib import Path

import numpy as np
import torch

from datetime import datetime

from torchmetrics.classification import MulticlassJaccardIndex

from bio_image_datasets.lizard_dataset import LizardDataset
from bio_image_datasets.pannuke import PanNukeDataset
from bio_image_datasets.consep_dataset import ConSePDataset
from bio_image_datasets.schuerch_dataset import SchuerchDataset
from bio_image_datasets.segpath_dataset import SegPath
from dotenv import load_dotenv
from benchmark import utils
from benchmark.lmdb_dataset import LMDBDataset
from benchmark.mask2former import Mask2FormerModel
from benchmark.mask2former import load_m2f_model_and_transform


def normalize(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-5)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # load the environment variables
    dotenv_path = Path(__file__).parents[1] / ".env"
    load_dotenv(dotenv_path=dotenv_path)
    num_classes = 7
    min_count = 10

    outdir = "outputs/test_m2f_hf_{date}".format(date=datetime.now().strftime("%Y%m%d%H%M"))
    os.makedirs(outdir, exist_ok=True)

    model_name = "uni2"
    print(f"Instantiating model {model_name}...")
    model = Mask2FormerModel(model_name=model_name, num_classes=num_classes)
    model.cuda()
    dataset = LMDBDataset(
        path="/fast/AG_Kainmueller/data/patho_foundation_model_bench_data/lizard_dataset/lizard_tiled_lmdb/",
    )

    for i, img_idx in enumerate([1, 11, 111, 113]):
        sample = dataset[img_idx]
        image_cpu = sample["image"]
        image = torch.Tensor(image_cpu).cuda()
        ground_truth = sample["semantic_mask"]
        instance_mask = sample["instance_mask"]

        print("image", image.shape, image.min(), image.max())
        processed_image = model.image_processor(image, return_tensors="pt")
        processed_image = {
            "pixel_values": processed_image["pixel_values"].cuda(),
            "pixel_mask": processed_image["pixel_mask"].cuda(),
        }
        print("ground_truth", ground_truth.shape, ground_truth.min(), ground_truth.max())

        seg, seg_logits = model(processed_image)  # .unsqueeze(0))
        seg_logits_cpu = seg_logits.detach().cpu().numpy()
        seg = seg.detach().cpu().numpy()
        print("segmentation_logits shape, min, max", seg_logits_cpu.shape, seg_logits_cpu.min(), seg_logits_cpu.max())
        print("seg_map shape, min, max", seg.shape, seg.min(), seg.max())
        seg_armax = np.argmax(seg_logits_cpu, axis=1, keepdims=True)

        # get the unique values and their counts
        unique, counts = np.unique(seg, return_counts=True)
        vals_count_seg = {el: count for el, count in zip(unique, counts) if count > min_count}

        unique, counts = np.unique(ground_truth, return_counts=True)
        vals_count_gt = {el: count for el, count in zip(unique, counts) if count > min_count}

        print("GT", vals_count_gt)
        print("PRED", vals_count_seg)

        f, a = plt.subplots(1, 4)
        f.set_size_inches(25, 5)
        a[0].imshow(normalize(image_cpu.transpose(1, 2, 0)))
        a[1].imshow(normalize(ground_truth))
        a[2].imshow(normalize(seg[0, :, :]))
        a[3].imshow(normalize(seg_logits_cpu[0, :3, :, :].transpose(1, 2, 0)))
        plt.savefig(os.path.join(outdir, f"segmented_image_{i}.png"), bbox_inches="tight")

        # Assuming `segmentation_logits` is the predicted segmentation map and `ground_truth` is the actual label
        if seg_logits.max() < num_classes:
            score = MulticlassJaccardIndex(num_classes=num_classes, average="macro")
            score_value = score(seg_logits.cpu(), torch.tensor(ground_truth))
            print("Jaccard Score:", score_value)
