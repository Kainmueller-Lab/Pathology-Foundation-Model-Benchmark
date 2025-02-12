import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from bio_image_datasets.consep_dataset import ConSePDataset
from bio_image_datasets.lizard_dataset import LizardDataset
from bio_image_datasets.pannuke import PanNukeDataset
from bio_image_datasets.schuerch_dataset import SchuerchDataset
from bio_image_datasets.segpath_dataset import SegPath
from dotenv import load_dotenv
from torchmetrics.classification import MulticlassJaccardIndex

from benchmark.lmdb_dataset import LMDBDataset
from benchmark.unetr import UnetR


def normalize(x):  # noqa: D103
    return (x - x.min()) / (x.max() - x.min() + 1e-5)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # load the environment variables
    dotenv_path = Path(__file__).parents[1] / ".env"
    load_dotenv(dotenv_path=dotenv_path)
    num_classes = 7

    outdir = "/fast/AG_Kainmueller/outputs/test_unetr_{date}".format(date=datetime.now().strftime("%Y%m%d%H%M"))
    os.makedirs(outdir, exist_ok=True)

    model_name = "phikonv2"
    print(f"Instantiating model {model_name}...")
    model = UnetR(
        model_name=model_name,
        num_classes=num_classes,
    )
    model.cuda()
    model.eval()
    '''
    dataset = LMDBDataset(
        path="/fast/AG_Kainmueller/data/patho_foundation_model_bench_data/lizard_dataset/lizard_tiled_lmdb/",
    )

    images = []
    gts = []
    for img_idx in [1, 11, 111, 113]:
        sample = dataset[img_idx]
        images.append(torch.tensor(sample["image"], dtype=torch.float32).cuda())
        gts.append(np.array(sample["semantic_mask"], dtype=np.uint8))
    #print("gt", gts[0].shape, gts[0].min(), gts[0].max())
    '''
    images = [torch.zeros(3, 224, 224) for _ in range(2)]
    image = torch.stack(images).cuda()
    print("image", image.shape, image.min(), image.max())

    output = model(image)
    output = output.detach().cpu().numpy()
    print("output shape, min, max", output.shape, output.min(), output.max())

    seg_armax = np.argmax(output, axis=1, keepdims=True)

    # get the unique values and their counts
    unique, counts = np.unique(output[0, ...], return_counts=True)
    vals_count_seg = {el: count for el, count in zip(unique, counts) if count > 10}

    unique, counts = np.unique(seg_armax[0, ...], return_counts=True)
    vals_count_gt = {el: count for el, count in zip(unique, counts) if count > 10}

    print("GT", vals_count_gt)
    print("PRED", vals_count_seg)

    f, a = plt.subplots(1, 4)
    f.set_size_inches(25, 5)
    a[0].imshow(normalize(image[0, ...].cpu().numpy().transpose(1, 2, 0)))
    #a[1].imshow(normalize(gts[0]))
    a[2].imshow(normalize(output[0, :3, :, :].transpose(1, 2, 0)))
    a[3].imshow(normalize(seg_armax[0, 0, ...]))
    save_path = os.path.join(outdir, "segmented_image.png")
    print(f"Saving fig at {save_path}")
    plt.savefig(save_path, bbox_inches="tight")

    # Assuming `segmentation_logits` is the predicted segmentation map and `ground_truth` is the actual label
    score = MulticlassJaccardIndex(num_classes=num_classes, average="macro")
    #score_value = score(torch.tensor(seg_armax[0, 0, :, :]), torch.tensor(gts[0]))
    #print("Jaccard Score:", score_value)
