import os
from pathlib import Path

import torch
from benchmark.m2f_wrapper import BACKBONE_ARCHS

from torchmetrics.classification import MulticlassJaccardIndex

from backbones import encoder_decoder_m2f
from benchmark.m2f_wrapper import M2FSegmentationModel
from bio_image_datasets.lizard_dataset import LizardDataset
from bio_image_datasets.pannuke import PanNukeDataset
from bio_image_datasets.consep_dataset import ConSePDataset
from bio_image_datasets.schuerch_dataset import SchuerchDataset
from bio_image_datasets.segpath_dataset import SegPath
from dotenv import load_dotenv
from benchmark import utils 

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mmengine.registry import MODELS

    # load the environment variables
    dotenv_path = Path(__file__).parents[1] / ".env"
    load_dotenv(dotenv_path=dotenv_path)
    HF_TOKEN = os.getenv("HF_TOKEN")

    print("Print all available models in the registry")

    print(MODELS.module_dict.keys())
    num_classes = 8

    DINOV2_BASE_URL = "https://dl.fbaipublicfiles.com/dinov2"
    backbone_size = "small"
    backbone_arch = BACKBONE_ARCHS[backbone_size]
    backbone_name = f"dinov2_{backbone_arch}"
    head_scale_count = 4  # more scales: slower but better results, in (1,2,3,4,5)
    head_dataset = "voc2012"  # in ("ade20k", "voc2012")
    head_type = "ms"  # in ("ms, "linear")

    #head_config_url = f"{DINOV2_BASE_URL}/{backbone_name}/{backbone_name}_{head_dataset}_{head_type}_config.py"
    head_checkpoint_url = f"{DINOV2_BASE_URL}/{backbone_name}/{backbone_name}_{head_dataset}_{head_type}_head.pth"
    head_config_file = f"./config/{backbone_name}_{head_dataset}_{head_type}_config.py"

    print('head_checkpoint_url', head_checkpoint_url)

    print("Instantiating model...")
    model = M2FSegmentationModel(
        backbone_size=backbone_size,
        model_name="UNI",
        num_classes=num_classes,
        head_checkpoint_url=head_checkpoint_url,
        cfg_str=head_config_file,
        head_type=head_type,
        head_scale_count=head_scale_count,
        hf_token=HF_TOKEN,
    )
    dataset = LizardDataset(
        local_path="/fast/AG_Kainmueller/data/patho_foundation_model_bench_data/lizard_dataset/original_data",
    )
    sample = dataset[111]
    image = torch.Tensor(sample["image"]).cuda()
    ground_truth = sample["semantic_mask"]

    print('doing_inference...')
    segmentation_logits = model(image)
    segmented_image = utils.render_segmentation(segmentation_logits, head_scale_count)

    outdir = "../../outputs"
    os.makedirs(outdir, exist_ok=True)
    plt.imshow(segmented_image)
    plt.savefig(os.path.join(outdir, "segmented_image.png"), bbox_inches="tight")

    # Assuming `segmentation_logits` is the predicted segmentation map and `ground_truth` is the actual label
    score = MulticlassJaccardIndex(num_classes=num_classes, average="macro")
    score_value = score(segmentation_logits, ground_truth)
    print("Jaccard Score:", score_value)
