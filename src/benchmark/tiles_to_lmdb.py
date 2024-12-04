import argparse
import json
import os
from pathlib import Path
import sys

from benchmark.split_to_tiles import transform_to_tiles
import imageio as io
import imageio.v3 as iio
import lmdb
import numpy as np
from tqdm import tqdm

import bio_image_datasets
from bio_image_datasets.lizard_dataset import LizardDataset
from bio_image_datasets.pannuke import PanNukeDataset
from bio_image_datasets.schuerch_dataset import (
    SchuerchDataset,
    transform_semantic_mask,
    coarse_mapping,
    semantic_id_old_to_new,
)
from bio_image_datasets.segpath_dataset import SegPathDataset
from bio_image_datasets.consep_dataset import ConSePDataset


MAP_SIZE_IMG = int(1e12)  # 1TB
MAP_SIZE_META = int(1e8)  # 100MB


def normalize(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-5)


def save_tiles_to_lmdb(tiles, lmdb_dir, split_name="train", file_idx=0):
        for img_idx, fov in tqdm(enumerate(tiles), total=len(tiles)):
            fov_name_cleaned = "".join(e for e in str(fov) if e.isalnum())
            do_print = img_idx % 50 == 0
            if do_print:
                print(f'idx: {img_idx}/{len(fovs)}, fov: "{fov_name_cleaned}"')

            img_idx = f"{img_idx:04d}"
            metadata_dict = {}

            fov_path = os.path.join(path, fov)

            # get metadata
            metadata_dict["fov"] = fov
            metadata_bytes = json.dumps(metadata_dict).encode("utf-8")

            # get segmentation mask
            # segmentation mask has to be uint16 because of values of to ~3000 segments
            # Thus, cannot be jpeg compressed
            segmentation_mask = iio.imread(segmentation_path).squeeze().astype(np.uint16)

            crop_jpg_encoded = iio.imwrite(
                "<bytes>",
                crop,
                extension=".jpeg",
            )
            patch_idx = f"{img_idx}_p{x_crop_idx + y_crop_idx}"
            txn_imgs.put(
                crop_ch_idx_bytes,
                crop_jpg_encoded,
            )

            patch_idx_bytes = patch_idx.encode("utf-8")
            txn_labels.put(
                patch_idx_bytes,
                crop_mask.tobytes(),
            )
            txn_meta.put(
                patch_idx_bytes,
                metadata_bytes,
            )

    env.close()
    print(f"FINISHED DATASET {dataset}, SAVED AT: {dataset_lmdb_dir}")


def load_dataset(dataset_name: str, dataset_path: Path):
    if dataset_name == "lizard":
        return LizardDataset(dataset_path)
    elif dataset_name == "pannuke":
        return PanNukeDataset(dataset_path)
    elif dataset_name == "schuerch":
        return SchuerchDataset(
            dataset_path
        )  # TODO: make class attributes: transform_semantic_mask, coarse_mapping, semantic_id_old_to_new,
    elif dataset_name == "segpath":
        return SegPathDataset(dataset_path)
    elif dataset_name == "consep":
        return ConSePDataset(dataset_path)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def main(args):
    start_fov_idx = args.start_fov_idx
    end_fov_idx = args.end_fov_idx
    dataset_name = args.dataset_name
    dataset_path = args.dataset_path

    dataset = load_dataset(dataset_name, dataset_path)

    lmdb_dir = dataset_path + args.base_lmdb_dir_name
    os.makedirs(lmdb_dir, exist_ok=True)

    print(f"PROCESSING DATASET {dataset} stored in {dataset_path}...")

    lmdb_path = os.path.join(
        lmdb_dir,
        str(lmdb_dir) + f"-{start_fov_idx}-{end_fov_idx}_images",
    )

    os.makedirs(lmdb_path, exist_ok=True)
    env = lmdb.open(lmdb_path, map_size=MAP_SIZE_IMG)

    with env.begin(write=True) as txn:
        
        # TODO: limit iteration with start_fov_idx, end_fov_idx
        for idx in tqdm(range(len(dataset)), total=len(dataset)):
            img = dataset.get_image(idx)
            inst_mask = dataset.get_instance_mask(idx)
            semantic_mask = dataset.get_semantic_mask(idx)

            # TODO: handle segpath that doesnt have inst_mask
            img_tiles = transform_to_tiles(img)
            inst_mask_tiles = transform_to_tiles(inst_mask)
            semantic_mask_tiles = transform_to_tiles(semantic_mask)
            tiles_dict = {
                "image": img_tiles,
                "instance_mask": inst_mask_tiles,
                "semantic_mask": semantic_mask_tiles,
            }
            print(f"TOTAL #tiles {len(img_tiles)} FOR DATASET {dataset}")
            # TODO: add semantic mask, instance mask to lmdb tiling
            save_tiles_to_lmdb(txn, tiles_dict, base_lmdb_dir)


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--do_test_run",
        action=argparse.BooleanOptionalAction,
        help="Toggle test run with small subset of dataset",
        default=False,
    )
    parser.add_argument(
        "--start_fov_idx",
        type=int,
        help="Start index of FOVs to process",
        default=0,
    )
    parser.add_argument(
        "--end_fov_idx",
        type=int,
        help="End index of FOVs to process",
        default=-1,
    )
    parser.add_argument(
        "--base_lmdb_dir_name",
        type=str,
        help="Base lmdb dir name",
        default="_lmdb",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="Dataset path to load",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="Dataset name to load",
    )

    return parser


if __name__ == "__main__":
    args_parser = get_args_parser()
    args = args_parser.parse_args()
    sys.exit(main(args))


# channel_imgs = Parallel(n_jobs=n_jobs)(
#    delayed(load_channel)(channel_path) for channel_path in channel_paths
# )
