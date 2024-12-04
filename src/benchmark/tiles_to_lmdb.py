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


def save_tile_dict_to_lmdb(txn, tile_dict, lmdb_path):
    """Save a tile dictionary to an LMDB database.
    
    Args:
        txn: LMDB transaction object
        tile_dict: Dictionary containing the tile data
        lmdb_path: Path to the LMDB database
    """
    tile_name = tile_dict["tile_name"]
    sample_name = tile_dict["sample_name"]
    image = tile_dict["image"]
    semantic_mask = tile_dict["semantic_mask"]
    instance_mask = tile_dict.get("instance_mask", None)

    # Encode image as JPEG
    image_jpg_encoded = iio.imwrite("<bytes>", image, extension=".png")

    # Encode semantic mask as bytes
    semantic_mask_bytes = semantic_mask.tobytes()

    # Create metadata dictionary and encode as bytes
    metadata_dict = {
        "tile_name": tile_name,
        "sample_name": sample_name,
    }
    metadata_bytes = json.dumps(metadata_dict).encode("utf-8")

    # Create patch index
    patch_idx = tile_name.encode("utf-8")

    # Save image, semantic mask, and metadata to LMDB
    txn.put(patch_idx, image_jpg_encoded)
    txn.put(patch_idx + b"_semantic", semantic_mask_bytes)
    txn.put(patch_idx + b"_meta", metadata_bytes)

    # Save instance mask if it exists
    if instance_mask is not None:
        instance_mask_bytes = instance_mask.tobytes()
        txn.put(patch_idx + b"_instance", instance_mask_bytes)

    print(f"Saved tile {tile_name} to LMDB at {lmdb_path}")


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
            img = dataset.get_he(idx)
            inst_mask = dataset.get_instance_mask(idx)
            semantic_mask = dataset.get_semantic_mask(idx)
            sample_name = dataset.get_sample_name(idx)
            img_tiles = transform_to_tiles(img, tile_size=args.tile_size)
            inst_mask_tiles = transform_to_tiles(inst_mask, tile_size=args.tile_size)
            semantic_mask_tiles = transform_to_tiles(semantic_mask, tile_size=args.tile_size)
            tile_names = [f"{sample_name}_TILE_{i}" for i in range(len(img_tiles))]
            for idx in range(len(img_tiles)):
                tile_dict = {
                    "tile_name": tile_names[idx],
                    "sample_name": sample_name,
                    "image": img_tiles[idx],
                    "semantic_mask": semantic_mask_tiles[idx],
                }
                if inst_mask is not None:
                    tile_dict["instance_mask"] = inst_mask_tiles[idx]
                save_tile_to_lmdb(txn, tile_dict, lmdb_path)
            print(f"TOTAL #tiles {len(img_tiles)} FOR DATASET {dataset}")
            # TODO: add semantic mask, instance mask to lmdb tiling
            # save_tiles_to_lmdb(txn, tiles_dict, base_lmdb_dir)


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
    parser.add_argument(
        "--tile_size",
        type=int,
        help="Tile size",
        default=224,
    )
    return parser


if __name__ == "__main__":
    args_parser = get_args_parser()
    args = args_parser.parse_args()
    sys.exit(main(args))


# channel_imgs = Parallel(n_jobs=n_jobs)(
#    delayed(load_channel)(channel_path) for channel_path in channel_paths
# )
