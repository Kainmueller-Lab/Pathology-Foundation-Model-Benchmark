import os
import lmdb
import pickle

from tqdm import tqdm
import numpy as np
from bio_image_datasets.lizard_dataset import LizardDataset
from benchmark.split_to_tiles import transform_to_tiles


def create_lmdb_database(dataset, lmdb_path, tile_size=224):
    """
    Create an LMDB database for a dataset

    Args:
        dataset: Instance of a dataset.
        lmdb_path: Path to the LMDB file to create.
        tile_size: The size of the tiles to create.
    """
    # Estimate max LMDB map size as 1 TB
    map_size = 2**40  # Just use 1 TB

    # Create the LMDB environment
    env = lmdb.open(lmdb_path, map_size=map_size)

    tile_idx = 0  # Global tile index for keys

    with env.begin(write=True) as txn:
        # Loop over the dataset
        for idx in tqdm(range(len(dataset))):
            # Extract the relevant data from the dataset
            img = dataset.get_he(idx)
            inst_mask = dataset.get_instance_mask(idx)
            semantic_mask = dataset.get_semantic_mask(idx)
            sample_name = dataset.get_sample_name(idx)
            img_tiles = transform_to_tiles(img, tile_size=tile_size)
            if inst_mask is not None:
                inst_mask_tiles = transform_to_tiles(inst_mask, tile_size=tile_size)
            semantic_mask_tiles = transform_to_tiles(semantic_mask, tile_size=tile_size)
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
                # Serialize the tile dictionary
                tile_data = pickle.dumps(tile_dict)

                # Use the tile_idx as the key, converted to bytes
                key = f"{tile_idx:08}".encode("ascii")

                # Put the data into LMDB
                txn.put(key, tile_data)

                tile_idx += 1

                if tile_idx % 1000 == 0:
                    print(f"Processed {tile_idx} tiles...")

    env.close()
    print(f"LMDB file created at {lmdb_path} with {tile_idx} tiles.")


if __name__ == "__main__":
    import argparse

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Create LMDB files for LizardDataset tiles.")
    parser.add_argument(
        "--local_path",
        type=str,
        default="~/projects/lab_hackathon_2024/Bio-Image-Datasets/downloads",
        help="Local path to the dataset.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="~/projects/lab_hackathon_2024/Bio-Image-Datasets/downloads/lizard_lmdb/lizard_tilee.lmdb",
        help="Output Path for LMDB files.",
    )
    parser.add_argument("--tile_size", type=int, default=224, help="Tile size (default: 224).")

    args = parser.parse_args()

    # Expand user in paths
    args.local_path = os.path.expanduser(args.local_path)
    args.output_path = os.path.expanduser(args.output_path)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    # Instantiate the dataset
    dataset = LizardDataset(local_path=args.local_path)

    # Create LMDB database
    create_lmdb_database(dataset, lmdb_path=args.output_path, tile_size=args.tile_size)
