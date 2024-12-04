import numpy as np


def center_pad_to_size(array, desired_shape):
    """
    Center pad an array with zeros to reach the desired shape.

    Args:
        array (numpy.ndarray): The array to pad.
        desired_shape (tuple of int): The desired shape after padding. Must have the same number of dimensions as `array`.

    Returns:
        numpy.ndarray: The center-padded array.

    Raises:
        AssertionError: If `array` and `desired_shape` have different numbers of dimensions or if any dimension in `desired_shape` is negative.
    """
    current_shape = array.shape
    assert len(current_shape) == len(desired_shape), "Array and desired shape must have the same number of dimensions."
    assert all(dim >= 0 for dim in desired_shape), "Desired shape values must be non-negative."

    pad_widths = []
    for current_size, desired_size in zip(current_shape, desired_shape):
        total_pad = max(desired_size - current_size, 0)
        pad_before = total_pad // 2
        pad_after = total_pad - pad_before
        pad_widths.append((pad_before, pad_after))

    padded_array = np.pad(array, pad_width=pad_widths, mode="constant", constant_values=0)
    return padded_array


def transform_to_tiles(sample, tile_size=224):
    """
    Transform a sample into center-padded tiles of specified size.

    Args:
        sample (dict): A dictionary containing the sample data with keys 'image', 'instance_mask', 'semantic_mask', and 'sample_name'.
        tile_size (int, optional): The desired tile size after padding. Defaults to 224.

    Returns:
        list of dict: A list of dictionaries, each containing a tile of the image, instance mask, semantic mask, and the sample name for that tile.

    Raises:
        KeyError: If required keys are missing in the sample dictionary.
    """
    # Get the data
    image = sample["image"]  # Image in CHW format
    inst_mask = sample["instance_mask"]
    seg_mask = sample["semantic_mask"]
    sample_name = sample["sample_name"]

    # Get the dimensions of the image
    _, height, width = image.shape  # C x H x W

    # Decide on the number of tiles along each dimension
    n_tiles_y = int(np.ceil(height / tile_size))
    n_tiles_x = int(np.ceil(width / tile_size))

    # Compute the tile boundaries to evenly cover the image
    y_indices = np.linspace(0, height, n_tiles_y + 1, dtype=int)
    x_indices = np.linspace(0, width, n_tiles_x + 1, dtype=int)

    tiles = []
    tile_idx = 0
    for i in range(n_tiles_y):
        for j in range(n_tiles_x):
            y_start = y_indices[i]
            y_end = y_indices[i + 1]
            x_start = x_indices[j]
            x_end = x_indices[j + 1]

            # Extract the tile from the image and masks
            img_tile = image[:, y_start:y_end, x_start:x_end]
            inst_tile = inst_mask[y_start:y_end, x_start:x_end]
            seg_tile = seg_mask[y_start:y_end, x_start:x_end]

            # Center pad the tiles to the desired size
            img_tile = center_pad_to_size(img_tile, (image.shape[0], tile_size, tile_size))
            inst_tile = center_pad_to_size(inst_tile, (tile_size, tile_size))
            seg_tile = center_pad_to_size(seg_tile, (tile_size, tile_size))

            # Renumber the instances in the inst_tile
            unique_instances = np.unique(inst_tile)
            unique_instances = unique_instances[unique_instances != 0]  # Exclude background
            inst_map = np.zeros_like(inst_tile)
            for new_id, old_id in enumerate(unique_instances, start=1):
                inst_map[inst_tile == old_id] = new_id

            # Create the tile sample name
            tile_sample_name = f"{sample_name}_tile_{tile_idx}"

            # Create the tile dictionary
            tile_dict = {
                "image": img_tile,
                "semantic_mask": seg_tile,
                "instance_mask": inst_map,
                "sample_name": tile_sample_name,
            }

            # Append the tile dictionary to the list
            tiles.append(tile_dict)
            tile_idx += 1

    return tiles


if __name__ == "__main__":
    import os
    from matplotlib import pyplot as plt
    from bio_image_datasets.lizard_dataset import LizardDataset


    # Define the folder to save visualizations
    visualization_folder = "./visualizations"
    os.makedirs(visualization_folder, exist_ok=True)

    # Instantiate the dataset
    dataset = LizardDataset(
        local_path="~/projects/lab_hackathon_2024/Bio-Image-Datasets/downloads",
    )

    # Sample index to visualize
    sample_idx = 111
    print(f"\nVisualizing tiles for sample index: {sample_idx}")

    # Get the sample
    sample = dataset[sample_idx]
    sample["sample_name"] = dataset.get_sample_name(sample_idx)

    # Transform the sample into tiles
    tiles = transform_to_tiles(sample, tile_size=224)
    print(f"Number of tiles created: {len(tiles)}")

    # Visualize each tile
    for tile in tiles:
        tile_sample_name = tile["sample_name"]
        print(f"\nProcessing tile: {tile_sample_name}")

        # Extract data from the tile
        img_tile = tile["image"]
        semantic_mask_tile = tile["semantic_mask"]
        instance_mask_tile = tile["instance_mask"]

        # Create a subfolder for each tile
        tile_folder = os.path.join(visualization_folder, tile_sample_name)
        os.makedirs(tile_folder, exist_ok=True)

        # Save the H&E image
        plt.figure()
        img_tile_disp = np.transpose(img_tile, (1, 2, 0))  # Convert CHW to HWC
        plt.imshow(img_tile_disp)
        plt.title(f"H&E Image - {tile_sample_name}")
        plt.axis("off")
        he_image_path = os.path.join(tile_folder, f"{tile_sample_name}_he_image.png")
        plt.savefig(he_image_path)
        plt.close()
        print(f"H&E image saved to: {he_image_path}")

        # Save the semantic mask
        plt.figure()
        plt.imshow(semantic_mask_tile, cmap="jet")
        plt.title(f"Semantic Mask - {tile_sample_name}")
        plt.axis("off")
        semantic_mask_path = os.path.join(tile_folder, f"{tile_sample_name}_semantic_mask.png")
        plt.savefig(semantic_mask_path)
        plt.close()
        print(f"Semantic mask saved to: {semantic_mask_path}")

        # Save the instance mask
        plt.figure()
        plt.imshow(instance_mask_tile, cmap="jet")
        plt.title(f"Instance Mask - {tile_sample_name}")
        plt.axis("off")
        instance_mask_path = os.path.join(tile_folder, f"{tile_sample_name}_instance_mask.png")
        plt.savefig(instance_mask_path)
        plt.close()
        print(f"Instance mask saved to: {instance_mask_path}")

        # Save the combined visualization
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(img_tile_disp)
        ax[0].set_title("H&E Image")
        ax[0].axis("off")

        ax[1].imshow(semantic_mask_tile, cmap="jet")
        ax[1].set_title("Semantic Mask")
        ax[1].axis("off")

        ax[2].imshow(instance_mask_tile, cmap="jet")
        ax[2].set_title("Instance Mask")
        ax[2].axis("off")

        full_sample_path = os.path.join(tile_folder, f"{tile_sample_name}_full_sample.png")
        plt.suptitle(f"Tile Visualization - {tile_sample_name}")
        plt.tight_layout()
        plt.savefig(full_sample_path)
        plt.close()
        print(f"Combined visualization saved to: {full_sample_path}")
