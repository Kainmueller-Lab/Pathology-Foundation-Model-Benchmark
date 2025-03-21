import numpy as np

from benchmark.utils.utils import get_height_width, to_tuple


def center_pad_to_size(array, desired_shape):
    """
    Center pad an array with zeros to reach the desired shape.

    Args:
        array (numpy.ndarray): The array to pad.
        desired_shape (tuple of int): The desired shape after padding. Must have the same number of dimensions as `array`.

    Returns
    -------
        numpy.ndarray: The center-padded array.

    Raises
    ------
        AssertionError: If `array` and `desired_shape` have different numbers of dimensions or if any dimension in `desired_shape` is negative.
    """
    current_shape = array.shape
    desired_shape = to_tuple(desired_shape)

    if len(current_shape) != len(desired_shape):
        # We assume that the first dimension is the channel dimension
        desired_shape = (current_shape[0],) + desired_shape
    assert all(dim >= 0 for dim in desired_shape), "Desired shape values must be non-negative."

    pad_widths = []
    for current_size, desired_size in zip(current_shape, desired_shape):
        total_pad = max(desired_size - current_size, 0)
        pad_before = total_pad // 2
        pad_after = total_pad - pad_before
        pad_widths.append((pad_before, pad_after))

    padded_array = np.pad(array, pad_width=pad_widths, mode="constant", constant_values=0)
    return padded_array


def transform_to_tiles(array, tile_size=224, renumber_instances=False):
    """
    Split an array into tiles of specified size, center-padded as needed.

    Args:
        array (numpy.ndarray): The array to split into tiles. Expected to be at least 2D (height x width)
            or 3D (channels x height x width).
        tile_size (int or tuple of int, optional): The desired tile size after padding. If an integer,
            the same size is used for height and width. If a tuple, should be (height, width). Defaults to 224.
        renumber_instances (bool, optional): Whether to renumber instance IDs in each tile to ensure
            sequential numbering starting from 1. Useful for instance segmentation masks. Defaults to False.

    Returns
    -------
        list of numpy.ndarray: A list containing the tiles of the array.

    """
    # Get the height and width of the image
    height, width = get_height_width(array)

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

            # Extract the tile from the image or masks
            if len(array.shape) == 2:
                arr_tile = array[y_start:y_end, x_start:x_end]
            else:
                arr_tile = array[:, y_start:y_end, x_start:x_end]

            # Center pad the tiles to the desired size
            arr_tile = center_pad_to_size(arr_tile, (tile_size, tile_size))

            if renumber_instances:
                # Renumber the instances in the inst_tile
                unique_instances = np.unique(arr_tile)
                unique_instances = unique_instances[unique_instances != 0]  # Exclude background
                inst_map = np.zeros_like(arr_tile)
                for new_id, old_id in enumerate(unique_instances, start=1):
                    inst_map[arr_tile == old_id] = new_id
                arr_tile = inst_map

            # Append the tile dictionary to the list
            tiles.append(arr_tile)
            tile_idx += 1

    return tiles


if __name__ == "__main__":
    import os

    from bio_image_datasets.lizard_dataset import LizardDataset
    from matplotlib import pyplot as plt

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
    sample_name = dataset.get_sample_name(sample_idx)

    # Extract data from the sample
    image = sample["image"]  # C x H x W
    semantic_mask = sample.get("semantic_mask")  # H x W
    instance_mask = sample.get("instance_mask")  # H x W

    # Transform the sample into tiles
    image_tiles = transform_to_tiles(image, tile_size=224)
    semantic_mask_tiles = transform_to_tiles(semantic_mask, tile_size=224) if semantic_mask is not None else None
    instance_mask_tiles = (
        transform_to_tiles(instance_mask, tile_size=224, renumber_instances=True) if instance_mask is not None else None
    )

    print(f"Number of tiles created: {len(image_tiles)}")

    # Visualize each tile
    for idx, img_tile in enumerate(image_tiles):
        tile_sample_name = f"{sample_name}_tile_{idx}"
        print(f"\nProcessing tile: {tile_sample_name}")

        # Create a subfolder for each tile
        tile_folder = os.path.join(visualization_folder, tile_sample_name)
        os.makedirs(tile_folder, exist_ok=True)

        # Prepare image tile for display
        img_tile_disp = np.transpose(img_tile, (1, 2, 0))  # Convert CHW to HWC

        # Get corresponding semantic and instance mask tiles
        semantic_mask_tile = semantic_mask_tiles[idx] if semantic_mask_tiles is not None else None
        instance_mask_tile = instance_mask_tiles[idx] if instance_mask_tiles is not None else None

        # Save the H&E image
        plt.figure()
        plt.imshow(img_tile_disp)
        plt.title(f"H&E Image - {tile_sample_name}")
        plt.axis("off")
        he_image_path = os.path.join(tile_folder, f"{tile_sample_name}_he_image.png")
        plt.savefig(he_image_path)
        plt.close()
        print(f"H&E image saved to: {he_image_path}")

        # Save the semantic mask if available
        if semantic_mask_tile is not None:
            plt.figure()
            plt.imshow(semantic_mask_tile, cmap="jet")
            plt.title(f"Semantic Mask - {tile_sample_name}")
            plt.axis("off")
            semantic_mask_path = os.path.join(tile_folder, f"{tile_sample_name}_semantic_mask.png")
            plt.savefig(semantic_mask_path)
            plt.close()
            print(f"Semantic mask saved to: {semantic_mask_path}")

        # Save the instance mask if available
        if instance_mask_tile is not None:
            plt.figure()
            plt.imshow(instance_mask_tile, cmap="jet")
            plt.title(f"Instance Mask - {tile_sample_name}")
            plt.axis("off")
            instance_mask_path = os.path.join(tile_folder, f"{tile_sample_name}_instance_mask.png")
            plt.savefig(instance_mask_path)
            plt.close()
            print(f"Instance mask saved to: {instance_mask_path}")

        # Save the combined visualization
        num_subplots = 1 + int(semantic_mask_tile is not None) + int(instance_mask_tile is not None)
        fig, ax = plt.subplots(1, num_subplots, figsize=(5 * num_subplots, 5))

        ax_idx = 0
        ax[ax_idx].imshow(img_tile_disp)
        ax[ax_idx].set_title("H&E Image")
        ax[ax_idx].axis("off")
        ax_idx += 1

        if semantic_mask_tile is not None:
            ax[ax_idx].imshow(semantic_mask_tile, cmap="jet")
            ax[ax_idx].set_title("Semantic Mask")
            ax[ax_idx].axis("off")
            ax_idx += 1

        if instance_mask_tile is not None:
            ax[ax_idx].imshow(instance_mask_tile, cmap="jet")
            ax[ax_idx].set_title("Instance Mask")
            ax[ax_idx].axis("off")

        full_sample_path = os.path.join(tile_folder, f"{tile_sample_name}_full_sample.png")
        plt.suptitle(f"Tile Visualization - {tile_sample_name}")
        plt.tight_layout()
        plt.savefig(full_sample_path)
        plt.close()
        print(f"Combined visualization saved to: {full_sample_path}")
