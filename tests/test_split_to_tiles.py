import os
import shutil
import tempfile
import unittest

import numpy as np
import scipy.io as sio
from bio_image_datasets.lizard_dataset import LizardDataset
from PIL import Image

from benchmark.dataset.split_to_tiles import center_pad_to_size, transform_to_tiles


class TestSplitToTiles(unittest.TestCase):
    """Test suite for the center_pad_to_size and transform_to_tiles functions."""

    @classmethod
    def setUpClass(cls):
        """Set up mock data for all tests."""
        # Create a temporary directory
        cls.temp_dir = tempfile.mkdtemp()

        # Set up mock data directories
        cls.mock_local_path = cls.temp_dir
        image_dirs = [
            os.path.join(cls.mock_local_path, "lizard_images1", "Lizard_Images1"),
            os.path.join(cls.mock_local_path, "lizard_images2", "Lizard_Images2"),
        ]
        label_dir = os.path.join(cls.mock_local_path, "lizard_labels", "Lizard_Labels", "Labels")

        # Create directories
        for dir_path in image_dirs + [label_dir]:
            os.makedirs(dir_path, exist_ok=True)

        # Create mock image and label files
        cls.sample_filenames = ["sample1.png", "sample2.png"]
        for image_dir in image_dirs:
            for filename in cls.sample_filenames:
                # Create a mock image file (e.g., 500x500 pixels)
                image_path = os.path.join(image_dir, filename)
                mock_image = Image.new("RGB", (500, 500), color=(255, 0, 0))
                mock_image.save(image_path)

                # Create a corresponding mock .mat label file
                label_name = filename.replace(".png", ".mat")
                label_path = os.path.join(label_dir, label_name)
                # Create a mock instance map with some random instances
                inst_map = np.zeros((500, 500), dtype=np.int32)
                # Add some instances
                inst_map[100:200, 100:200] = 1
                inst_map[300:400, 300:400] = 2
                mock_label_data = {
                    "inst_map": inst_map,
                    "class": np.array([1, 2]),
                    "id": np.array([1, 2]),
                    "bbox": np.array([[100, 200, 100, 200], [300, 400, 300, 400]]),
                    "centroid": np.array([[150, 150], [350, 350]]),
                }
                sio.savemat(label_path, mock_label_data)

        # Instantiate the dataset with the mock data
        cls.dataset = LizardDataset(local_path=cls.mock_local_path)
        cls.sample_idx = 0  # Index of the sample to test
        cls.total_samples = len(cls.dataset)

    @classmethod
    def tearDownClass(cls):
        """Clean up the temporary directory after tests."""
        shutil.rmtree(cls.temp_dir)

    def test_center_pad_to_size(self):
        """Test the center_pad_to_size function."""
        # Test with a 2D array and desired_shape as tuple
        array_2d = np.ones((100, 150))
        desired_shape_2d = (224, 224)
        padded_array_2d = center_pad_to_size(array_2d, desired_shape_2d)
        self.assertEqual(
            padded_array_2d.shape,
            desired_shape_2d,
            "2D array should be padded to desired shape.",
        )
        self.assertTrue(
            np.array_equal(padded_array_2d[62:162, 37:187], array_2d),
            "Original array should be centered.",
        )

        # Test with desired_shape as int
        desired_shape_int = 224
        padded_array_2d_int = center_pad_to_size(array_2d, desired_shape_int)
        self.assertEqual(
            padded_array_2d_int.shape,
            desired_shape_2d,
            "2D array should be padded to desired shape with int input.",
        )
        self.assertTrue(
            np.array_equal(padded_array_2d_int[62:162, 37:187], array_2d),
            "Original array should be centered with int input.",
        )

        # Test with a 3D array (e.g., image with channels)
        array_3d = np.ones((3, 100, 150))
        desired_shape_3d = (3, 224, 224)
        padded_array_3d = center_pad_to_size(array_3d, desired_shape_3d)
        self.assertEqual(
            padded_array_3d.shape,
            desired_shape_3d,
            "3D array should be padded to desired shape.",
        )
        self.assertTrue(
            np.array_equal(padded_array_3d[:, 62:162, 37:187], array_3d),
            "Original array should be centered.",
        )

        # Test with desired_shape as int on a 3D array
        padded_array_3d_int = center_pad_to_size(array_3d, desired_shape_int)
        self.assertEqual(
            padded_array_3d_int.shape,
            desired_shape_3d,
            "3D array should be padded to desired shape with int input.",
        )
        self.assertTrue(
            np.array_equal(padded_array_3d_int[:, 62:162, 37:187], array_3d),
            "Original array should be centered with int input.",
        )

        # Test with no padding needed
        array_no_pad = np.ones((224, 224))
        padded_array_no_pad = center_pad_to_size(array_no_pad, (224, 224))
        self.assertEqual(
            padded_array_no_pad.shape,
            (224, 224),
            "Array should remain the same size if no padding needed.",
        )
        self.assertTrue(
            np.array_equal(padded_array_no_pad, array_no_pad),
            "Original array should remain unchanged.",
        )

    def test_center_pad_to_size_errors(self):
        """Test that center_pad_to_size raises errors with invalid inputs."""
        array = np.ones((100, 100))
        # Test with negative desired shape
        with self.assertRaises(AssertionError):
            center_pad_to_size(array, (-100, 100))

    def test_transform_to_tiles(self):
        """Test the transform_to_tiles function."""
        # Use the dataset with the mock data
        sample = self.dataset[self.sample_idx]
        image = sample["image"]
        semantic_mask = sample.get("semantic_mask")
        instance_mask = sample.get("instance_mask")

        # Transform the arrays into tiles
        image_tiles = transform_to_tiles(image, tile_size=224)
        semantic_mask_tiles = transform_to_tiles(semantic_mask, tile_size=224) if semantic_mask is not None else None
        instance_mask_tiles = (
            transform_to_tiles(instance_mask, tile_size=224, renumber_instances=True)
            if instance_mask is not None
            else None
        )

        # Verify that the number of tiles is as expected
        self.assertIsInstance(image_tiles, list, "image_tiles should be a list.")
        self.assertTrue(len(image_tiles) > 0, "There should be at least one image tile.")
        num_tiles = len(image_tiles)
        if semantic_mask_tiles is not None:
            self.assertEqual(
                len(semantic_mask_tiles),
                num_tiles,
                "semantic_mask_tiles should match image_tiles in length.",
            )
        if instance_mask_tiles is not None:
            self.assertEqual(
                len(instance_mask_tiles),
                num_tiles,
                "instance_mask_tiles should match image_tiles in length.",
            )

        # Test properties of each tile
        for idx, img_tile in enumerate(image_tiles):
            # Check image tile properties
            self.assertIsInstance(img_tile, np.ndarray, "Each image tile should be a NumPy array.")
            self.assertEqual(
                img_tile.shape,
                (3, 224, 224),
                "Image tile should be of shape (3, 224, 224).",
            )

            # Check semantic mask tile if available
            if semantic_mask_tiles is not None:
                semantic_mask_tile = semantic_mask_tiles[idx]
                self.assertIsInstance(
                    semantic_mask_tile,
                    np.ndarray,
                    "Each semantic mask tile should be a NumPy array.",
                )
                self.assertEqual(
                    semantic_mask_tile.shape,
                    (224, 224),
                    "Semantic mask tile should be of shape (224, 224).",
                )

            # Check instance mask tile if available
            if instance_mask_tiles is not None:
                instance_mask_tile = instance_mask_tiles[idx]
                self.assertIsInstance(
                    instance_mask_tile,
                    np.ndarray,
                    "Each instance mask tile should be a NumPy array.",
                )
                self.assertEqual(
                    instance_mask_tile.shape,
                    (224, 224),
                    "Instance mask tile should be of shape (224, 224).",
                )

                # Verify that the instance IDs have been renumbered starting from 1
                unique_ids = np.unique(instance_mask_tile)
                self.assertTrue(
                    np.all(unique_ids >= 0),
                    "Instance IDs should be non-negative integers.",
                )
                if len(unique_ids) > 1:  # If there are instances in the tile
                    self.assertEqual(
                        unique_ids.min(),
                        0,
                        "Instance IDs should start from 0 (background).",
                    )
                    self.assertEqual(
                        unique_ids.max(),
                        len(unique_ids) - 1,
                        "Instance IDs should be sequential starting from 0.",
                    )

    def test_transform_to_tiles_edge_cases(self):
        """Test transform_to_tiles function with edge cases."""
        # Use the dataset with the mock data
        sample = self.dataset[self.sample_idx]
        image = sample["image"]

        # Test with tile size larger than image dimensions
        tiles = transform_to_tiles(image, tile_size=600)
        self.assertEqual(
            len(tiles),
            1,
            "Should return one tile when tile size exceeds image dimensions.",
        )
        tile = tiles[0]
        self.assertEqual(tile.shape, (3, 600, 600), "Tile should be padded to (3, 600, 600).")

        # Test with tile size equal to image dimensions
        tiles = transform_to_tiles(image, tile_size=500)
        self.assertEqual(
            len(tiles),
            1,
            "Should return one tile when tile size equals image dimensions.",
        )
        tile = tiles[0]
        self.assertEqual(
            tile.shape,
            (3, 500, 500),
            "Tile should match image dimensions without padding.",
        )

        # Test with tile size that doesn't divide image dimensions evenly
        tiles = transform_to_tiles(image, tile_size=333)
        self.assertEqual(
            len(tiles),
            4,
            "Should return four tiles for 500x500 image with tile size 333.",
        )
        for tile in tiles:
            self.assertEqual(tile.shape, (3, 333, 333), "Tiles should be padded to (3, 333, 333).")


if __name__ == "__main__":
    unittest.main()
