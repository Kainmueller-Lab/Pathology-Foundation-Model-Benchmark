import unittest
import numpy as np
import os
import tempfile
import shutil
from bio_image_datasets.lizard_dataset import LizardDataset
from benchmark.split_to_tiles import center_pad_to_size, transform_to_tiles
import scipy.io as sio
from PIL import Image


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
        # Test with a 2D array
        array_2d = np.ones((100, 150))
        desired_shape_2d = (224, 224)
        padded_array_2d = center_pad_to_size(array_2d, desired_shape_2d)
        self.assertEqual(padded_array_2d.shape, desired_shape_2d, "2D array should be padded to desired shape.")
        self.assertTrue(np.array_equal(padded_array_2d[62:162, 37:187], array_2d), "Original array should be centered.")

        # Test with a 3D array (e.g., image with channels)
        array_3d = np.ones((3, 100, 150))
        desired_shape_3d = (3, 224, 224)
        padded_array_3d = center_pad_to_size(array_3d, desired_shape_3d)
        self.assertEqual(padded_array_3d.shape, desired_shape_3d, "3D array should be padded to desired shape.")
        self.assertTrue(
            np.array_equal(padded_array_3d[:, 62:162, 37:187], array_3d), "Original array should be centered."
        )

        # Test with no padding needed
        array_no_pad = np.ones((224, 224))
        padded_array_no_pad = center_pad_to_size(array_no_pad, (224, 224))
        self.assertEqual(
            padded_array_no_pad.shape, (224, 224), "Array should remain the same size if no padding needed."
        )
        self.assertTrue(np.array_equal(padded_array_no_pad, array_no_pad), "Original array should remain unchanged.")

    def test_transform_to_tiles(self):
        """Test the transform_to_tiles function."""
        # Use the dataset with the mock data
        sample = self.dataset[self.sample_idx]
        sample["sample_name"] = self.dataset.get_sample_name(self.sample_idx)
        tiles = transform_to_tiles(sample, tile_size=224)
        self.assertIsInstance(tiles, list, "Tiles should be returned as a list.")
        self.assertTrue(len(tiles) > 0, "There should be at least one tile.")

        # Test properties of each tile
        for tile in tiles:
            # Check that tile is a dictionary with expected keys
            expected_keys = {"image", "semantic_mask", "instance_mask", "sample_name"}
            self.assertIsInstance(tile, dict, "Each tile should be a dictionary.")
            self.assertTrue(expected_keys.issubset(tile.keys()), "Tile dictionary should contain all expected keys.")

            # Check image properties
            img_tile = tile["image"]
            self.assertIsInstance(img_tile, np.ndarray, "Tile image should be a NumPy array.")
            self.assertEqual(img_tile.shape, (3, 224, 224), "Tile image should be of shape (3, 224, 224).")

            # Check semantic mask properties
            semantic_mask_tile = tile["semantic_mask"]
            self.assertIsInstance(semantic_mask_tile, np.ndarray, "Tile semantic mask should be a NumPy array.")
            self.assertEqual(semantic_mask_tile.shape, (224, 224), "Tile semantic mask should be of shape (224, 224).")

            # Check instance mask properties
            instance_mask_tile = tile["instance_mask"]
            self.assertIsInstance(instance_mask_tile, np.ndarray, "Tile instance mask should be a NumPy array.")
            self.assertEqual(instance_mask_tile.shape, (224, 224), "Tile instance mask should be of shape (224, 224).")

            # Check sample name
            sample_name = tile["sample_name"]
            self.assertIsInstance(sample_name, str, "Tile sample name should be a string.")
            self.assertTrue(len(sample_name) > 0, "Tile sample name should not be empty.")

            # Verify that the instance IDs have been renumbered starting from 1
            unique_ids = np.unique(instance_mask_tile)
            self.assertTrue(np.all(unique_ids >= 0), "Instance IDs should be non-negative integers.")
            if len(unique_ids) > 1:  # If there are instances in the tile
                self.assertEqual(unique_ids.min(), 0, "Instance IDs should start from 0 (background).")
                self.assertEqual(
                    unique_ids.max(), len(unique_ids) - 1, "Instance IDs should be sequential starting from 0."
                )

    def test_transform_to_tiles_edge_cases(self):
        """Test transform_to_tiles function with edge cases."""
        # Use the dataset with the mock data
        sample = self.dataset[self.sample_idx]
        sample["sample_name"] = self.dataset.get_sample_name(self.sample_idx)

        # Test with tile size larger than image dimensions
        tiles = transform_to_tiles(sample, tile_size=600)
        self.assertEqual(len(tiles), 1, "Should return one tile when tile size exceeds image dimensions.")
        tile = tiles[0]
        self.assertEqual(tile["image"].shape, (3, 600, 600), "Tile should be padded to (3, 600, 600).")

        # Test with tile size equal to image dimensions
        tiles = transform_to_tiles(sample, tile_size=500)
        self.assertEqual(len(tiles), 1, "Should return one tile when tile size equals image dimensions.")
        tile = tiles[0]
        self.assertEqual(tile["image"].shape, (3, 500, 500), "Tile should match image dimensions without padding.")

        # Test with tile size that doesn't divide image dimensions evenly
        tiles = transform_to_tiles(sample, tile_size=333)
        self.assertEqual(len(tiles), 4, "Should return four tiles for 500x500 image with tile size 333.")
        for tile in tiles:
            self.assertEqual(tile["image"].shape, (3, 333, 333), "Tiles should be padded to (3, 333, 333).")

    def test_center_pad_to_size_errors(self):
        """Test that center_pad_to_size raises errors with invalid inputs."""
        array = np.ones((100, 100))
        # Test with mismatched dimensions
        with self.assertRaises(AssertionError):
            center_pad_to_size(array, (100,))
        # Test with negative desired shape
        with self.assertRaises(AssertionError):
            center_pad_to_size(array, (-100, 100))


if __name__ == "__main__":
    unittest.main()
