import os
import pickle
import shutil
import tempfile
import unittest

import lmdb
import numpy as np
import scipy.io as sio

# Import the necessary modules from your project
# Adjust the import paths as necessary
from bio_image_datasets.lizard_dataset import LizardDataset
from bio_image_datasets.segpath_dataset import SegPath
from PIL import Image
from skimage import io

from benchmark.dataset.split_to_tiles import transform_to_tiles
from benchmark.dataset.tiles_to_lmdb import create_lmdb_database


class TestCreateLMDBDatabase(unittest.TestCase):
    """Unit tests for the create_lmdb_database function."""

    def setUp(self):
        """Set up temporary directories and mock datasets."""
        self.temp_dir = tempfile.mkdtemp()
        self.lmdb_dir = tempfile.mkdtemp()
        self.tile_size = 64  # Use a small tile size for testing

    def tearDown(self):
        """Clean up temporary directories."""
        shutil.rmtree(self.temp_dir)
        shutil.rmtree(self.lmdb_dir)

    def create_mock_lizard_dataset(self):
        """Create a mock LizardDataset with minimal data."""
        # Create mock data directories
        image_dirs = [
            os.path.join(self.temp_dir, "lizard_images1", "Lizard_Images1"),
            os.path.join(self.temp_dir, "lizard_images2", "Lizard_Images2"),
        ]
        label_dir = os.path.join(self.temp_dir, "lizard_labels", "Lizard_Labels", "Labels")

        # Create directories
        for dir_path in image_dirs + [label_dir]:
            os.makedirs(dir_path, exist_ok=True)

        # Create a mock image file in each image directory
        image_filename = "sample1.png"
        for image_dir in image_dirs:
            image_path = os.path.join(image_dir, image_filename)
            mock_image = Image.new("RGB", (128, 128), color=(255, 0, 0))
            mock_image.save(image_path)

        # Create corresponding mock .mat label files for each image
        label_name = image_filename.replace(".png", ".mat")
        label_path = os.path.join(label_dir, label_name)
        mock_label_data = {
            "inst_map": np.ones((128, 128), dtype=np.int32),
            "class": np.array([1]),
            "id": np.array([1]),
            "bbox": np.array([[0, 127, 0, 127]]),
            "centroid": np.array([[64, 64]]),
        }
        sio.savemat(label_path, mock_label_data)

        # Instantiate the dataset
        dataset = LizardDataset(local_path=self.temp_dir)
        return dataset

    def create_mock_segpath_dataset(self):
        """Create a mock SegPath dataset with minimal data."""
        # Create mock data directories
        folder_names = ["panCK_Epithelium", "CD3CD20_Lymphocyte"]
        file_paths = []
        images = []
        masks = []
        for folder in folder_names:
            folder_path = os.path.join(self.temp_dir, folder)
            os.makedirs(folder_path, exist_ok=True)
            # Create a mock image file
            image_filename = f"{folder}_0_HE.png"
            image_path = os.path.join(folder_path, image_filename)
            mock_image = np.random.randint(0, 256, size=(128, 128, 3), dtype=np.uint8)
            io.imsave(image_path, mock_image)
            images.append(mock_image)

            # Create a corresponding mock mask file
            mask_filename = image_filename.replace("_HE.png", "_mask.png")
            mask_path = os.path.join(folder_path, mask_filename)
            mock_mask = np.random.randint(0, 2, size=(128, 128), dtype=np.uint8)
            io.imsave(mask_path, mock_mask)
            masks.append(mock_mask)

            file_paths.append(os.path.join(folder_path, image_filename.replace("_HE.png", "")))

        # Instantiate the dataset
        dataset = SegPath(local_path=self.temp_dir)
        return dataset

    def test_create_lmdb_lizard_dataset(self):
        """Test create_lmdb_database with the mock LizardDataset."""
        dataset = self.create_mock_lizard_dataset()
        lmdb_path = os.path.join(self.lmdb_dir, "lizard_lmdb")

        # Call the function with the mock dataset
        create_lmdb_database(dataset, lmdb_path=lmdb_path, tile_size=self.tile_size, map_size=2**26)

        # Verify that the LMDB database has been created
        self.assertTrue(os.path.exists(lmdb_path), "LMDB database was not created.")

        # Open the LMDB database and read the contents
        env = lmdb.open(lmdb_path, readonly=True, lock=False)
        with env.begin() as txn:
            # Get the number of entries
            cursor = txn.cursor()
            entries = list(cursor)
            self.assertTrue(len(entries) > 0, "LMDB database is empty.")

            # For simplicity, check the first entry
            key, value = entries[0]
            tile_data = pickle.loads(value)
            # Verify that the tile data contains expected keys
            expected_keys = {
                "tile_name",
                "sample_name",
                "image",
                "semantic_mask",
                "instance_mask",
            }
            self.assertTrue(
                expected_keys.issubset(tile_data.keys()),
                "Tile data is missing expected keys.",
            )
            # Verify that the image and masks are numpy arrays
            self.assertIsInstance(tile_data["image"], np.ndarray, "Image should be a NumPy array.")
            self.assertIsInstance(
                tile_data["semantic_mask"],
                np.ndarray,
                "Semantic mask should be a NumPy array.",
            )
            self.assertIsInstance(
                tile_data["instance_mask"],
                np.ndarray,
                "Instance mask should be a NumPy array.",
            )
        env.close()

    def test_create_lmdb_segpath_dataset(self):
        """Test create_lmdb_database with the mock SegPath dataset."""
        dataset = self.create_mock_segpath_dataset()
        lmdb_path = os.path.join(self.lmdb_dir, "segpath_lmdb")

        # Call the function with the mock dataset
        create_lmdb_database(dataset, lmdb_path=lmdb_path, tile_size=self.tile_size, map_size=2**26)

        # Verify that the LMDB database has been created
        self.assertTrue(os.path.exists(lmdb_path), "LMDB database was not created.")

        # Open the LMDB database and read the contents
        env = lmdb.open(lmdb_path, readonly=True, lock=False)
        with env.begin() as txn:
            # Get the number of entries
            cursor = txn.cursor()
            entries = list(cursor)
            self.assertTrue(len(entries) > 0, "LMDB database is empty.")

            # For simplicity, check the first entry
            key, value = entries[0]
            tile_data = pickle.loads(value)
            # Verify that the tile data contains expected keys
            expected_keys = {"tile_name", "sample_name", "image", "semantic_mask"}
            self.assertTrue(
                expected_keys.issubset(tile_data.keys()),
                "Tile data is missing expected keys.",
            )
            # Verify that the image and mask are numpy arrays
            self.assertIsInstance(tile_data["image"], np.ndarray, "Image should be a NumPy array.")
            self.assertIsInstance(
                tile_data["semantic_mask"],
                np.ndarray,
                "Semantic mask should be a NumPy array.",
            )
            # Since SegPath doesn't have instance masks, ensure 'instance_mask' is not in tile_data
            self.assertNotIn(
                "instance_mask",
                tile_data,
                "Instance mask should not be present in SegPath tiles.",
            )
        env.close()


if __name__ == "__main__":
    unittest.main()
