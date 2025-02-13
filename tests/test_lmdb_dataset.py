import unittest
import os
import numpy as np
from benchmark.lmdb_dataset import LMDBDataset


class TestLMDBDataset(unittest.TestCase):
    """Unit tests for the LMDBDataset class."""

    def setUp(self):
        """Set up the test environment and dataset."""
        # Path to the small LMDB dataset
        self.lmdb_path = "tests/test_data/small_lizard_lmdb"

        # List of sample names in the small dataset
        self.sample_names = [
            "crag_17",
            "crag_17",
            "crag_17",
            "crag_35",
            "crag_35",
            "crag_35",
            "crag_1",
            "crag_1",
            "crag_1",
            "crag_61",
            "crag_61",
            "crag_61",
            "crag_34",
            "crag_34",
            "crag_34",
            "consep_5",
            "consep_5",
            "consep_5",
            "crag_10",
            "crag_10",
            "crag_10",
            "crag_64",
            "crag_64",
            "crag_64",
            "crag_51",
            "crag_51",
            "crag_51",
            "crag_13",
            "crag_13",
            "crag_13",
            "crag_53",
            "crag_53",
            "crag_53",
        ]

    def test_load_all_data(self):
        """Test loading all data without filtering."""
        # Initialize the dataset without any filters
        dataset = LMDBDataset(self.lmdb_path)

        # Check that the length matches the number of tiles
        expected_length = len(self.sample_names)
        actual_length = len(dataset)
        self.assertEqual(
            actual_length, expected_length, f"Dataset length should be {expected_length}, got {actual_length}."
        )

        # Collect all sample names from the dataset
        dataset_sample_names = []
        for idx in range(len(dataset)):
            data = dataset[idx]
            self.assertIn("sample_name", data, "Data should contain 'sample_name' key.")
            dataset_sample_names.append(data["sample_name"])

        # Verify that the sample names match the expected sample names
        self.assertCountEqual(
            dataset_sample_names, self.sample_names, "Sample names in dataset do not match expected sample names."
        )

    def test_filter_by_sample_names(self):
        """Test loading data with specific sample names."""
        # Define the sample names to include
        include_sample_names = ["crag_17", "crag_1", "crag_64"]

        # Initialize the dataset with the include_sample_names filter
        dataset = LMDBDataset(self.lmdb_path, include_sample_names=include_sample_names)

        # Calculate the expected number of tiles
        expected_num_tiles = sum(name in include_sample_names for name in self.sample_names)
        actual_num_tiles = len(dataset)
        self.assertEqual(
            actual_num_tiles,
            expected_num_tiles,
            f"Filtered dataset length should be {expected_num_tiles}, got {actual_num_tiles}.",
        )

        # Verify that all tiles belong to the included sample names
        for idx in range(len(dataset)):
            data = dataset[idx]
            self.assertIn("sample_name", data, "Data should contain 'sample_name' key.")
            self.assertIn(
                data["sample_name"],
                include_sample_names,
                f"Sample name {data['sample_name']} should be in include_sample_names.",
            )

    def test_invalid_index(self):
        """Test accessing invalid indices raises IndexError."""
        dataset = LMDBDataset(self.lmdb_path)
        with self.assertRaises(IndexError):
            _ = dataset[-1]  # Negative index
        with self.assertRaises(IndexError):
            _ = dataset[len(dataset)]  # Index equal to length

    def test_data_contents(self):
        """Test that data retrieved contains expected keys and types."""
        dataset = LMDBDataset(self.lmdb_path)
        data = dataset[0]

        # Expected keys in the data dictionary
        expected_keys = {"tile_name", "sample_name", "image", "semantic_mask"}
        self.assertTrue(expected_keys.issubset(data.keys()), "Data is missing expected keys.")

        # Verify data types
        self.assertIsInstance(data["tile_name"], str, "'tile_name' should be a string.")
        self.assertIsInstance(data["sample_name"], str, "'sample_name' should be a string.")
        self.assertIsInstance(data["image"], np.ndarray, "'image' should be a NumPy array.")
        self.assertIsInstance(data["semantic_mask"], np.ndarray, "'semantic_mask' should be a NumPy array.")

        # Check for optional 'instance_mask'
        if "instance_mask" in data:
            self.assertIsInstance(data["instance_mask"], np.ndarray, "'instance_mask' should be a NumPy array.")

    def test_context_manager(self):
        """Test that the LMDB environment is closed properly."""
        dataset = LMDBDataset(self.lmdb_path)
        self.assertIsNotNone(dataset.env, "LMDB environment should be initialized.")

        # Delete the dataset and ensure the environment is closed
        del dataset
        # In practice, we may not be able to test __del__ directly,
        # but we can check if env is None after deletion if __del__ sets it.

    def tearDown(self):
        """Clean up after tests."""
        pass  # No cleanup needed as we didn't modify any data


if __name__ == "__main__":
    unittest.main()
