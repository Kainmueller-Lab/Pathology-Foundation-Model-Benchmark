import lmdb
import os
import pickle
import tempfile
import numpy as np
from torch.utils.data import Dataset
from benchmark.lmdb_dataset import LMDBDataset
from benchmark.utils import get_weighted_sampler


def create_mock_lmdb(lmdb_path, num_samples=10, include_sample_names=None):
    """
    Creates a mock LMDB database for testing.

    Args:
        lmdb_path (str): Path to the LMDB file.
        num_samples (int): Number of samples to create.
        include_sample_names (list, optional): List of sample names to include. If None, all samples are created.
    """
    env = lmdb.open(lmdb_path, map_size=2**26)

    with env.begin(write=True) as txn:
        for i in range(num_samples):
            sample_name = f"sample_{i}"
            if include_sample_names and sample_name not in include_sample_names:
                continue

            # create random semantic mask with 70 % zeros and 30 % ones
            semantic_mask = np.hstack((np.zeros(int(np.floor(224*224*0.7))), np.ones(int(np.ceil(224*224*0.3))))).reshape(224, 224) 

            tile_dict = {
                "tile_name": f"{sample_name}_tile_{i}",
                "sample_name": sample_name,
                "data": np.random.rand(224, 224, 3).tolist(),
                "semantic_mask": semantic_mask
            }
            key = f"{i:08}".encode("ascii")
            txn.put(key, pickle.dumps(tile_dict))

    env.close()


def test_len():
    with tempfile.TemporaryDirectory() as tmp_dir:
        lmdb_path = f"{tmp_dir}/mock_lmdb"
        os.makedirs(os.path.dirname(lmdb_path), exist_ok=True)
        create_mock_lmdb(lmdb_path, num_samples=5)
        dataset = LMDBDataset(path=lmdb_path)
        assert len(dataset) == 5
        del dataset

def test_getitem():
    with tempfile.TemporaryDirectory() as tmp_dir:
        lmdb_path = f"{tmp_dir}/mock_lmdb"
        os.makedirs(os.path.dirname(lmdb_path), exist_ok=True)
        create_mock_lmdb(lmdb_path, num_samples=5)
        dataset = LMDBDataset(path=lmdb_path)
        sample = dataset[0]
        assert "tile_name" in sample
        assert "sample_name" in sample
        assert "data" in sample
        del dataset


def test_filtered_dataset():
    with tempfile.TemporaryDirectory() as tmp_dir:
        lmdb_path = f"{tmp_dir}/mock_lmdb"
        os.makedirs(os.path.dirname(lmdb_path), exist_ok=True)
        create_mock_lmdb(lmdb_path, num_samples=10, include_sample_names=["sample_1", "sample_3", "sample_5"])
        dataset = LMDBDataset(path=lmdb_path, include_sample_names=["sample_1", "sample_3"])
        assert len(dataset) == 2
        for sample in dataset:
            assert sample["sample_name"] in ["sample_1", "sample_3"]
        del dataset


def test_index_out_of_range():
    with tempfile.TemporaryDirectory() as tmp_dir:
        lmdb_path = f"{tmp_dir}/mock_lmdb"
        os.makedirs(os.path.dirname(lmdb_path), exist_ok=True)
        create_mock_lmdb(lmdb_path, num_samples=5)
        dataset = LMDBDataset(path=lmdb_path)
        try:
            dataset[5]
            assert False, "Expected IndexError but did not raise."
        except IndexError:
            pass
        del dataset


def test_no_include_sample_names():
    with tempfile.TemporaryDirectory() as tmp_dir:
        lmdb_path = f"{tmp_dir}/mock_lmdb"
        os.makedirs(os.path.dirname(lmdb_path), exist_ok=True)
        create_mock_lmdb(lmdb_path, num_samples=5)
        dataset = LMDBDataset(path=lmdb_path)
        assert len(dataset) == 5
        for sample in dataset:
            assert sample["sample_name"].startswith("sample_")
        del dataset

def test_sampler():
    with tempfile.TemporaryDirectory() as tmp_dir:
        lmdb_path = f"{tmp_dir}/mock_lmdb"
        os.makedirs(os.path.dirname(lmdb_path), exist_ok=True)
        create_mock_lmdb(lmdb_path, num_samples=5)
        dataset = LMDBDataset(path=lmdb_path)
        sampler = get_weighted_sampler(dataset, classes=[0, 1])

        assert len(sampler.weights) == 5
        assert sampler.weights.tolist() == [0.4, 0.4, 0.4, 0.4, 0.4]