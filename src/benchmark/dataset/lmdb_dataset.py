import os
import pickle

import lmdb
from torch.utils.data import Dataset

# path -> (pid, env)
_ENV_CACHE = {}


def _get_shared_env(path):
    """One LMDB env per (process, path); safe across fork."""
    path = os.path.abspath(path)
    pid = os.getpid()
    entry = _ENV_CACHE.get(path)

    if entry is not None:
        owner_pid, env = entry
        if owner_pid == pid:
            return env
        # Inherited from the parent via fork: drop our copy of it so
        # LMDB's per-process registry lets us open a fresh one.
        # (Only closes this process's copy; the parent is unaffected.)
        try:
            env.close()
        except Exception:
            pass
        del _ENV_CACHE[path]

    env = lmdb.open(
        path,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )
    _ENV_CACHE[path] = (pid, env)
    return env


class LMDBDataset(Dataset):
    def __init__(self, path, include_sample_names=None, include_tile_names=None):
        self.lmdb_path = path
        self.include_sample_names = include_sample_names
        self.include_tile_names = include_tile_names
        self._filter_lmdb_indices()

    def _get_env(self):
        return _get_shared_env(self.lmdb_path)

    def _filter_lmdb_indices(self):
        env = self._get_env()
        include_keys = []

        with env.begin(write=False) as txn:
            for key, value in txn.cursor():
                data = pickle.loads(value)

                if self.include_tile_names is not None:
                    if data["tile_name"] in self.include_tile_names:
                        include_keys.append(key)
                elif self.include_sample_names is not None:
                    name = data["sample_name"]
                    if name in self.include_sample_names or name + ".hdf" in self.include_sample_names:
                        include_keys.append(key)
                else:
                    include_keys.append(key)

        self.lmdb_key_list = include_keys

    def __len__(self):
        return len(self.lmdb_key_list)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.lmdb_key_list):
            raise IndexError("Index out of range.")

        env = self._get_env()
        key = self.lmdb_key_list[idx]

        with env.begin(write=False) as txn:
            value = txn.get(key)
            if value is None:
                raise ValueError(f"No data found for key {key}")
            data = pickle.loads(value)

        if "image" in data and data["image"] is not None:
            if data["image"].max() > 1:
                data["image"] = data["image"] / data["image"].max()

        return data


if __name__ == "__main__":
    # Example usage of the LMDBDataset
    lmdb_path = "tests/test_data/small_lizard_lmdb"
    dataset = LMDBDataset(lmdb_path)
    print(f"Number of tiles in the dataset: {len(dataset)}")

    # Retrieve a tile from the dataset
    idx = 0
    tile_data = dataset[idx]
    print(f"Tile data: {tile_data}")