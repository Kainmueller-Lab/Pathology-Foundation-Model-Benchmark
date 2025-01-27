import lmdb
import pickle
from torch.utils.data import Dataset


class LMDBDataset(Dataset):
    def __init__(self, path, include_sample_names=None):
        """
        Initializes the LMDBDataset.

        Args:
            path (str): Path to the LMDB file.
            include_sample_names (list, optional): List of sample names to include. Default is None, meaning all samples are included.
        """
        self.lmdb_path = path
        self.include_sample_names = include_sample_names

        # Open the LMDB environment
        self.env = lmdb.open(self.lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)

        # Filter the keys based on sample names
        self._filter_lmdb_indices()

    def _filter_lmdb_indices(self):
        """
        Filters LMDB keys based on the include_sample_names list.
        """
        with self.env.begin(write=False) as txn:
            # Retrieve all keys and associated sample names
            include_keys = []

            cursor = txn.cursor()
            for key, value in cursor:
                data = pickle.loads(value)
                if self.include_sample_names is None:
                    include_keys.append(key)
                elif data["sample_name"] in self.include_sample_names:
                    include_keys.append(key)
        self.lmdb_key_list = include_keys
        
    def __len__(self):
        """
        Returns the number of filtered tiles in the dataset.
        """
        return len(self.lmdb_key_list)

    def __getitem__(self, idx):
        """
        Retrieves a tile from the LMDB file based on the given index.

        Args:
            idx (int): Index of the tile to retrieve.

        Returns:
            dict: The data associated with the tile.
        """
        if idx < 0 or idx >= len(self.lmdb_key_list):
            raise IndexError("Index out of range.")

        key = self.lmdb_key_list[idx]
        with self.env.begin(write=False) as txn:
            value = txn.get(key)
            if value is None:
                raise ValueError(f"No data found for key {key}")
            data = pickle.loads(value)
        if data['image'].max() > 1: # Normalize image if it is not already normalized
            data['image'] = data['image'] / data['image'].max()
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
