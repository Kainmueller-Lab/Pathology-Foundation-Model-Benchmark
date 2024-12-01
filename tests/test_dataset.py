from benchmark.benchmark import Dataset


def test_dataset():
    class ChildDataset(Dataset):
        def download(self):
            pass

        def __len__(self):
            return 0

        def __getitem__(self, idx):
            return None

        def load_sample(self, idx):
            return None

        def load_mask(self, idx):
            return None
    
    child_dataset = ChildDataset(local_path='path/to/dataset')
    assert isinstance(child_dataset, Dataset)
