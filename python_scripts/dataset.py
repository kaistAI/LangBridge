from torch.utils.data import Dataset
from datasets import load_dataset


class Data(Dataset):
    def __init__(self, dataset_path, split):
        super().__init__()
        if 'json' in dataset_path:
            self.ds = load_dataset(
                'json',
                data_files=dataset_path)['train']
        else:
            self.ds = load_dataset(dataset_path)[split]

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        x = self.ds[idx]
        return {
            'input': x['input'],
            'output': x['output']
        }
