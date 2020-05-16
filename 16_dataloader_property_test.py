from random import sample
import json
import time

import torch
from torch.utils.data import Dataset, DataLoader

class PadCollate:
    """
    a variant of callate_fn that pads according to the longest sequence in
    a batch of sequences
    """

    def __init__(self):
        """
        args:
            dim - the dimension to be padded (dimension of time in sequences)
        """

    def pad_collate(self, batch):
        batch["A"][0][0]  = 100
        batch["B"] = 100

    def __call__(self, batch):
        return self.pad_collate(batch)

class TestDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.all_instances = []

        for i in range(100):
            self.all_instances.append({"A":[1,2,3,4], "B":[5,6,7,8]})

    def __len__(self):
        return len(self.all_instances)

    def __getitem__(self, idx):
        return self.all_instances[idx]


test_dataset = TestDataset()
test_dataloader = DataLoader(test_dataset, batch_size=1,
                                     shuffle=False, num_workers=2, collate_fn=PadCollate())


for i, batch in enumerate(test_dataloader):
    print("="*20)
    print(test_dataset[i])
    print(batch)

    input("AA")
