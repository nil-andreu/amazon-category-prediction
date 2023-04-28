import numpy as np
import torch
from torch.utils.data import DataLoader

from shared.model.data.dataset import ProductsDataset


def get_data_loaders(
    product_dataset: ProductsDataset,
    batch_size: int = 128,
    valid_size: float = 0.15,
    shuffle=True,
    num_workers=4,
):
    range_idx = list(range(len(product_dataset)))
    split = int(len(range_idx) * (1 - valid_size))
    np.random.shuffle(range_idx)

    train = torch.utils.data.Subset(product_dataset, range_idx[:split])
    valid = torch.utils.data.Subset(product_dataset, range_idx[split:])

    return DataLoader(
        train, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    ), DataLoader(
        valid, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
