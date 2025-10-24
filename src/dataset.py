#!/usr/bin/env python
# coding=utf-8
"""
@author: xiaxu
@license: MIT
@file: dataset.py
@date: 2025/10/23 17:04
@desc: 
"""
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

from src import config


class InputMethodeDataset(Dataset):
    """
    自定义数据集
    """
    def __init__(self,path):
        """
        读取jsonl文件，可能是train_dataset.jsonl,test_dataset.jsonl
        """
        self.data = pd.read_json(path,lines=True,orient='records').to_dict(orient='records')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        读取index，转换为tensor类型
        :param index:
        :return:
        """
        input_tensor = torch.tensor(self.data[index]['input']) #
        target_tensor = torch.tensor(self.data[index]['target'])
        return input_tensor, target_tensor

def get_dataloader(train = True):
    path = config.PROCESSED_DATA_DIR / ('train_dataset.jsonl' if train else 'test_dataset.jsonl')
    dataset = InputMethodeDataset(path)
    return DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=train)

if __name__ == '__main__':
    train_loader = get_dataloader(train=True)
    test_loader = get_dataloader(train=False)
    print(len(train_loader))
    print(len(test_loader))

    for input_tensor, target_tensor in train_loader:
        print(input_tensor.shape)
        print(target_tensor.shape)
        break #因为train_loader为iter