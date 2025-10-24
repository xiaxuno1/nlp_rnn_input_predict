#!/usr/bin/env python
# coding=utf-8
"""
@author: xiaxu
@license: MIT
@file: train.py
@date: 2025/10/23 17:52
@desc: 
"""
import time

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter, writer
from tqdm import tqdm

from src import config
from src.dataset import get_dataloader
from src.model import InputMethodRNN
from src.tokenizer import JiebaTokenizer


def train_one_epoch(model, dataloader, optimizer, loss_fn,device):
    model.train()
    train_loss = 0
    for inputs, target in tqdm(dataloader,desc='train'):
        inputs = inputs.to(device)
        target = target.to(device)

        # 前向传播
        outputs = model(inputs)
        loss = loss_fn(outputs, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss / len(dataloader)

def train():
    # 确定设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #数据集
    dataloader = get_dataloader()

    # vocab
    tokenize = JiebaTokenizer.from_vocab(config.MODELS_DIR / 'vocab_list.txt')

    #模型
    model = InputMethodRNN(vocab_size=tokenize.vocab_size).to(device)

    # loss
    loss_function = nn.CrossEntropyLoss().to(device)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # tensorboard
    writer = SummaryWriter(log_dir=config.LOGS_DIR / time.strftime("%Y-%m-%d_%H-%M-%S"))

    #开始训练
    best_loss = float('inf') #初始化为无穷大
    for epoch in range(config.EPOCHS):
        print('Epoch {}/{}'.format(epoch+1, config.EPOCHS))
        # one epoch
        loss = train_one_epoch(model, dataloader, optimizer, loss_function, device)
        print('\tLoss: {:.4f}'.format(loss))

        #记录训练结果
        writer.add_scalar('Loss/train', loss, epoch+1)

        #保存模型
        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), config.MODELS_DIR / 'best.pth')
            print('Best model saved')
    writer.close()

if __name__ == '__main__':
    train()