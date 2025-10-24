#!/usr/bin/env python
# coding=utf-8
"""
@author: xiaxu
@license: MIT
@file: model.py
@date: 2025/10/23 18:41
@desc: 
"""
from torch import nn

from src import config


class InputMethodRNN(nn.Module):
    def __init__(self,vocab_size):
        super().__init__()
        # 把每个vocab映射为一个稠密向量，dim为维度
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=config.EMBEDDING_DIM)
        self.rnn = nn.RNN(input_size=config.EMBEDDING_DIM,
                          hidden_size=config.HIDDEN_SIZE,
                          batch_first=True) #[batch_size,sql_len,embbedding_dim]
        # 把最后的隐藏状态映射到词表大小的维度
        self.out = nn.Linear(in_features=config.HIDDEN_SIZE,
                             out_features=vocab_size)

    def forward(self,x):
        # x.shape: [batch_size, seq_len]
        embed = self.embedding(x)
        # embed.shape:[batch_size, seq_len,embedding_dim]
        output, hidden = self.rnn(embed)
        # output.shape [batch_size, seq_len,hidden_size]
        last_hidden_state = output[:,-1,:] #取最后一个时间步的隐藏状态ht
        # last_hidden_state[ batch_size,hidden_size]
        output = self.out(last_hidden_state)
        return output  #[batch_size,vocab_size]