#!/usr/bin/env python
# coding=utf-8
"""
@author: xiaxu
@license: MIT
@file: predict.py
@date: 2025/10/24 09:52
@desc: 
"""
import torch

from src import config
from src.model import InputMethodRNN
from src.tokenizer import JiebaTokenizer


def predict_batch(model, inputs):
    model.eval()
    with torch.no_grad():
        outputs = model(inputs) #inputs:[batch_size,seq_len] output:[batch_size,vocab_size] logits
    top5_indexes = torch.topk(outputs, 5).indices  #返回索引，[batch_size,5]

    top5_indexes = top5_indexes.tolist() #转换为python list
    return top5_indexes

#预测流程
def predict(text,model,tokenizer, device):
    # 处理输入
    indexes = tokenizer.encode(text) # 编码 word2index
    input_tensor = torch.tensor([indexes], dtype=torch.long, device=device) # tensor，因为model接收[batch_size,seq_len],因此[indexes]

    # 预测
    top5_indexes = predict_batch(model, input_tensor) #[[top5]]
    top5_tokens = [tokenizer.index_to_word[index] for index in top5_indexes[0]] #返回top5的token; index2word
    return top5_tokens

def run_pridict():
    # 准备资源
    # 1. 确定设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2.词表
    tokenizer = JiebaTokenizer.from_vocab(config.MODELS_DIR / 'vocab_list.txt')
    print("词表加载成功")

    # 3. 模型
    model = InputMethodRNN(vocab_size=tokenizer.vocab_size).to(device)
    model.load_state_dict(torch.load(config.MODELS_DIR / 'best.pth'))
    print("模型加载成功")

    print("欢迎使用输入法模型(输入q或者quit退出)")
    input_history = ''
    while True:
        user_input = input("> ")
        if user_input in ['q', 'quit']:
            print("欢迎下次再来")
            break
        if user_input.strip() == '':
            print("请输入内容")
            continue
        input_history += user_input
        print(f'输入历史:{input_history}')
        top5_tokens = predict(input_history, model, tokenizer, device)
        print(f'预测结果:{top5_tokens}')

if __name__ == '__main__':
    run_pridict()