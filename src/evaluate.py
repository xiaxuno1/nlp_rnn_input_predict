#!/usr/bin/env python
# coding=utf-8
"""
@author: xiaxu
@license: MIT
@file: evaluate.py
@date: 2025/10/23 19:37
@desc:模型的评估，使用test_dataset评估
"""
import torch

from src import config
from src.dataset import get_dataloader
from src.model import InputMethodRNN
from src.predict import predict_batch
from src.tokenizer import JiebaTokenizer


def evaluate(model, test_dataloader,device):
    """
    在测试集上计算top1 和top5准确率
    :param model:
    :param test_dataset:
    :param device:
    :return:
    """
    model.eval()
    top1_acc_count = 0
    top5_acc_count = 0
    total = 0
    for data, targets in test_dataloader:
        inputs = data.to(device) #[batch_size,seq_len]
        targets = targets.tolist() #[batch_size,]->[batch_size]
        top5_indexes_list = predict_batch(model, inputs) #list:[bathc_size,5]
        for target,top5_indexes in zip(targets,top5_indexes_list): #zip实现了打包成对迭代[(target[0],top5_indexes_list[0]),()]
            total +=1
            if target == top5_indexes[0]: # 判断预测的第一个是否为target，即top1的准确率
                top1_acc_count+=1
            if target in top5_indexes: #top5的准确率
                top5_acc_count+=1
    return top1_acc_count/total, top5_acc_count/total

def run_evaluation():
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 词表
    tokenizer = JiebaTokenizer.from_vocab(config.MODELS_DIR / 'vocab_list.txt')

    # model
    model = InputMethodRNN(vocab_size=tokenizer.vocab_size).to(device)
    model.load_state_dict(torch.load(config.MODELS_DIR / 'best.pth')) #加载训练完成模型参数
    print("模型加载完成，开始评估")
    # 数据集
    test_dataloader = get_dataloader(train=False)

    #评估
    top1_acc,top5_acc = evaluate(model, test_dataloader, device)
    print('Top-1 Accuracy:{:.2f}'.format(top1_acc))
    print('Top-5 Accuracy:{:.2f} '.format(top5_acc) )

if __name__ == '__main__':
    run_evaluation()