#!/usr/bin/env python
# coding=utf-8
"""
@author: xiaxu
@license: MIT
@file: process.py
@date: 2025/10/23 15:20
@desc: 处理原始的jsonl数据
"""
from tqdm import tqdm
import pandas as pd
from src import config
from sklearn.model_selection import train_test_split

from src.tokenizer import JiebaTokenizer


def build_dataset(sentences,tokenizer):
    """
    :param sentences:
    :param tokenizer:
    :return:[{'input':[1,2,3,4,5],'target':5},{'input':[2,3,4,5,6],'target':7}]
    """
    index_sentences = [tokenizer.encode(sentence) for sentence in sentences] #遍历返回每个句子的index:[[12,36,0],]
    dataset = []
    for sentence in tqdm(index_sentences,desc="generate dataset"):
        for i in range(len(sentence)-config.SEQ_LEN):
            input = sentence[i:i+config.SEQ_LEN] # {'input':[1,2,3,4,5],'target':6}
            target = sentence[i+config.SEQ_LEN]
            dataset.append({'input':input,'target':target})
    return dataset


def process():
    """

    :return:
    """
    # 读取文件，随机抽取其中1%的样本；文件是一个json line格式每一行都是一个独立的json对象，lines和orient作用类似；最终生成的dataframe格式
    df = pd.read_json(config.RAW_DATA_DIR / 'synthesized_.jsonl',lines = True,orient = "record").sample(frac=0.01)
    #提取句子
    sentences = []
    for dialog in df['dialog']: #提取全体的dialog，遍历
        # print(dialog) #dialog为一个list
        for sentence in dialog: #遍历每个dialog list :['user1':xxx,'user2':xxx]
            # print(sentence)
            sentences.append(sentence.split('：')[1]) #用中文的：
    print(sentences[0])
    print("sentences:",len(sentences))

    #划分数据集和测试集
    train_sentences, test_sentences = train_test_split(sentences, test_size = 0.2) #测试占0.2

    # tokenizer
    JiebaTokenizer.build_vocab(train_sentences,config.MODELS_DIR / "vocab_list.txt") #构建 vocab_list
    tokenize = JiebaTokenizer.from_vocab(config.MODELS_DIR / "vocab_list.txt") #读取词表，返回的是cls，因此可以直接调用encode
    # build_dataset
    train_dataset = build_dataset(train_sentences, tokenize)
    test_dataset = build_dataset(test_sentences, tokenize)

    #保存
    pd.DataFrame(train_dataset).to_json(config.PROCESSED_DATA_DIR / 'train_dataset.jsonl',lines=True,orient="records")
    pd.DataFrame(test_dataset).to_json(config.PROCESSED_DATA_DIR / "test_dataset.jsonl",lines=True,orient="records")

if __name__ == '__main__':
    process()