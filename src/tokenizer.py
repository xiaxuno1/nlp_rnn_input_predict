#!/usr/bin/env python
# coding=utf-8
"""
@author: xiaxu
@license: MIT
@file: tokenizer.py
@date: 2025/10/23 16:04
@desc: 
"""
import jieba
from tqdm import tqdm
import config


class JiebaTokenizer:
    unk_token = '<unk>'

    def __init__(self,vocab_list):
        self.vocab_list = vocab_list
        self.vocab_size = len(self.vocab_list)
        self.word_to_index = {word:index for index,word in enumerate(self.vocab_list)}
        self.index_to_word = {index:word for index,word in enumerate(self.vocab_list)}
        self.unk_index = self.word_to_index[self.unk_token]

    @staticmethod
    def tokenize(sentence):
        return jieba.lcut(sentence)

    def encode(self,sentence):
        tokens = self.tokenize(sentence)
        return [self.word_to_index.get(token,self.unk_index) for token in tokens] #返回句子分词后的索引列表

    @classmethod
    def build_vocab(cls,sentences,vocab_path):
        """
        构建词表，根据输入的sentences,分词，统计词语构建词表
        :param vocab_path: 存储路径
        :param sentences:
        :return: [a,b,c]
        """
        vocab_set = set()
        for sentence in tqdm(sentences, desc="build vocab"):
            vocab_set.update(jieba.lcut(sentence))
        vocab_list = [cls.unk_token]+list(vocab_set) # 添加unk
        print("vocab_list:",len(vocab_list))

        # 保存词表
        with open(vocab_path,"w",encoding="utf-8") as f:
            f.write("\n".join(vocab_list)) #list内的每个vocab保存为一行

    @classmethod
    def from_vocab(cls,vocab_path):
        with open(vocab_path,"r",encoding="utf-8") as f:
            vocab_list = [line.strip() for line in f.readlines()]
        return cls(vocab_list)

if __name__ == '__main__':
    tokenizer = JiebaTokenizer.from_vocab(config.MODELS_DIR / 'vocab.txt')
    print(tokenizer.encode("你好，小明，今天天气真好"))

