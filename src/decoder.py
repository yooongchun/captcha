#! -*- coding: utf-8 -*-

import numpy as np
from itertools import groupby


class Decoder:
    """解码器"""

    def __init__(self, vocabulary: list):
        self.vocabulary = vocabulary
        self.blank_index = len(self.vocabulary)

    def ctc_greedy_decoder(self, probs_seq: np.ndarray):
        """CTC贪婪（最佳路径）解码器。
        由最可能的令牌组成的路径被进一步后处理
        删除连续的重复和所有的空白。
        """
        # 尺寸验证
        for probs in probs_seq:
            if not len(probs) == len(self.vocabulary) + 1:
                raise ValueError("probs_seq 尺寸与词汇不匹配")
        # argmax以获得每个时间步长的最佳指标
        max_index_list = np.argmax(probs_seq, -1)
        # 删除连续的重复索引
        index_list = [index_group[0] for index_group in groupby(max_index_list)]
        # 删除空白索引
        index_list = [index for index in index_list if index != self.blank_index]
        # 将索引列表转换为字符串
        return ''.join([self.vocabulary[index] for index in index_list])

    def label_to_text(self, label):
        """标签转文字"""
        return "".join([self.vocabulary[index] for index in label if index != -1])
