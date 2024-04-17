#! -*- coding: utf-8 -*-

import re
import random
from collections import deque

import paddle
import Levenshtein as Lev
from paddle.metric import Metric

from src.helper import decoder


class WordsErrorRate(Metric):
    """单词级别错误率"""

    def __init__(self, vocabulary, name='words-error-rate'):
        super().__init__()
        self.vocabulary = vocabulary
        self.value = 0
        self.count = 0
        self.samples = deque(maxlen=10)
        self._name = name
        self.decoder = decoder.Decoder(vocabulary)

    def update(self, outputs, labels):
        """更新统计指标"""
        if not isinstance(outputs, paddle.Tensor):
            outputs = paddle.to_tensor(outputs)
        outputs = paddle.nn.functional.softmax(outputs, axis=-1)
        # 解码获取识别结果
        for output, label in zip(outputs, labels):
            pred_text = self.decoder.ctc_greedy_decoder(output)
            label_text = self.decoder.label_to_text(label)
            if random.random() < 0.01:
                # 按照1%概率打印最多10个样本
                self.samples.append({"pred": pred_text, "label": label_text})
            # 计算字错率
            self.value += self.seq_dis(pred_text, label_text) / float(len(label_text))
            self.count += 1

    def reset(self):
        """
        Resets all the metric state.
        """
        self.value = 0
        self.count = 0
        print("--" * 20)
        while len(self.samples) > 0:
            print(self.samples.pop())
        print("--" * 20)

    def accumulate(self):
        """
        Calculate the final precision.

        Returns:
            A scaler float: results of the calculated precision.
        """
        return self.value / max(1, self.count)

    def name(self):
        """
        Returns metric name
        """
        return self._name

    @staticmethod
    def seq_dis(pred, truth):
        s1, s2, = re.sub(r"\s+", "", pred), re.sub(r"\s+", "", truth)
        return Lev.distance(s1, s2)


class SampleAccuracy(Metric):
    """样本级别正确率"""

    def __init__(self, vocabulary, name='sample-acc'):
        super().__init__()
        self.vocabulary = vocabulary
        self.value = 0
        self.count = 0
        self._name = name
        self.decoder = decoder.Decoder(vocabulary)

    def update(self, outputs, labels):
        """更新统计指标"""
        if not isinstance(outputs, paddle.Tensor):
            outputs = paddle.to_tensor(outputs)
        outputs = paddle.nn.functional.softmax(outputs, axis=-1)
        # 解码获取识别结果
        for output, label in zip(outputs, labels):
            pred_text = self.decoder.ctc_greedy_decoder(output)
            label_text = self.decoder.label_to_text(label)
            # 计算样本正确率
            self.value += int(pred_text == label_text)
            self.count += 1

    def reset(self):
        """
        Resets all the metric state.
        """
        self.value = 0
        self.count = 0

    def accumulate(self):
        """
        Calculate the final precision.

        Returns:
            A scaler float: results of the calculated precision.
        """
        return self.value / max(1, self.count)

    def name(self):
        """
        Returns metric name
        """
        return self._name
