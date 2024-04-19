#! -*- coding:utf-8 -*-
# /usr/bin/python3

"""
@Desc:
@Author: zhayongchun
@Date: 2023/12/21
"""
from typing import Optional
import numpy as np
from PIL import Image
from paddle.vision.transforms import transforms

from src import vocabulary_path


class DataUtil:
    def __init__(self, vocabulary: Optional[list] = None, max_len: int = 6):
        self.vocabulary = vocabulary if vocabulary else self._load_vocabulary()
        self._vocabulary_dict = {t: i for i, t in enumerate(self.vocabulary)}

        self.max_len = max_len
        self.std = np.array([0.23375643, 0.23862716, 0.23951546])
        self.mean = np.array([0.55456273, 0.5225813, 0.51677391])

        self.channels = ["red", "blue", "black", "yellow", "text", "random"]
        self.transform = transforms.Compose([transforms.Normalize(mean=self.mean, std=self.std),
                                             # transforms.RandomPerspective(),
                                             # transforms.RandomAffine(degrees=15,
                                             #                         translate=(0.05, 0.05),
                                             #                         scale=(0.8, 1.1),
                                             #                         shear=15)
                                             ])

    @staticmethod
    def _load_vocabulary():
        with open(vocabulary_path, encoding="utf-8") as f:
            vocabulary = f.readlines()
        vocabulary = [w.strip() for w in vocabulary if w.strip()]
        return vocabulary

    def restore_img(self, img_arr: np.ndarray):
        """img_arr恢复为图片对象"""
        img = img_arr[:3, :, :].transpose([1, 2, 0])
        norm = (img * self.std + self.mean) * 255.0
        return norm.astype(np.uint8)

    def restore_label(self, label):
        """label恢复为text"""
        return "".join(self.vocabulary[i] for i in label if i != -1)

    def process_img(self, img: Image):
        # 图片加载&转换
        img = img.convert("RGB")
        img_arr = np.array(img, np.float32).transpose([2, 0, 1]) / 255.0
        img_arr = self.transform(img_arr)
        return img_arr

    def process_channel(self, channel: str):
        # 颜色信息处理
        assert channel in self.channels, f"channel only can be one of {self.channels}"
        color_index = 1.0 * self.channels.index(channel) / len(self.channels)
        return color_index

    def process_label(self, label: str):
        # 标签处理
        label_seq = [self._vocabulary_dict[t] for t in label]
        if len(label_seq) < self.max_len:
            label_seq += [-1] * (self.max_len - len(label_seq))
        label_arr = np.array(label_seq).astype("int32")
        return label_arr
