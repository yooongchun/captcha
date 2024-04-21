#! -*- coding:utf-8 -*-
# /usr/bin/python3

"""
@Desc:
@Author: zhayongchun
@Date: 2023/12/21
"""
from pathlib import Path

import numpy as np
from PIL import Image
from paddle.vision.transforms import transforms


class DataUtil:
    def __init__(
        self, vocabulary_path: str, simple_mode: bool = False, max_len: int = 6
    ):
        self.simple_mode = simple_mode
        self.vocabulary_path = vocabulary_path

        self.max_len = max_len
        self.std = np.array([0.23375643, 0.23862716, 0.23951546])
        self.mean = np.array([0.55456273, 0.5225813, 0.51677391])

        self.channels = ["red", "blue", "black", "yellow", "text"]
        self.transform = transforms.Compose(
            [
                transforms.Normalize(mean=self.mean, std=self.std),
                # transforms.RandomPerspective(),
                # transforms.RandomAffine(degrees=15,
                #                         translate=(0.05, 0.05),
                #                         scale=(0.8, 1.1),
                #                         shear=15)
            ]
        )

    def _load_vocabulary(self):
        if self.simple_mode:
            return list("0123456789abcdefghijklmnopqrstuvwxyz".upper())
        with open(self.vocabulary_path, encoding="utf-8") as f:
            vocabulary = f.readlines()
        vocabulary = [w.strip() for w in vocabulary if w.strip()]
        return vocabulary

    def get_vocabulary(self):
        return self._load_vocabulary()

    def get_vocabulary_dict(self):
        return {t: i for i, t in enumerate(self._load_vocabulary())}

    def restore_img(self, img_arr: np.ndarray):
        """img_arr恢复为图片对象"""
        img = img_arr[:3, :, :].transpose([1, 2, 0])
        norm = (img * self.std + self.mean) * 255.0
        return norm.astype(np.uint8)

    def restore_label(self, label):
        """label恢复为text"""
        voca = self.get_vocabulary()
        return "".join(voca[i] for i in label if i != -1)

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
        voca_dict = self.get_vocabulary_dict()
        label_seq = [voca_dict[t] for t in label]
        if len(label_seq) < self.max_len:
            label_seq += [-1] * (self.max_len - len(label_seq))
        label_arr = np.array(label_seq).astype("int32")
        return label_arr


class ImageUtil:
    def __init__(self, path: Path):
        self.path = path
        self.red_channel = None
        self.blue_channel = None
        self.black_channel = None
        self.yellow_channel = None

        self.split_channel()

    def split_channel(self, upper: int = 220, lower: int = 60):
        img = Image.open(self.path)

        img_red = np.array(img.copy())
        img_red[
            ~(
                (img_red[:, :, 0] > upper)
                & (img_red[:, :, 1] < lower)
                & (img_red[:, :, 2] < lower)
            )
        ] = 255
        self.red_channel = Image.fromarray(img_red)

        img_blue = np.array(img.copy())
        img_blue[
            ~(
                (img_blue[:, :, 0] < lower)
                & (img_blue[:, :, 1] < lower)
                & (img_blue[:, :, 2] > upper)
            )
        ] = 255

        self.blue_channel = Image.fromarray(img_blue)

        img_black = np.array(img.copy())
        img_black[
            ~(
                (img_black[:, :, 0] < lower)
                & (img_black[:, :, 1] < lower)
                & (img_black[:, :, 2] < lower)
            )
        ] = 255
        self.black_channel = Image.fromarray(img_black)

        img_yellow = np.array(img.copy())
        img_yellow[
            ~(
                (img_yellow[:, :, 0] > upper)
                & (img_yellow[:, :, 1] > upper)
                & (img_yellow[:, :, 2] < lower)
            )
        ] = 255
        self.yellow_channel = Image.fromarray(img_yellow)

    def get_channel(self, channel: str):
        assert channel in [
            "red",
            "blue",
            "black",
            "yellow",
        ], f"channel only can be one of ['red', 'blue', 'black', 'yellow']"
        return getattr(self, f"{channel}_channel")
