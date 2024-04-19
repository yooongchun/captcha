#! -*- coding: utf-8 -*-

import json
import pathlib
import random
from typing import Optional, List

from PIL import Image
from paddle.io import Dataset
from loguru import logger

from src import vocabulary_path
from src.helper.util import DataUtil
from src.helper.generate_captcha import CaptchaGenerator


def _load_meta_info(dataset_dir: str, mode: str):
    json_file = dataset_dir + f"/{mode}.json"
    if not pathlib.Path(json_file).exists():
        logger.warning(f"file {json_file} not exists!")
        return []
    with open(json_file, "r", encoding="utf-8") as fin:
        meta_info = json.load(fin)
        for meta in meta_info:
            meta["path"] = str(pathlib.Path(dataset_dir, meta["path"]).absolute())
        logger.info(f"load meta info from {json_file}, total {len(meta_info)}")
        return meta_info


class CaptchaDataset(Dataset):
    """数据集加载器
    auto_gen: 是否使用自动生成，当此参数启用时，数据即时生成，否则从本地路径中加载数据
    auto_num: 当使用auto_gen时数据可以是无限多，但为了训练适配训练流程，需指定数据集大小
    dataset_dir: 当从本地路径加载数据集时的路径地址
    channel: 生成的标签需要的颜色类型，可选两种模式，第一种是指定具体颜色类型，第二种是随机选择一个颜色类型，这种模式下模型能识别所有颜色类型
    """

    def __init__(
            self,
            auto_gen: bool = False,
            auto_num: int = 100_000,
            dataset_dir: List[str] = None,
            mode: str = "train",
            channel: str = "text",
            max_len: int = 6,
            simple_mode: bool = False
    ):
        super(CaptchaDataset, self).__init__()

        self.auto_gen = auto_gen
        self.auto_num = auto_num
        self.max_len = max_len
        self.channel = channel
        self.dataset_dir = dataset_dir
        self.generator = CaptchaGenerator(vocabulary_path=vocabulary_path, max_words=max_len, simple_mode=simple_mode)
        self.vocabulary = self.generator.characters if simple_mode else self.generator.vocabulary
        self.num_classes = len(self.vocabulary)

        self.data_util = DataUtil(vocabulary=self.vocabulary, max_len=max_len)
        assert channel in self.data_util.channels, f"channel only can be one of {self.data_util.channels}"
        assert auto_gen or dataset_dir, "dataset_dir must be set when auto_gen is False"
        assert mode in ["train", "test"], "mode can only be train or test!"

        if not auto_gen and dataset_dir:
            if isinstance(dataset_dir, str):
                self.meta_info = _load_meta_info(dataset_dir, mode)
            elif isinstance(dataset_dir, list):
                self.meta_info = []
                for path in dataset_dir:
                    self.meta_info.extend(_load_meta_info(path, mode))
            else:
                raise ValueError("dataset_dir must be str or list[str]")
            # 过滤掉颜色不存在的文件
            if channel not in ["text", "random"]:
                self.meta_info = [meta for meta in self.meta_info if meta.get(channel)]
            logger.info(f"load {len(self.meta_info)} samples from {dataset_dir}")

    def _data_from(self, idx):
        """自动生成或者从本地数据读取"""
        if self.auto_gen:  # 自动生成
            img, label_map = self.generator.gen_one(min_num=4, max_num=self.max_len)
            if self.channel not in ["text", "random"]:
                while self.channel not in label_map:  # 生成的图片无对应颜色数据则重新生成
                    img, label_map = self.generator.gen_one(min_num=4, max_num=self.max_len)
        elif self.dataset_dir:  # 从本地数据读取
            label_map: dict = self.meta_info[idx]
            img = Image.open(label_map["path"])
        else:
            raise ValueError("dataset_dir must be set when auto_gen is False")

        # 如果是random则随机生成一个
        keys = [key for key in label_map if key in self.data_util.channels]
        channel = random.choice(keys) if self.channel == "random" else self.channel
        return img, channel, label_map

    def __getitem__(self, idx):
        # 读取数据，来自本地或者自动生成
        img, channel, label_map = self._data_from(idx)

        # 图片加载&转换
        img_arr = self.data_util.process_img(img)
        # 颜色信息处理
        color_index = self.data_util.process_channel(channel)
        # 标签处理
        label_arr = self.data_util.process_label(label_map[channel])

        return (img_arr, color_index), label_arr

    def __len__(self):
        return len(self.meta_info) if not self.auto_gen else self.auto_num
