#! -*- coding: utf-8 -*-

import json
from pathlib import Path
from typing import List

from PIL import Image
from paddle.io import Dataset
from loguru import logger

from src.helper.util import DataUtil


class CaptchaDataset(Dataset):
    """数据集加载器"""

    def __init__(
        self,
        dataset_dirs: List[str] | str,
        vocabulary_path: str,
        mode: str,
        channel: str,
        data_type: str,
        max_len: int = 6,
        simple_mode: bool = False,
    ):
        super(CaptchaDataset, self).__init__()

        self.mode = mode
        self.max_len = max_len
        self.channel = channel
        self.simple_mode = simple_mode
        self.data_type = data_type
        self.dataset_dirs = dataset_dirs

        self.data_util = DataUtil(vocabulary_path=vocabulary_path, simple_mode=simple_mode, max_len=max_len)
        self.num_classes = len(self.data_util.get_vocabulary())

        assert channel in self.data_util.channels, f"channel only can be one of {self.data_util.channels}"
        assert mode in ["train", "test"], "mode can only be train or test!"
        assert data_type in [
            "single",
            "color",
        ], "data_type can only be single or color!"

        self.meta_info = []
        if isinstance(dataset_dirs, str):
            dataset_dirs = [dataset_dirs]
        for path in dataset_dirs:
            self.meta_info.extend(self._load_meta_info(path))
        logger.info(f"load {len(self.meta_info)} samples from {dataset_dirs}")

    def _load_sample(self, idx):
        """自动生成或者从本地数据读取"""
        label_map: dict = self.meta_info[idx]
        img = Image.open(label_map["path"])
        return img, label_map

    def _load_meta_info(self, dataset_dir: str):
        json_file = Path(dataset_dir, f"{self.mode}.json")
        if not json_file.exists():
            logger.warning(f"file {json_file} not exists!")
            return []
        with open(json_file, "r", encoding="utf-8") as fin:
            _meta_info = json.load(fin)
            meta_info = []
            for meta in _meta_info:
                if (self.simple_mode and not meta["simple"]) or meta["type"] != self.data_type:
                    continue
                meta["path"] = str(Path(dataset_dir, meta["path"]).absolute())
                if not Path(meta["path"]).exists():
                    logger.warning(f"file {meta['path']} not exists!")
                    continue
                meta_info.append(meta)
            logger.info(f"load meta info from {json_file}, num {len(meta_info)}")
            return meta_info

    def __getitem__(self, idx):
        # 读取数据，来自本地或者自动生成
        img, label_map = self._load_sample(idx)

        # 图片加载&转换
        img_arr = self.data_util.process_img(img)
        # 颜色信息处理
        color_index = self.data_util.process_channel(self.channel)
        # 标签处理
        label_arr = self.data_util.process_label(label_map["text"])

        return (img_arr, color_index), label_arr

    def __len__(self):
        return len(self.meta_info)
