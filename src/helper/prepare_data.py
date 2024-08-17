"""
准备数据集标签
@author:yooongchun@foxmail.com
@date:2024-08-17
"""
import re
import json
from random import random
from pathlib import Path
from collections import defaultdict
from loguru import logger

dataset_path = Path(__file__).parent.parent.parent / "dataset/labeled"
assert dataset_path.is_dir(), f"dataset path not exists: {dataset_path}"


def generate_label(test_ratio=0.1):
    """
    从图片中解析获取标签
    新的标签样式为：
    ["xx1.png","xx2.png"] <xxx>
    ["xx1.png","xx2.png"] <xxx>
    """
    keys = ["train", "test"]
    txt_path = {key: dataset_path / f"{key}.txt" for key in keys}
    # 含有中文的样本是稀有样本，单独取出来后期可单独加大权重训练一次，不用区分训练集和测试集
    train_hard_txt_path = dataset_path / "train-hard.txt"

    label_data = {key: defaultdict(set) for key in keys}
    train_hard_data = defaultdict(set)

    # 词表中不存在的词动态添加
    words_path = dataset_path / "words_dict.txt"
    assert words_path.exists(), "words_dict.txt not exists!"
    words_dict = set()
    with open(words_path, "r") as f:
        words_dict |= set(v.strip() for v in f.readlines() if v.strip())
    logger.info(f"words dict num: {len(words_dict)}")

    for image_path in dataset_path.glob("images/**/*.png"):
        path = Path(image_path).absolute()
        if not path.is_file():
            logger.warning(f"sample {path} doesn't exists!")
            continue
        if not re.match(r"^((red)|(blue)|(yellow)|(black))-.{1,6}-\d+\.png$", path.name):
            logger.warning(f"sample {path} name is not valid!")
            continue
        label = path.name.split("-")[1]
        for v in label:
            if v not in words_dict:
                logger.warning(f"sample {path} label {label}:{v} not in words dict, add it!!")
                words_dict.add(v)
        rela_path = "images/" + path.name
        if random() < test_ratio:
            label_data["test"][label].add(rela_path)
        else:
            label_data["train"][label].add(rela_path)
        # 判断是否包括中文字符
        if any('\u4e00' <= c <= '\u9fa5' for c in label):
            train_hard_data[label].add(rela_path)
    for key in keys:
        logger.info(f"{key} sample num: {count(label_data[key].values())}")
        with open(txt_path[key], "w") as fout:
            for text, names in label_data[key].items():
                fout.write(f"{json.dumps(list(names), ensure_ascii=False)}\t{text}\n")
        logger.info(f"{key} data save to: {txt_path[key]}")
    with open(train_hard_txt_path, "w") as fout:
        for text, names in train_hard_data.items():
            fout.write(f"{json.dumps(list(names), ensure_ascii=False)}\t{text}\n")
        logger.info(f"train_hard data save to: {train_hard_txt_path}")
    logger.info(f"train-hard sample num: {count(train_hard_data.values())}")

    with open(words_path, "w") as f:
        f.write("\n".join(words_dict))
    logger.info(f"words dict save to: {words_path}, words num: {len(words_dict)}")


def count(values):
    return sum(len(v) for v in values)


if __name__ == "__main__":
    generate_label()
