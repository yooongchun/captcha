#! -*- coding: utf-8 -*-

import json
import os
import random
import pathlib
import argparse
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm
import numpy as np
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont
from scipy.interpolate import make_interp_spline
from src import assets_path, vocabulary_path


class FontUtil:
    """判断unicode字符是否在字体文件中"""

    def __init__(self, font_path: str):
        self.font = TTFont(font_path)

    def is_char_in_font(self, unicode_char):
        # 加载字体文件
        for cmap in self.font['cmap'].tables:
            if cmap.isUnicode():
                if ord(unicode_char) in cmap.cmap:
                    return True
        return False


class CaptchaGenerator:
    """
    生成验证码图片
    font_path: （数字+字母）字体路径
    font2_path: （汉字）字体路径
    vocabulary_path: 单词表路径
    width: 图片宽度
    height: 图片高度
    max_words: 最大字符数
    simple_mode: 是否简单模式，简单模式下只包含数字和字母
    """

    def __init__(self, font_path=str(assets_path / "font/3D-1.ttf"),
                 font2_path=str(assets_path / "font/SanJiJinSongJianTi-Xi-2.ttf"),
                 width=120, height=50, max_words=6, simple_mode=False):
        self.scale = 5
        self.width = width * self.scale
        self.height = height * self.scale
        self.simple_mode = simple_mode
        self.max_words = max_words
        self.vocabulary_path = vocabulary_path
        self.font = ImageFont.truetype(font_path, 40 * self.scale)
        self.font2 = ImageFont.truetype(font2_path, 30 * self.scale)
        self.characters = [chr(i) for i in range(65, 91)]  # 大写字母
        self.characters += [str(i) for i in range(10)]  # 阿拉伯数字
        self.vocabulary = self._load_vocabulary()  # 词汇表
        self.chinese_words = [w for w in self.vocabulary if w not in self.characters]
        # 判断字体文件是否包含全部字符
        font_util = FontUtil(font2_path)
        not_exists = [w for w in self.vocabulary if not font_util.is_char_in_font(w)]
        assert len(not_exists) == 0, f"字体文件{font2_path}不包含以下字符: {not_exists}"

    def _load_vocabulary(self):
        with open(self.vocabulary_path, encoding="utf-8") as f:
            vocabulary = f.readlines()
        vocabulary = [w.strip() for w in vocabulary if w.strip()]

        print(f"total vocabulary words: {len(vocabulary)}")
        return vocabulary

    @staticmethod
    def get_light_colors(num_colors):
        # 初始化一个空列表来存储RGB值
        light_colors = []

        # 生成指定数量的浅色RGB值
        for _ in range(num_colors):
            r = random.randint(200, 255)
            g = random.randint(200, 255)
            b = random.randint(200, 255)
            light_colors.append((r, g, b))

        return light_colors

    @staticmethod
    def get_dark_colors(num_colors):
        # 初始化一个空列表来存储RGB值
        dark_colors = []

        # 生成指定数量的深色RGB值
        for _ in range(num_colors):
            r = random.randint(100, 150)
            g = random.randint(100, 150)
            b = random.randint(100, 150)
            dark_colors.append((r, g, b))

        return dark_colors

    def draw_river(self, img: Image.Image):
        draw = ImageDraw.Draw(img)

        bg_colors = self.get_dark_colors(1)
        # 创建两条平行的样条曲线
        xs = np.linspace(0, self.width, num=500)
        y1 = 75 * np.sin(
                1.2 * np.pi * xs / self.width + 2 * np.random.uniform(-np.pi, np.pi)
        )
        y2 = y1 + self.height  # 修改这里使曲线更宽

        # 使用scipy的make_interp_spline函数生成样条曲线
        spl1 = make_interp_spline(xs, y1)
        spl2 = make_interp_spline(xs, y2)
        xnew = np.linspace(0, self.width, num=1000)
        y1new = spl1(xnew)
        y2new = spl2(xnew)

        # 将两条曲线之间的区域填充颜色，设置颜色为淡绿色
        points = list(
                zip(
                        np.concatenate([xnew, xnew[::-1]]), np.concatenate([y1new, y2new[::-1]])
                )
        )
        draw.polygon(points, fill=bg_colors[0])
        # 添加带颜色的噪点
        r = 3
        for _ in range(200):  # 添加1000个噪点
            x = random.randint(0, self.width)
            y = random.randint(0, self.height)
            color = (
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255),
            )  # 随机颜色
            draw.ellipse((x - r, y - r, x + r, y + r), fill=color)

    @staticmethod
    def draw_line(img: Image.Image, color=(0, 255, 0)):
        draw = ImageDraw.Draw(img)

        width, height = img.size
        num = random.randint(0, 2)
        # 添加两条随机线段
        for _ in range(num):
            x1 = random.randint(0, width)
            y1 = random.randint(0, height)
            x2 = random.randint(0, width)
            y2 = random.randint(0, height)
            draw.line([(x1, y1), (x2, y2)], fill=color, width=5)

    @staticmethod
    def shift_image(img: Image.Image, dx: int, dy: int) -> Image.Image:
        """
        平移图片
        :param img: PIL Image对象
        :param dx: 水平平移距离，正值向右，负值向左
        :param dy: 垂直平移距离，正值向下，负值向上
        :return: 平移后的PIL Image对象
        """
        width, height = img.size
        # 使用Image.transform函数进行平移
        new_img = img.transform((width, height), Image.AFFINE, (1, 0, dx, 0, 1, dy))
        return new_img

    def _gen_label(self, ratio=0.4, num=6):
        label = []  # 存储label: 类型， 0-->字母或数字，1-->汉字
        if random.random() < ratio:
            # 汉字随机生成1~3个
            chinese_num = random.randint(1, min(3, (num + 1) // 2))
            for i in range(chinese_num):
                # 随机选择一个汉字
                text = random.choice(self.chinese_words)
                label.append([text, 1])
        for _ in range(len(label), num):
            # 随机选择一个字母或数字
            text = random.choice(self.characters)
            label.append([text, 0])
        return label

    def draw_text_rotate(self, img: Image.Image, num, ratio=0.4) -> dict:
        """
        若要添加字样，则设为true，数字默认开启
        :param ratio: 汉字出现概率
        :param img:
        :param num: 验证码字符个数
        :return: label
        """

        colors = {
                (0, 0, 0): "black",
                (255, 0, 0): "red",
                (0, 0, 255): "blue",
                (255, 255, 0): "yellow",
        }  # 随机颜色
        # 生成label
        label = self._gen_label(ratio=ratio, num=num)
        # 当前字符的起始位置，尽量保证居中
        word_width = int(self.width / self.max_words * 0.9)
        start_x = word_width * (self.max_words - num) / 2
        for i in range(num):
            idx = num - 1 - i
            text, t = label[idx]
            img0 = Image.new("RGBA", img.size, (255, 255, 255, 0))
            draw0 = ImageDraw.Draw(img0, mode="RGBA")
            # 放在中间
            h, w = self.font.size, self.font.getlength(text)
            x = self.width / 2 - w / 2 * (1 + t)
            y = self.height / 2 - h / 2
            color = random.choice(list(colors.keys()))
            label[idx].append(color)
            draw0.text((x, y), text, color, font=self.font2 if t > 0 else self.font)
            # 获得一个旋转字
            img1 = img0.rotate(random.uniform(-50, 50), Image.BILINEAR, expand=False)
            # 平移
            shift_x = int(start_x + word_width / 2 - self.width / 2 + word_width * i)
            img2 = self.shift_image(img1, shift_x, random.randint(-6, 6))
            # Crop img1 to the same size as img
            img.paste(Image.alpha_composite(img, img2))
        # 按照color分组
        label_map = {}
        for text, _, c in label:
            key = colors[c]
            label_map[key] = label_map.get(key, "") + text
        label_map["text"] = "".join([text for text, _, _ in label])

        return label_map

    def gen_one(self, min_num=4, max_num=6):
        """生成一个验证码"""
        assert min_num <= max_num <= self.max_words, "不能超出最大字符数"
        bg_colors = self.get_light_colors(1)
        img = Image.new("RGBA", (self.width, self.height), color=bg_colors[0])
        self.draw_river(img)
        self.draw_line(img)
        # 通过设置比率为0控制不生成汉字
        chinese_words_ratio = 0 if self.simple_mode else 0.4
        label = self.draw_text_rotate(img, random.randint(min_num, max_num), ratio=chinese_words_ratio)
        return img.resize((self.width // self.scale, self.height // self.scale)), label

    def gen_batch(self, batch_size=100, min_num=4, max_num=6):
        """批次生成"""
        assert min_num <= max_num <= self.max_words, "不能超出最大字符数"
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            results = executor.map(self.gen_one, [min_num] * batch_size, [max_num] * batch_size)
        return list(results)


def save_batch(results, output_dir: pathlib.Path, test_ratio=0.4, index=0):
    train_file = output_dir / "train.json"
    test_file = output_dir / "test.json"
    image_dir = output_dir.absolute() / "images"
    os.makedirs(image_dir, exist_ok=True)
    # 加载已存在的json文件
    train_records = []
    test_records = []
    if os.path.isfile(train_file):
        with open(train_file, "r", encoding="utf-8") as f:
            train_records = json.load(f)
    if os.path.isfile(test_file):
        with open(test_file, "r", encoding="utf-8") as f:
            test_records = json.load(f)
    # 保存结果
    channels = ["black", "red", "blue", "yellow"]
    for i, (img, label) in enumerate(results):
        # 随机选择一个channel
        channel = random.choice([k for k in label if k in channels])
        # 保存文件
        save_path = image_dir / f"{channel}-{index + i:0>7d}-{label[channel]}.png"
        img.save(save_path)
        # 保存标签
        label["index"] = index + i
        label["path"] = save_path.parent.name + "/" + save_path.name
        if random.random() > test_ratio:
            train_records.append(label)
        else:
            test_records.append(label)
    with open(train_file, "w", encoding="utf-8") as f1, open(
            test_file, "w", encoding="utf-8"
    ) as f2:
        json.dump(train_records, f1, ensure_ascii=False, indent=4)
        json.dump(test_records, f2, ensure_ascii=False, indent=4)


def main():
    args = parse_args()

    if os.path.isdir(args.output):
        print("remove old dataset: ", args.output)
        shutil.rmtree(args.output)

    batch_size = 10000
    gen = CaptchaGenerator(max_words=args.max_num, simple_mode=args.simple_mode)

    tbar = tqdm(range(args.num // batch_size + int(args.num % batch_size != 0)))
    for i in tbar:
        batch_data = gen.gen_batch(batch_size=min(args.num, batch_size), min_num=args.min_num, max_num=args.max_num)
        tbar.set_description(f"batch={i + 1}")

        save_batch(
                batch_data,
                args.output,
                test_ratio=args.test_ratio,
                index=batch_size * i,
        )


def parse_args():
    proj_dir = pathlib.Path(__file__).absolute().parent.parent.parent

    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, required=True, help="需要生成的样本数量")
    parser.add_argument("--min_num", type=int, default=4, help="验证码最短长度")
    parser.add_argument("--max_num", type=int, default=6, help="验证码最大长度")
    parser.add_argument("--output", type=pathlib.Path, default=proj_dir / "dataset/generated", help="数据保存路径")
    parser.add_argument("--test_ratio", type=float, default=0.4, help="测试集所占比例")
    parser.add_argument("--simple_mode", action="store_true", help="是否简单模式，简单模式下只包含数字和字母")

    return parser.parse_args(sys.argv[1:])


def is_char_in_font(unicode_char, font):
    for cmap in font['cmap'].tables:
        if cmap.isUnicode():
            if ord(unicode_char) in cmap.cmap:
                return True
    return False


if __name__ == "__main__":
    main()
