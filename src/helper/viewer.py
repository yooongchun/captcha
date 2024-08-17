"""
样本标签分布
"""
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
import json

# Modify rcParams to set the default font for Matplotlib
plt.rcParams['font.sans-serif'] = ['SimHei']  # 替换为您系统中的中文字体名称
plt.rcParams['axes.unicode_minus'] = False


def get_words(words_file):
    with open(words_file) as fin:
        return set(v.strip() for v in fin.readlines() if v.strip())


def get_words_freq(txt_file):
    with open(txt_file) as fin:
        lines = fin.readlines()
    count = Counter()
    for line in lines:
        if not line.strip():
            continue
        num = len(json.loads(line.split("\t")[0]))
        text = line.split("\t")[1].strip()
        for v in text:
            count[v] += num
    return count


def plot(words, count):
    counted = sorted(count.items(), key=lambda item: item[1], reverse=True)
    keys = [c[0] for c in counted]
    nums = [c[1] for c in counted]
    total = len(nums)
    ncol = 2
    num = 50
    npic = total // num + int(total % num > 0)
    nrow = npic // ncol + int(npic % ncol > 0)
    p, axes = plt.subplots(nrow, ncols=ncol)
    for i, ax in enumerate(axes.flatten()):
        start = i * num
        end = (i + 1) * num
        if start >= len(nums) or nums[start] == 0:
            break
        ax.bar(keys[start:end], nums[start:end])
        ax.set_ylabel(f"top {start}-{end}")
    plt.show()

    empty_map = [key for key in words if key not in count]
    print(f"Empty keys {len(empty_map)}: {empty_map}")
    print(f"Present keys {len(count)}: {keys}")


if __name__ == "__main__":
    dataset_path = Path(__file__).parent.parent.parent / "dataset/labeled"
    text_file = dataset_path / 'train.txt'
    character_set_file = dataset_path / 'words_dict.txt'
    plot(get_words(character_set_file), get_words_freq(text_file))
