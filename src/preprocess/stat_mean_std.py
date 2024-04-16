#! -*- coding: utf-8 -*-

import os
import time

import numpy as np
from concurrent.futures import ThreadPoolExecutor

try:
    import cv2
except ImportError:
    os.system('pip3 install opencv-python')
    import cv2


def process_image(filename):
    img = cv2.imdecode(np.fromfile(filename, dtype=np.uint8), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    pixel_values = img.sum(axis=(0, 1))
    square_pixel_values = np.sum(np.square(img), axis=(0, 1))
    return pixel_values, square_pixel_values


def compute_mean_std_multithread(dataset_dir, num_threads=8):
    num_images = 0
    sum_pixel_values = np.zeros(3)
    sum_square_pixel_values = np.zeros(3)

    # 获取所有图片文件的路径
    filenames = [os.path.join(dataset_dir, filename) for filename in os.listdir(dataset_dir) if
                 filename.endswith('.jpg') or filename.endswith('.png')]
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # 使用线程池并行处理所有图片文件
        for pixel_values, square_pixel_values in executor.map(process_image, filenames):
            sum_pixel_values += pixel_values
            sum_square_pixel_values += square_pixel_values
            num_images += 1

    img = cv2.imdecode(np.fromfile(filenames[0], dtype=np.uint8), cv2.IMREAD_COLOR)
    mean = sum_pixel_values / (num_images * img.shape[0] * img.shape[1])
    std = np.sqrt((sum_square_pixel_values / (num_images * img.shape[0] * img.shape[1])) - np.square(mean))

    return mean, std


if __name__ == '__main__':
    start = time.time()
    proj_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dataset_dir = os.path.join(proj_dir, 'dataset', 'captcha_100w', 'images')
    mean, std = compute_mean_std_multithread(dataset_dir, os.cpu_count() * 2)
    end = time.time()
    print("time:", end - start)
    print(mean, std)
