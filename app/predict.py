#! -*- coding: utf-8 -*-

import pathlib
from typing import List

import paddle
from PIL import Image
from loguru import logger
from ppqi import InferenceModel

from util import DataUtil
from decoder import Decoder

# 加载模型
inference_model_path = pathlib.Path(__file__).absolute().parent.parent / "inference/model"
logger.info(f"Load model from {inference_model_path}...")
model = InferenceModel(
    modelpath=str(inference_model_path),
    use_gpu=False,
    use_mkldnn=True
)
model.eval()
# 解码器
data_util = DataUtil()
decoder = Decoder(data_util.vocabulary)


def predict(img_list: List[Image.Image], colors: List[str]):
    batch_img = paddle.to_tensor([data_util.process_img(img) for img in img_list], dtype=paddle.float32)
    batch_color = paddle.to_tensor([data_util.process_channel(color) for color in colors], dtype=paddle.float32)
    outputs = paddle.to_tensor(model(batch_img, batch_color))
    outputs = paddle.nn.functional.softmax(outputs, axis=-1)
    # 解码获取识别结果
    pred_list = []
    for output in outputs:
        pred = decoder.ctc_greedy_decoder(output)
        pred_list.append(pred)
    return pred_list
