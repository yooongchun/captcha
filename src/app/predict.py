#! -*- coding: utf-8 -*-

import paddle
from PIL import Image
from loguru import logger
from ppqi import InferenceModel

from src import inference_path
from src.helper.util import DataUtil
from src.helper.decoder import Decoder

# 加载模型
inference_model_path = inference_path / "model"
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


def predict(img: Image.Image, channel: str):
    batch_img = paddle.to_tensor([data_util.process_img(img)], dtype=paddle.float32)
    batch_channel = paddle.to_tensor([data_util.process_channel(channel)], dtype=paddle.float32)
    outputs = paddle.to_tensor(model(batch_img, batch_channel))
    outputs = paddle.nn.functional.softmax(outputs, axis=-1)
    # 解码获取识别结果
    return decoder.ctc_greedy_decoder(outputs[0])
