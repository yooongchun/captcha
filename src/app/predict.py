#! -*- coding: utf-8 -*-

import paddle
from PIL import Image
from loguru import logger
from ppqi import InferenceModel

from src import inference_path, vocabulary_path
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
data_util = DataUtil(vocabulary_path)
decoder = Decoder(data_util.get_vocabulary())


def predict(img: Image.Image, r: int = 3):
    batch_img = paddle.to_tensor([data_util.process_img(img)], dtype=paddle.float32)
    outputs = paddle.to_tensor(model(batch_img))
    outputs = paddle.nn.functional.softmax(outputs, axis=-1)
    # 解码获取识别结果
    label, ci_list = decoder.ctc_greedy_decoder(outputs[0], keep_ci=True)
    ci_new = [(c, round(ci, r)) for c, ci in ci_list]
    avg_ci = sum(ci for _, ci in ci_list) / len(ci_list) if len(ci_list) > 0 else 0
    ci_new.append((label, round(avg_ci, r)))
    return label, ci_new
