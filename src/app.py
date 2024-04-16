#! -*- coding:utf-8 -*-
# /usr/bin/python3

"""
@Desc:
@Author: zhayongchun
@Date: 2023/12/20
"""
import io
import re
from typing import List

from PIL import Image
from loguru import logger
from fastapi import UploadFile, File, FastAPI, Request

from predict import data_util, predict

app = FastAPI()
logger.add("logs/visit.log", rotation="10 MB", encoding="utf-8", enqueue=True, compression="zip", retention="100 days")


@app.post("/captcha/predict")
async def upload_images(request: Request, files: List[UploadFile] = File(...)):
    filenames = [file.filename for file in files]
    channels = [re.split(r"[.\-_]", filename)[0] for filename in filenames]
    host = request.client.host
    logger.info(f"[{host}] visit /captcha/predict, channels: {channels}, file num: {len(files)}, files: {filenames}")
    if len(channels) != len(files):
        return {"error": "channels must be equal to files"}
    for channel in channels:
        if channel not in data_util.channels:
            return {"error": f"channel must be one of {data_util.channels}"}
    img_list = []
    for channel, file in zip(channels, files):
        contents = await file.read()
        # 将文件内容转换为字节流
        image_stream = io.BytesIO(contents)
        # 使用PIL库读取图片数据
        img = Image.open(image_stream)
        img_list.append(img)
    labels = predict(img_list, channels)
    ans = [{"filename": file.filename, "label": label} for file, label in zip(files, labels)]
    logger.info(f"predicts: {ans}")
    return {"data": ans}
