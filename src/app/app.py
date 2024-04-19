#! -*- coding:utf-8 -*-
# /usr/bin/python3

"""
@Desc:
@Author: zhayongchun
@Date: 2023/12/20
"""
import io

from PIL import Image
from loguru import logger
from fastapi import UploadFile, File, FastAPI, Request, Form

from src.app.predict import data_util, predict

app = FastAPI()
logger.add("logs/visit.log", rotation="10 MB", encoding="utf-8", enqueue=True, compression="zip", retention="100 days")


@app.post("/api/v1/captcha/predict")
async def upload_images(request: Request, channel: str = Form(), file: UploadFile = File(...)):
    host = request.client.host
    logger.info(f"[{host}] visit {request.url.path}, channel: {channel}, filename: {file.filename}")
    if channel not in data_util.channels:
        return {"code": 1, "error": f"channel must be one of {data_util.channels}"}
    contents = await file.read()
    # 将文件内容转换为字节流
    image_stream = io.BytesIO(contents)
    # 使用PIL库读取图片数据
    img = Image.open(image_stream)
    label = predict(img, channel)
    res = {"code": 0, "data": {"filename": file.filename, "predict_label": label}}
    logger.info(f"predicts: {res}")
    return res
