#! -*- coding:utf-8 -*-
# /usr/bin/python3

"""
@Desc:
@Author: zhayongchun
@Date: 2023/12/20
"""
import base64
import io

from PIL import Image
from loguru import logger
from pydantic import BaseModel
from fastapi import UploadFile, File, FastAPI, Request

from src.app.predict import predict

app = FastAPI()
logger.add("logs/visit.log", rotation="10 MB", encoding="utf-8", enqueue=True, compression="zip", retention="100 days")


class Item(BaseModel):
    img: str


@app.post("/api/v1/captcha/predict-by-file")
async def upload_images(request: Request, file: UploadFile = File(...)):
    host = request.client.host
    logger.info(f"[{host}] visit {request.url.path}, filename: {file.filename}")
    contents = await file.read()
    # 将文件内容转换为字节流
    image_stream = io.BytesIO(contents)
    # 使用PIL库读取图片数据
    img = Image.open(image_stream)
    label, ci = predict(img)
    res = {"code": 0, "data": {"filename": file.filename, "predict_label": label, "ci": ci}}
    logger.info(f"predicts: {res}")
    return res


@app.post("/api/v1/captcha/predict-by-base64")
async def upload_base64(request: Request, item: Item):
    host = request.client.host
    logger.info(f"[{host}] visit {request.url.path} with base64 img")
    img = Image.open(io.BytesIO(base64.b64decode(item.img)))
    # 使用PIL库读取图片数据
    label, ci = predict(img)
    res = {"code": 0, "data": {"filename": "-", "predict_label": label, "ci": ci}}
    logger.info(f"predicts: {res}")
    return res
