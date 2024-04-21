    # 端到端不定长验证码识别模型

## 摘要

国税网发票查验验证码识别模型。

### 1. 数据集

- 可以自己生成

```bash
cd captcha/train/helper

pip3 install -r requirements.txt
python3 generate_captcha.py --num 100_000 --output ../../dataset
```

- 也可以使用已经生成好的，在这里下载：https://aistudio.baidu.com/datasetdetail/251503/0

### 2. 训练

```bash
cd captcha/train

pip3 install -r requirements.txt
bash train.sh
```

### 3. 部署

将output/checkpoint内的模型产出拷贝到deploy/inference目录

```bash
cd captcha/deploy
pip3 install -r requirements.txt
uvicorn predict:app --host 0.0.0.0 --port 8080
```

然后通过http `POSt`请求数据:`http://localhost:8080/captcha`

验证码图片通过files参数携带，可一次传递多个！
