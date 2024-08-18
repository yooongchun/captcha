<h1 align="center">
<a href="https://zoz.cool" ><img width=100 align="center" src="https://yooongchun.github.io/picx-images-hosting/zoz-logo.9gwhuriu9p.svg" alt="zoz-logo-z" ></a>
Captcha - 验证码识别
</h1>

<p aligin="center">
    <a href=""><img src="https://img.shields.io/badge/release-v1.0.0-blue?logo=github" ></a>
    <a href=""><img src="https://img.shields.io/badge/license-MIT-orange?logo=github" ></a>
<br/>
国税网发票查验验证码识别模型。含中文3000+文字和0-9数字+26大写字母。识别率约90%
</p>

---

# 如何使用 | How to use

## 数据生成

先从国税官网下载验证码图片，使用以下命令下载：
```bash
pip3 install -r src/helper/requirements.txt
python3 main.py download
```
刚开始会自动下载chromium浏览器内核，如果网络问题下载失败，可以手动下载，下载地址：https://download-chromium.appspot.com/ ，下载完成后解压缩，然后在启动时指定参数
```bash
python3 main.py download -b <your-browser-path>
```

<img src="https://yooongchun.github.io/picx-images-hosting/Snipaste_2024-08-17_07-44-27.5j44di864z.webp" width="50%"  alt=""/>

此时，程序就会自动开始下载验证码图片，默认会下载10000张之后停止，可以通过`-n`参数指定下载数量。

## 打标签

数据下载好之后需要开始打标签，可以使用以下命令启动打标签工具
```bash
python3 main.py tag
```
具体使用参数不多介绍，这里着重说一下`--pred-url`参数，这个参数是用来指定预测模型的地址，这样可以在打标签的时候实时预测验证码，这样可以提高打标签的效率。
比如你可以先打标签1000张，然后训练一个模型，然后用该模型提供预测服务，然后再打标签1000张，这样可以提高打标签的效率。
如下图，主页一共三个区域；1是待打标签的图片，2是已经打标签的图片，3是统计的当前已打标签的数据分布，每次启动的时候会优先选择分类数少的来打标签。

<img src="https://yooongchun.github.io/picx-images-hosting/zoz-2.6bgzv9x4j8.webp" width="50%" alt=""/>

点击start开始打标签，如果你指定了`--pred-url`的参数，那么这里会自动实时预测验证码，如果预测错误，可以手动删除填写标签值，没有的颜色留空即可，点击`tag it`或者回车完成打标。自动进入下一张图片打标。
如果发觉有错误需要修改上一张，则可以先点击`stop`停止当前打标，然后在主页面的右侧(2区域)找到你打标的图片双击修改标签即可。

<img src="https://yooongchun.github.io/picx-images-hosting/zoz-3.8ojmchaxq2.webp" width="30%" />
<img src="https://yooongchun.github.io/picx-images-hosting/zoz-4.1hs4z5qfqy.webp" width="47%" />

对于有些比较复杂的图片想要跳过，直接点击`skip`即可。

## 数据保存

数据是比较宝贵的，如果一不小心把辛辛苦苦打标好的数据弄丢了那可就太可惜了，为此，这里也提供了一种数据版本管理的方案，自动上传到云服务的对象管理中。
这里实现的是阿里云oss自动上传，使用前，需要准备好阿里云的oss相关配置信息
在`src/helper/oss_util.py`文件中配置
- access_key_id（放在环境变量中）
- access_key_secret（放在环境变量中）
- bucket_name（我这里是`zoz-captcha`）
- endpoint

```python
class OSSUtil:
    """OSS管理工具"""

    def __init__(self):
        self.endpoint = "https://oss-cn-beijing.aliyuncs.com"
        access_key_id = os.getenv("OSS_ACCESS_KEY_ID")
        access_key_secret = os.getenv("OSS_ACCESS_KEY_SECRET")
        assert access_key_id and access_key_secret, "Please set OSS_ACCESS_KEY_ID and OSS_ACCESS_KEY_SECRET"
        auth = oss2.Auth(access_key_id, access_key_secret)
        self.bucket = oss2.Bucket(auth, self.endpoint, "zoz-captcha")
```

然后使用以下命令上传和下载
```bash
# 上传
python3 main.py up
# 下载
python3 main.py dw
```
打完标之后直接上传，会自动根据远程版本推断最新版本，并自动压缩本地文件上传。下载的时候也会自动下载最新版本，当然也可以指定下载路径，不过需要注意的是，下载的时候会自动覆盖本地文件，需要注意本地的是否都已经保存上传。

<img width="48%" src="https://yooongchun.github.io/picx-images-hosting/zoz-5.2domemqbgf.webp" />   
<img width="48%" src="https://yooongchun.github.io/picx-images-hosting/zoz-6.6m3toggk92.webp" />

## 数据分布
数据的分布对效果影响较大，通过以下命令可以查看一下当前的数据分布，还可以看到当前哪些字没有样本，针对词频较低的字要增加样本。
```bash
python3 src/helper/viewer.py 
```
<img width="50%" src="https://yooongchun.github.io/picx-images-hosting/zoz-7.1ovcuxc85x.webp" alt="zoz-7" />

## 训练模型
> todo
