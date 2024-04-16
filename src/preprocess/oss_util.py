# -*- coding: utf-8 -*-

import os
import oss2


class OSSUtil:
    """OSS管理工具"""

    def __init__(self):
        self.endpoint = "https://oss-cn-beijing.aliyuncs.com"
        access_key_id = os.getenv("OSS_ACCESS_KEY_ID")
        access_key_secret = os.getenv("OSS_ACCESS_KEY_SECRET")
        auth = oss2.Auth(access_key_id, access_key_secret)
        self.bucket = oss2.Bucket(auth, self.endpoint, "zoz-captcha")

    def upload(self, file_path: str):
        """上传文件"""
        assert os.path.exists(file_path), f"File not exists: {file_path}"
        key = os.path.basename(file_path)
        print(f"Uploading {file_path} to {key}...")
        self.bucket.put_object(key, open(file_path, "rb").read())

    def download(self, key: str, save_dir: str):
        """下载文件"""
        print(f"Downloading {key} to {save_dir}...")
        with open(save_dir + "/" + key, "wb") as f:
            f.write(self.bucket.get_object(key).read())

    def delete(self, key: str):
        """删除文件"""
        self.bucket.delete_object(key)

    def list(self):
        """列举文件"""
        return [object_info.key for object_info in oss2.ObjectIterator(self.bucket)]


if __name__ == "__main__":
    import dotenv

    dotenv.load_dotenv()
    util = OSSUtil()
    util.upload("C:/Users/zyc12/Documents/captcha/dataset/labeled-3.2k-v2.tgz")
    print(util.list())
