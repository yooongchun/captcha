#! /usr/bin/python3
# -*- coding:utf-8 -*-
"""
@Desc: Manage the dataset
Upload to oss or download from oss
"""
import re
import pathlib

import click
from loguru import logger

from src import dataset_path
from src.helper import oss_util


def get_next_version(file_vers: list) -> int:
    if not file_vers:
        return 1
    vers = set()
    for ver in file_vers:
        if re.match(r".*v\d+.*", ver):
            v = int(ver.split("-")[-1].split(".")[0][1:])
            vers.add(v)
    return max(vers) + 1


def upload_to_oss(local_path: str):
    """Upload the local file to oss"""
    assert local_path, "Please specify the local file path"
    file_path = pathlib.Path(local_path).absolute()
    assert file_path.is_file(), "The file does not exist"
    oss = oss_util.OSSUtil()
    files_vers = oss.list()
    logger.info(f"Current files in oss: {files_vers}")
    next_ver = get_next_version(files_vers)
    filename = file_path.stem + f"-v{next_ver}" + file_path.suffix
    oss.upload(local_path, filename)


def download_from_oss(key: str, save_dir: str):
    """Download the file from oss"""
    if not save_dir:
        save_dir = dataset_path
    if not key:
        oss = oss_util.OSSUtil()
        files_vers = oss.list()
        if not files_vers:
            logger.info("No files in oss")
            return
        cur_ver = get_next_version(files_vers) - 1
        key = [f for f in files_vers if f"-v{cur_ver}." in f][0]
        logger.info(f"Latest version file: {key}")
    oss = oss_util.OSSUtil()
    oss.download(key, str(save_dir))


@click.command()
@click.option("-m", "--mode", required=True, help="upload or download")
@click.option("-k", "--key", help="oss key")
@click.option("-d", "--save_dir", help="save directory")
@click.option("-p", "--local_path", help="local file path")
def main(mode: str, key: str, save_dir: str, local_path: str):
    if mode == "upload":
        upload_to_oss(local_path)
    elif mode == "download":
        download_from_oss(key, save_dir)
    else:
        logger.error("Invalid mode, please choose upload or download!")


if __name__ == "__main__":
    main()
