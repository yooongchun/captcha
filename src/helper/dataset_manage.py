#! /usr/bin/python3
# -*- coding:utf-8 -*-
"""
@Desc: Manage the dataset
Upload to oss or download from oss
"""
import re
from pathlib import Path
import tarfile

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


def create_tarfile(output_filename: str, source_dir: Path) -> Path:
    logger.info(f"Creating tarfile {output_filename}<--{source_dir}...")
    tar_path = source_dir.parent / output_filename
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(source_dir, arcname=Path(source_dir).name)
    return tar_path


def extract_tarfile(file_path: Path, extract_path: Path):
    tar_path = extract_path / file_path.stem.split("-")[0]
    if tar_path.exists():
        over = input(f"Directory {tar_path} already exists, overwrite it?[y|n]:")
        if over.lower() != "y":
            logger.info("Cancel extracting, returned!!!")
            return
    logger.info(f"Extracting {file_path} to {extract_path}...")
    with tarfile.open(file_path, 'r') as tar:
        tar.extractall(path=extract_path)


def count_img(file_path: Path) -> int:
    """Count the number of images in the directory"""
    return len(list(file_path.glob("*.png")))


def upload_to_oss(key: str, save_dir: str):
    """Upload the local file to oss"""
    # Check the file path
    if not save_dir:
        save_dir = dataset_path / "labeled"
    file_path = Path(save_dir).absolute()
    assert file_path.is_dir(), "The file does not exist"

    # generate name if key is not specified
    oss = oss_util.OSSUtil()
    if not key:
        files_vers = oss.list()
        logger.info(f"Current files in oss: {files_vers}")
        next_ver = get_next_version(files_vers)
        count = count_img(file_path / "images")
        key = file_path.stem + f"-{round(count / 1000, 2)}k-v{next_ver}.tgz"
    tar_path = create_tarfile(key, file_path)
    oss.upload(str(tar_path), key)
    tar_path.unlink()


def download_from_oss(key: str, save_dir: Path):
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
    tar_file = save_dir / key
    extract_tarfile(tar_file, save_dir)
    tar_file.unlink()
