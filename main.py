# -*- coding:utf-8 -*-
"""
@Desc: Entry of the project
@Author: zoz-cool
"""
import argparse
from pathlib import Path
import pandas as pd

import uvicorn
import click

from src import proj_path, vocabulary_path, dataset_path, assets_path


@click.group()
def cli():
    pass


@cli.command()
@click.option("-k", "--key", help="oss key")
@click.option("-d", "--save-dir", help="save directory")
def up(key: str, save_dir: str):
    """upload dataset to oss"""
    from src.helper import dataset_manage

    dataset_manage.upload_to_oss(key, save_dir)


@cli.command()
@click.option("-k", "--key", help="oss key")
@click.option("-d", "--save-dir", type=Path, help="save directory")
def dw(key: str, save_dir: Path):
    """download dataset from oss"""
    from src.helper import dataset_manage

    dataset_manage.download_from_oss(key, save_dir)


@cli.command()
@click.option("--dataset-dirs", type=str, default=str(dataset_path / "labeled"), help="dataset directory")
@click.option("--vocabulary", type=Path, default=vocabulary_path, help="vocabulary file")
@click.option("--save-path", type=Path, default=proj_path / "output/checkpoint", help="save directory")
@click.option("--log-path", type=Path, default=proj_path / "output/logs", help="log directory")
@click.option("--wandb-name", type=str, help="wandb name")
@click.option("--wandb-mode", type=str, default="online", help="wandb mode")
@click.option("--pretrained", type=str, help="pretrained model")
@click.option("--eval-freq", type=int, default=5, help="evaluate frequency")
@click.option("--save-freq", type=int, default=1, help="save frequency")
@click.option("--max-len", type=int, default=6, help="max length of captcha")
@click.option("--channel", type=click.Choice(["text", "red", "blue", "black", "yellow"]), required=True, help="channel")
@click.option("--data-type", type=click.Choice(["color", "single"]), required=True, help="data type")
@click.option("--simple-mode", is_flag=True, help="simple mode")
@click.option("--model", type=str, default="custom", help="which model to use?")
@click.option("-b", "--batch-size", type=int, required=True, help="batch size")
@click.option("-n", "--num-epoch", type=int, required=True, help="number of epoch")
@click.option("--lr", type=float, default=0.001, help="learning rate")
@click.option("--num-workers", type=int, default=0, help="number of workers")
@click.option("--export", is_flag=True, help="export model")
@click.option("--evaluate", is_flag=True, help="evaluate model")
def train(
    dataset_dirs: str,
    vocabulary: str,
    save_path: str,
    log_path: str,
    wandb_name: str,
    wandb_mode: str,
    pretrained: str,
    eval_freq: int,
    save_freq: int,
    max_len: int,
    channel: str,
    data_type: str,
    simple_mode: bool,
    model: str,
    batch_size: int,
    num_epoch: int,
    lr: float,
    num_workers: int,
    export: bool,
    evaluate: bool,
):
    """train the model"""
    from src.train import train as m_train

    args = argparse.Namespace(
        dataset_dir=dataset_dirs,
        vocabulary_path=vocabulary,
        save_dir=save_path,
        log_dir=log_path,
        wandb_name=wandb_name,
        wandb_mode=wandb_mode,
        data_type=data_type,
        pretrained=pretrained,
        eval_freq=eval_freq,
        save_freq=save_freq,
        max_len=max_len,
        channel=channel,
        simple_mode=simple_mode,
        model=model,
        batch_size=batch_size,
        num_epoch=num_epoch,
        lr=lr,
        num_workers=num_workers,
        export=export,
        evaluate=evaluate,
    )
    trainer = m_train.Trainer(args)
    if args.export:
        trainer.export()
    elif args.evaluate:
        trainer.evaluate()
    else:
        trainer.train()


@cli.command()
@click.option("--host", default="0.0.0.0", type=str, help="host")
@click.option("--port", default=8000, type=int, help="port")
def app(host: str, port: int):
    """run the app"""
    from src.app import app as m_app

    uvicorn.run(m_app.app, host=host, port=port)


@cli.command()
@click.option("-s", "--save-dir", help="output directory")
@click.option("-n", "--num", default=10000, help="how many captcha images to download")
@click.option("-d", "--debug", is_flag=True, help="debug mode")
@click.option("-b", "--browser-path", help="chrome browser path")
def download(save_dir: str, num: int, debug: bool, browser_path: str):
    """download captcha images"""
    from src.helper import download_captcha

    inv_data = pd.read_csv(str(assets_path / "inv_data.csv"), encoding="utf-8", dtype=str)
    inv_data = inv_data.reset_index(drop=True)
    if not save_dir:
        save_dir = str(dataset_path / "origin")
    download_captcha.task(inv_data, save_dir, num, debug=debug, browser_path=browser_path)


@cli.command()
@click.option("--dataset-dir", type=str, default="origin")
@click.option("--output-dir", type=str, default="labeled")
@click.option("--test-ratio", type=float, default=0.1)
@click.option("--pred-host", type=str, default="")
def tag(dataset_dir, output_dir, test_ratio, pred_host):
    """tag the dataset"""
    from src.helper import make_tag

    dataset_dir = dataset_path / dataset_dir
    save_dir = dataset_path / output_dir / "images"
    save_dir.mkdir(parents=True, exist_ok=True)
    make_tag.main(dataset_dir, save_dir, test_ratio, pred_host)


@cli.command()
@click.option("--src", type=Path, help="source image")
@click.option("--dst", type=Path, help="destination image")
def split(src: Path, dst: Path):
    """split channel"""
    from src.helper import split_channel

    if not src:
        src = dataset_path / "origin"
    if not dst:
        dst = dataset_path / "splitted"
    split_channel.convert(Path(src), Path(dst))


if __name__ == "__main__":
    cli()
