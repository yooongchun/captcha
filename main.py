# -*- coding:utf-8 -*-
"""
@Desc: Entry of the project
@Author: zoz-cool
"""
import argparse
import click
from pathlib import Path

from src import proj_path, assets_path, dataset_path
from src.helper import dataset_manage
from src.train import train as m_train


@click.group()
def cli():
    pass


@cli.command()
@click.option("-k", "--key", help="oss key")
@click.option("-d", "--save_dir", help="save directory")
def up(key: str, save_dir: str):
    dataset_manage.upload_to_oss(key, save_dir)


@cli.command()
@click.option("-k", "--key", help="oss key")
@click.option("-d", "--save_dir", help="save directory")
def dw(key: str, save_dir: str):
    dataset_manage.download_from_oss(key, save_dir)


@cli.command()
@click.option("--dataset", type=str, default=str(dataset_path / "labeled"), help="dataset directory")
@click.option("--vocabulary", type=Path, default=assets_path / "vocabulary.txt", help="vocabulary file")
@click.option("--save-path", type=Path, default=proj_path / "output/checkpoint", help="save directory")
@click.option("--log-path", type=Path, default=proj_path / "output/logs", help="log directory")
@click.option("--wandb-name", type=str, help="wandb name")
@click.option("--wandb-mode", type=str, help="wandb mode")
@click.option("--auto-num", type=int, default=1000000, help="auto number")
@click.option("--pretrained", type=str, help="pretrained model")
@click.option("--eval-freq", type=int, default=5, help="evaluate frequency")
@click.option("--save-freq", type=int, default=1, help="save frequency")
@click.option("--max-len", type=int, default=6, help="max length of captcha")
@click.option("--channel", type=click.Choice(["random", "text", "red", "blue", "black", "yellow"]), required=True,
              help="channel")
@click.option("--simple-mode", is_flag=True, help="simple mode")
@click.option("--model", type=str, default="custom", help="which model to use?")
@click.option("--batch-size", type=int, required=True, help="batch size")
@click.option("--num-epoch", type=int, required=True, help="number of epoch")
@click.option("--lr", type=float, default=0.001, help="learning rate")
@click.option("--num-workers", type=int, default=0, help="number of workers")
@click.option("--export", is_flag=True, help="export model")
@click.option("--evaluate", is_flag=True, help="evaluate model")
def train(dataset: str, vocabulary: str, save_path: str, log_path: str, wandb_name: str, wandb_mode: str, auto_num: int,
          pretrained: str, eval_freq: int, save_freq: int, max_len: int, channel: str, simple_mode: bool, model: str,
          batch_size: int, num_epoch: int, lr: float, num_workers: int, export: bool, evaluate: bool):
    args = argparse.Namespace(
            dataset_dir=dataset,
            vocabulary_path=vocabulary,
            save_dir=save_path,
            log_dir=log_path,
            wandb_name=wandb_name,
            wandb_mode=wandb_mode,
            auto_num=auto_num,
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
            evaluate=evaluate
    )
    trainer = m_train.Trainer(args)
    if args.export:
        trainer.export()
    elif args.evaluate:
        trainer.evaluate()
    else:
        trainer.train()


if __name__ == "__main__":
    cli()
