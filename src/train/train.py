#! -*- coding: utf-8 -*-

import json
import argparse
from pathlib import Path

import paddle
import prettytable
from loguru import logger

from src.train import model, dataset, metric, loss


class PrintLastLROnEpochStart(paddle.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        logger.info(f"Epoch {epoch + 1}, learning rate is {self.model._optimizer.get_lr()}")


def check_dataset(dataset_dir: str):
    """检查数据集是否存在"""
    logger.info(f"Check dataset {dataset_dir}...")
    train_json = Path(dataset_dir, "train.json")
    meta_info = []
    if train_json.is_file():
        with open(train_json, "rb") as fp:
            train_meta = json.load(fp)
            meta_info += train_meta
            logger.info(f"Train dataset has {len(train_meta)} samples")
    test_json = Path(dataset_dir, "test.json")
    if test_json.is_file():
        with open(test_json, "rb") as fp:
            test_meta = json.load(fp)
            meta_info += test_meta
            logger.info(f"Test dataset has {len(test_meta)} samples")
    ok = True
    for meta in meta_info:
        img_path = Path(dataset_dir, meta["path"])
        if not img_path.is_file():
            logger.error(f"Image {img_path} not found")
            ok = False
    return ok


class Trainer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self._init_data()
        self._init_model()

    def _init_data(self):
        # 传入数据集地址时使用已有数据集，否则边训练边生成数据集
        if self.args.evaluate:  # evaluate 模式
            assert self.args.dataset_dir is not None, "dataset_dir must be set when evaluate"
            logger.info(f"Start evaluate for dataset {self.args.dataset_dir}...")
            self.test_dataset = dataset.CaptchaDataset(
                    dataset_dirs=self.args.dataset_dir,
                    vocabulary_path=self.args.vocabulary_path,
                    mode="test",
                    channel=self.args.channel,
                    max_len=self.args.max_len,
                    simple_mode=self.args.simple_mode,
                    data_type=self.args.data_type,
            )
            self.test_dataloader = paddle.io.DataLoader(
                    self.test_dataset,
                    batch_size=self.args.batch_size,
                    shuffle=True,
                    num_workers=self.args.num_workers,
                    use_shared_memory=False,
            )
        else:  # 获取训练数据
            self.train_dataset = dataset.CaptchaDataset(
                    dataset_dirs=self.args.dataset_dir,
                    vocabulary_path=self.args.vocabulary_path,
                    mode="train",
                    channel=self.args.channel,
                    max_len=self.args.max_len,
                    simple_mode=self.args.simple_mode,
                    data_type=self.args.data_type,
            )
            self.train_dataloader = paddle.io.DataLoader(
                    self.train_dataset,
                    batch_size=self.args.batch_size,
                    shuffle=True,
                    num_workers=self.args.num_workers,
                    use_shared_memory=False,
            )

            # 获取测试数据
            self.test_dataset = dataset.CaptchaDataset(
                    dataset_dirs=self.args.dataset_dir,
                    vocabulary_path=self.args.vocabulary_path,
                    mode="test",
                    channel=self.args.channel,
                    max_len=self.args.max_len,
                    simple_mode=self.args.simple_mode,
                    data_type=self.args.data_type,
            )
            self.test_dataloader = paddle.io.DataLoader(
                    self.test_dataset,
                    batch_size=self.args.batch_size,
                    shuffle=False,
                    num_workers=self.args.num_workers,
                    use_shared_memory=False,
            )
        self.vocabulary = self.test_dataset.data_util.get_vocabulary()
        self.num_classes = len(self.vocabulary)

        t = prettytable.PrettyTable(["field", "number"])
        t.add_row(["num_classes", self.num_classes])
        if hasattr(self, "train_dataset"):
            t.add_row(["train_dataset", len(self.train_dataset)])
        t.add_row(["test_dataset", len(self.test_dataset)])
        print(t)

    def _init_model(self):
        # 获取模型
        m = model.Model(self.num_classes, self.args.max_len)
        img_size = self.test_dataset[0][0].shape
        label_size = self.test_dataset[0][1].shape
        inputs_shape = paddle.static.InputSpec([None, *img_size], dtype="float32", name="input")
        labels_shape = paddle.static.InputSpec([None, *label_size], dtype="int64", name="label")
        self.model = paddle.Model(m, inputs_shape, labels_shape)

        # 打印模型和数据信息
        self.model.summary(input_size=(self.args.batch_size, *img_size))

        # 设置优化方法
        def make_optimizer(parameters=None):
            boundaries = [5, 50, 100]
            warmup_steps = 4
            values = [self.args.lr * (0.1 ** i) for i in range(len(boundaries) + 1)]
            learning_rate = paddle.optimizer.lr.PiecewiseDecay(boundaries=boundaries, values=values)
            learning_rate = paddle.optimizer.lr.LinearWarmup(
                    learning_rate=learning_rate,
                    warmup_steps=warmup_steps,
                    start_lr=self.args.lr / 5.0,
                    end_lr=self.args.lr,
                    verbose=False,
            )
            optimizer = paddle.optimizer.Adam(learning_rate=learning_rate, parameters=parameters)
            return optimizer

        self.optimizer = make_optimizer(self.model.parameters())
        # 获取损失函数
        ctc_loss = loss.CTCLoss(self.num_classes)

        self.model.prepare(
                self.optimizer,
                ctc_loss,
                metrics=[metric.WordsErrorRate(self.vocabulary), metric.SampleAccuracy(self.vocabulary)],
        )

        # 加载预训练模型
        if self.args.pretrained:
            logger.info(f"Load pretrained model from {self.args.pretrained}")
            self.model.load(self.args.pretrained)

    def train(self):
        """开始训练"""
        assert check_dataset(self.args.dataset_dir), "check dataset failed!"

        vdl_log_dir = str(Path(self.args.log_dir, "vdl"))
        callbacks = [
                paddle.callbacks.VisualDL(log_dir=vdl_log_dir),
                paddle.callbacks.LRScheduler(by_step=False, by_epoch=True),
                PrintLastLROnEpochStart(),
        ]
        if self.args.wandb_mode in ["online", "offline"]:
            name = f"bs{self.args.batch_size}"
            if self.args.wandb_name:
                name = name + "-" + self.args.wandb_name
            Path(self.args.log_dir, "wandb").mkdir(parents=True, exist_ok=True)
            logger.info(f"Use wandb to record log, name: {name}, mode: {self.args.wandb_mode}")
            wandb_callback = paddle.callbacks.WandbCallback(
                    project="captcha",
                    dir=self.args.log_dir,
                    name=name,
                    mode=self.args.wandb_mode,
                    job_type="simple" if self.args.simple_mode else "complex",
                    group=self.args.channel,
            )
            callbacks.append(wandb_callback)
        self.model.fit(
                train_data=self.train_dataloader,
                eval_data=self.test_dataloader,
                epochs=self.args.num_epoch,
                callbacks=callbacks,
                eval_freq=self.args.eval_freq,
                log_freq=1,
                save_freq=self.args.save_freq,
                save_dir=self.args.save_dir,
                verbose=1,
        )
        # save for inference
        self.model.save(str(Path(self.args.save_dir, "inference/model")), False)

    def export(self):
        inference_dir = Path(self.args.save_dir, "inference/model")
        logger.info(f"Export inference model to {inference_dir}...")
        self.model.save(str(inference_dir), False)

    def evaluate(self):
        res = self.model.evaluate(self.test_dataloader, batch_size=self.args.batch_size, verbose=1)
        logger.info(res)
