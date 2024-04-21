#! -*- coding:utf-8 -*-
# /usr/bin/python3

"""
@Desc:
@Author: zhayongchun
@Date: 2023/12/21
"""
import os
import time
import random
import shutil
import json
import pathlib
import threading

import requests
from glob import glob
from loguru import logger
from PySide6.QtWidgets import (
    QApplication,
    QListWidget,
    QHBoxLayout,
    QWidget,
    QPushButton,
    QLabel,
    QVBoxLayout,
    QDialog,
    QLineEdit,
    QMessageBox,
    QProgressBar,
)
from PySide6.QtGui import QPixmap, QFont
from PySide6.QtCore import QSize, Qt, QMetaObject, Q_ARG
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


def get_predict_label(host: str, img_path: str, channel: str):
    assert channel in [
        "red",
        "blue",
        "black",
        "yellow",
    ], f"channel {channel} is not supported!"

    st = time.time()
    res = requests.post(
        f"{host}/api/v1/captcha/predict",
        files={"file": open(img_path, "rb")},
        data={"channel": channel},
    )
    ed = time.time()
    assert res.status_code == 200, f"request failed, status code: {res.status_code}"
    assert (
        res.json()["code"] == 0
    ), f"request failed, error message: {res.json()['error']}"

    pred_label = res.json()["data"]["predict_label"]
    logger.info(
        f"request time: {round(ed - st, 2)}s, filename: {os.path.basename(img_path)}, "
        f"channel: {channel}, label: {pred_label}"
    )
    return pred_label


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi, facecolor=None)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)


class TagDialog(QDialog):
    """one picture with multi tag"""

    def __init__(self, img_path: str, pred_host: str = ""):
        super().__init__()

        self.img_path = img_path
        self.pred_host = pred_host
        # Layout
        self.setWindowTitle("Make Multi-Tag Window")
        self.setFixedSize(QSize(400, 400))

        # make a new layout
        layout = QVBoxLayout()

        # make a label to show the image
        scale = 2
        self.img_label = QLabel()
        self.img_label.setFixedSize(QSize(int(120 * scale), int(50 * scale)))
        self.img_label.setScaledContents(True)

        self.img_label.setPixmap(QPixmap(img_path))

        layout.addWidget(self.img_label)
        layout.addStretch(1)
        # add all channel color input line
        self.inputs = {}
        for color in ["black", "red", "blue", "yellow"]:
            hbox = QHBoxLayout()
            label = QLabel(color.upper().ljust(15))
            # label.setStyleSheet(f"color: {color}")
            label.setFont(QFont("Arial", 16))
            hbox.addWidget(label)
            text_input = MyLineEdit()
            text_input.setFixedSize(120, 30)
            text_input.setFont(QFont("Arial", 16))
            text_input.setStyleSheet(f"color: {color}")
            self.inputs[color] = text_input
            hbox.addWidget(text_input)
            # hbox.addStretch(1)
            layout.addLayout(hbox)

        # layout.addStretch(1)
        # function button
        hbox2 = QHBoxLayout()
        tag_button = QPushButton("tag it")
        tag_button.clicked.connect(self.accept_tag)
        skip_button = QPushButton("skip")
        skip_button.clicked.connect(self.skip_tag)
        stop_button = QPushButton("stop")
        stop_button.clicked.connect(self.reject_tag)
        hbox2.addWidget(tag_button)
        hbox2.addWidget(skip_button)
        hbox2.addWidget(stop_button)
        layout.addLayout(hbox2)

        self.setLayout(layout)

        if self.pred_host:
            threading.Thread(target=self.get_auto_tag).start()

    def reject_tag(self):
        self.reject()

    #
    def skip_tag(self):
        self.done(11)

    def accept_tag(self):
        self.accept()

    def get_tag(self):
        # get user input tag
        return {color: self.inputs[color].text().upper() for color in self.inputs}

    def get_auto_tag(self):
        logger.info(
            f"start to get auto tag, filename: {os.path.basename(self.img_path)}"
        )
        for color in self.inputs:
            pred_label = get_predict_label(self.pred_host, self.img_path, color)
            if pred_label:
                QMetaObject.invokeMethod(
                    self.inputs[color],
                    "setText",
                    Qt.QueuedConnection,
                    Q_ARG(str, pred_label),
                )
        logger.info(f"auto tag finished, filename: {os.path.basename(self.img_path)}")

    def keyPressEvent(self, event):
        # check if press enter
        if event.key() == Qt.Key_Return:
            # if press enter, do the corresponding operation
            self.accept_tag()
        elif event.key() == Qt.Key_Escape:
            self.skip_tag()


class MyLineEdit(QLineEdit):
    def __init__(self, *args, **kwargs):
        super(MyLineEdit, self).__init__(*args, **kwargs)
        self.textChanged.connect(self.on_text_changed)

    def on_text_changed(self, text):
        self.textChanged.disconnect(self.on_text_changed)
        new_text = "".join(
            [
                i.upper() if i.isalpha() and not "\u4e00" <= i <= "\u9fff" else i
                for i in text
            ]
        )
        self.setText(new_text)
        self.textChanged.connect(self.on_text_changed)


class RenameDialog(QDialog):
    def __init__(self, old_path):
        super().__init__()
        self.setWindowTitle("Rename File")
        self.setWindowTitle("Make Tag Window")
        self.setFixedSize(QSize(300, 250))

        layout = QVBoxLayout()
        scale = 2
        self.img_label = QLabel(self)
        self.img_label.setFixedSize(QSize(int(120 * scale), int(50 * scale)))
        self.img_label.setScaledContents(True)
        self.img_label.setPixmap(QPixmap(old_path))
        layout.addWidget(self.img_label)

        self.input = QLineEdit(os.path.basename(old_path))
        layout.addWidget(self.input)
        layout.addStretch(1)
        hbox = QHBoxLayout()
        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.accept)
        hbox.addWidget(self.ok_button)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        hbox.addWidget(self.cancel_button)
        layout.addLayout(hbox)
        self.setLayout(layout)

    def get_new_name(self):
        return self.input.text()

    def keyPressEvent(self, event):
        # check if press enter
        if event.key() == Qt.Key_Return:
            # if press enter, do the corresponding operation
            self.accept()


class TagWindow(QWidget):
    def __init__(
        self,
        dataset_dir: pathlib.Path,
        target_dir: pathlib.Path,
        test_ratio: float = 0.4,
        pred_host: str = "",
        multi_tag: bool = False,
    ):
        super().__init__()

        # 创建一个新的MplCanvas实例
        self.canvas = MplCanvas(self, width=3, height=2, dpi=100)
        self.canvas.axes.axis("off")  # close the axis

        self.pred_host = pred_host
        self.multi_tag = multi_tag
        self.dataset_dir = dataset_dir
        self.target_dir = target_dir
        self.test_ratio = test_ratio

        self.file_paths = glob(f"{self.dataset_dir}/*.png")
        random.shuffle(self.file_paths)
        self.filename_map = {
            os.path.basename(file_path): file_path for file_path in self.file_paths
        }

        processed_files = [
            os.path.basename(file) for file in glob(f"{self.target_dir}/*.png")
        ]
        self.processed_num = len(processed_files)
        self.total_num = len(self.file_paths) + self.processed_num

        # count the number of each tag
        self.yellow_cnt = sum(1 for key in processed_files if key.startswith("yellow"))
        self.blue_cnt = sum(1 for key in processed_files if key.startswith("blue"))
        self.black_cnt = sum(1 for key in processed_files if key.startswith("black"))
        self.red_cnt = sum(1 for key in processed_files if key.startswith("red"))

        # ========UI Layout============
        self.setWindowTitle("Make Tag Window")
        self.setFixedSize(QSize(800, 600))

        # create two list
        self.unprocessed_list = QListWidget()
        self.processed_list = QListWidget()
        # add all file name to the list
        self.unprocessed_list.addItems(list(self.filename_map.keys()))

        self.unprocessed_label = QLabel(f"Unprocessed:{self.unprocessed_list.count()}")
        self.processed_label = QLabel(
            f"Processed:{self.processed_list.count()}(Total {self.processed_num})"
        )

        # function button
        self.multi_tag_btn = QPushButton("Start")
        self.multi_tag_btn.clicked.connect(self.process_multi_tag)

        quit_button = QPushButton("Exit")
        quit_button.clicked.connect(QApplication.instance().quit)

        # layout
        layout = QHBoxLayout()
        v1 = QVBoxLayout()
        v1.addWidget(self.unprocessed_label)
        v1.addWidget(self.unprocessed_list)
        layout.addLayout(v1)

        v2 = QVBoxLayout()

        v2.addStretch(2)
        v2.addWidget(self.multi_tag_btn)
        v2.addStretch(1)
        v2.addWidget(quit_button)
        v2.addStretch(1)
        layout.addLayout(v2)

        v3 = QVBoxLayout()
        v3.addWidget(self.processed_label)
        v3.addWidget(self.processed_list)
        layout.addLayout(v3)

        # progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setValue(self.processed_num)
        self.progress_bar.setMaximum(self.total_num)
        self.progress_bar.setFormat("Progress: %p%")

        vbox = QVBoxLayout()
        vbox.addWidget(self.progress_bar)
        # 将这个canvas添加到你的布局中
        vbox.addWidget(self.canvas)

        vbox.addLayout(layout)

        self.setLayout(vbox)
        self.update_figure()

        # 在 TagWindow 类的 __init__ 方法中添加以下代码来连接信号和槽
        self.processed_list.itemDoubleClicked.connect(self.rename_file)

    def rename_file(self, item):
        old_name = item.text()
        old_path = str(self.target_dir / old_name)
        if not os.path.exists(old_path):
            QMessageBox.warning(self, "Warning", "File does not exist", QMessageBox.Yes)
            return
        dialog = RenameDialog(old_path)
        if dialog.exec() == QDialog.Accepted:
            new_name = dialog.get_new_name()
            new_path = os.path.join(os.path.dirname(old_path), new_name)
            os.rename(old_path, new_path)
            self.filename_map[new_name] = new_path
            item.setText(new_name)

    @staticmethod
    def tag_filename(filename, tag, idx, channel):
        suffix = filename.split(".")[-1]
        return f"{channel}-{tag}-{idx}.{suffix}"

    def update_figure(self):
        # 在这个函数中，你可以更新你的图形

        sizes = [
            self.yellow_cnt + 1,
            self.blue_cnt + 1,
            self.black_cnt + 1,
            self.red_cnt + 1,
        ]
        labels = ["Yellow", "Blue", "Black", "Red"]

        self.canvas.axes.clear()
        wedges, texts, autotexts = self.canvas.axes.pie(
            sizes, labels=labels, autopct="%1.1f%%", startangle=90
        )
        self.canvas.axes.axis(
            "equal"
        )  # Equal aspect ratio ensures that pie is drawn as a circle.

        # add annotation
        # 添加每种颜色标签的具体数量
        legend_labels = [f"{label}: {size}" for label, size in zip(labels, sizes)]
        self.canvas.axes.legend(
            wedges,
            legend_labels,
            title="Colors",
            loc="center left",
            bbox_to_anchor=(0.8, 0, 0.5, 1),
        )

        self.canvas.draw()

    def save_image(self, filename, tag, index, channel=None):
        if channel is None:
            channel = filename.split("-")[0]
        tag_filename = self.tag_filename(filename, tag, index, channel)
        target_file_path = self.target_dir / tag_filename
        logger.info(
            f"move image from {os.path.basename(self.filename_map[filename])} to {target_file_path.name}"
        )
        shutil.copy(self.filename_map[filename], target_file_path)
        json_file = "train.json" if random.random() > self.test_ratio else "test.json"
        json_file_path = self.target_dir.parent / json_file
        json_list = []
        if json_file_path.exists():
            with open(json_file_path, "r", encoding="utf-8") as f:
                json_list = json.load(f)
        record = {"path": self.target_dir.name + "/" + tag_filename, channel: tag}
        json_list.append(record)
        logger.info(f"json record: {record}")
        with open(json_file_path, "w", encoding="utf-8") as f:
            json.dump(json_list, f, ensure_ascii=False, indent=4)
        return tag_filename

    def select_item_based_on_ratio(self):
        # calculate the ratio of each channel
        total = self.yellow_cnt + self.blue_cnt + self.black_cnt + self.red_cnt
        total = max(total, 1)
        channels = ["yellow", "blue", "black", "red"]
        ratios = [
            self.yellow_cnt / total,
            self.blue_cnt / total,
            self.black_cnt / total,
            self.red_cnt / total,
        ]
        acc_ratios = [ratios[i - 1] + ratios[i] for i in range(1, 4)]
        channel = channels[-1]
        for i in range(4):
            if acc_ratios[-1] * random.random() <= acc_ratios[i]:
                channel = channels[i]
                break
        assert channel is not None, "channel is None"
        # select item based on channel randomly
        yellow_items = [
            self.unprocessed_list.item(i)
            for i in range(self.unprocessed_list.count())
            if self.unprocessed_list.item(i).text().startswith("yellow")
        ]
        blue_items = [
            self.unprocessed_list.item(i)
            for i in range(self.unprocessed_list.count())
            if self.unprocessed_list.item(i).text().startswith("blue")
        ]
        black_items = [
            self.unprocessed_list.item(i)
            for i in range(self.unprocessed_list.count())
            if self.unprocessed_list.item(i).text().startswith("black")
        ]
        red_items = [
            self.unprocessed_list.item(i)
            for i in range(self.unprocessed_list.count())
            if self.unprocessed_list.item(i).text().startswith("red")
        ]
        if channel == "yellow" and yellow_items:
            selected_item = random.choice(yellow_items)
            self.yellow_cnt += 1
            return selected_item
        elif channel == "yellow":
            channel = random.choice(["blue", "black", "red"])
        if channel == "blue" and blue_items:
            selected_item = random.choice(blue_items)
            self.blue_cnt += 1
            return selected_item
        elif channel == "blue":
            channel = random.choice(["black", "red"])
        if channel == "black" and black_items:
            selected_item = random.choice(black_items)
            self.black_cnt += 1
            return selected_item
        elif channel == "black":
            channel = "red"
        if channel == "red" and red_items:
            selected_item = random.choice(red_items)
            self.red_cnt += 1
            return selected_item
        else:
            raise ValueError(
                f"unknown channel: {channel}, {len(yellow_items)}, {len(blue_items)}, {len(black_items)}, {len(red_items)}"
            )

    def process_multi_tag(self):
        self.multi_tag = True
        self.process_file()

    def process_single_tag(self):
        self.multi_tag = False
        self.process_file()

    def process_file(self):
        # if all file has been processed
        for i in range(self.processed_num, self.total_num):
            # select one item from the list
            file_item = self.select_item_based_on_ratio()
            filename = file_item.text()
            self.unprocessed_list.setCurrentItem(file_item)

            # process the file
            dialog = TagDialog(self.filename_map[filename], self.pred_host)
            result = dialog.exec()
            # remove the item from the list
            row = self.unprocessed_list.row(file_item)
            self.unprocessed_list.takeItem(row)
            self.unprocessed_list.setCurrentItem(self.unprocessed_list.item(0))

            if result == QDialog.Accepted:
                self.unprocessed_label.setText(
                    f"Unprocessed:{self.unprocessed_list.count()}"
                )
                tag = dialog.get_tag()
                for color in tag:
                    if tag[color] == "":
                        continue
                    tag_filename = self.save_image(filename, tag[color], i, color)
                    self.processed_list.addItem(tag_filename)
                self.processed_label.setText(
                    f"Processed:{self.processed_list.count()}"
                    f"(Total{self.processed_list.count() + self.processed_num})"
                )
                self.progress_bar.setValue(i)
                self.update_figure()  # update the color ratio figure
                # remove the file from the origin folder
                os.remove(self.filename_map[filename])
            elif result == QDialog.Rejected:
                break
            elif result == 11:
                self.unprocessed_label.setText(
                    f"Unprocessed:{self.unprocessed_list.count()}"
                )
                self.progress_bar.setValue(i)
                continue
            else:
                raise ValueError("unknown result")


dark_stylesheet = """
    QWidget {
        background-color: #2b2b2b;
        color: #b1b1b1;
    }
    QLineEdit {
        background-color: #353535;
        color: #b1b1b1;
    }
    QPushButton {
        background-color: #353535;
        color: #b1b1b1;
    }
    QListWidget { 
        background-color: #353535;
        color: #b1b1b1;
    }
    QProgressBar {
        background-color: #353535;
        color: #b1b1b1;
    }
"""


def main(dataset_dir: str, output_dir: str, test_ratio: float, pred_host: str):
    app = QApplication([])
    app.setStyleSheet(dark_stylesheet)
    window = TagWindow(
        pathlib.Path(dataset_dir), pathlib.Path(output_dir), test_ratio, pred_host
    )
    window.show()
    app.exec()
