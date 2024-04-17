"""
给图片打标签
"""
import os
import pathlib
import shutil
from glob import glob
from queue import Queue

import click
import requests
from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (QApplication, QInputDialog, QLabel, QLineEdit,
                               QMainWindow, QPushButton, QVBoxLayout, QWidget)

root_dir = pathlib.Path(__file__).parent.parent.parent
cached_label = Queue()


def get_predict_label(img_path: str):
    res = requests.post('http://localhost:8000/captcha/predict', files={"files": open(img_path, 'rb')})
    return res.json()["data"][0]["label"]


class TagWindow(QMainWindow):
    def __init__(self, files_dir, target_dir):
        super().__init__()
        self.files = glob(f"{files_dir}/*.png")
        self.target_dir = pathlib.Path(target_dir)
        os.makedirs(self.target_dir, exist_ok=True)
        self.exist_count = len(glob(f"{target_dir}/*.png"))
        self.img_iter = self.get_img_iter()

        # 布局
        self.setWindowTitle('Make Tag Window')
        self.setFixedSize(QSize(800, 600))

        self.label = QLabel('IMAGE HERE')
        self.label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        scale = 2
        self.label.setFixedSize(QSize(int(120 * scale), int(50 * scale)))
        self.label.setScaledContents(True)
        btn = QPushButton('START')
        btn.clicked.connect(self.next_img)
        vbox = QVBoxLayout()
        vbox.addWidget(self.label)
        vbox.addWidget(btn)
        widget = QWidget()
        widget.setLayout(vbox)
        self.setCentralWidget(widget)

    def get_img_iter(self):
        for i, filepath in enumerate(self.files):
            label = get_predict_label(filepath)
            yield i + 1, label, filepath

    def next_img(self):
        """recur image path"""
        try:
            idx, auto_label, file_path = next(self.img_iter)
            channel = os.path.basename(file_path).split('-')[0]
            self.label.setPixmap(QPixmap(file_path))
            title = f"[{idx}/{len(self.files)}]"
            tip = f"Input {channel} words in captcha img:(predict: {auto_label})"
            checked_label, ok = QInputDialog.getText(self, title, tip, QLineEdit.Normal, auto_label)
            if ok and checked_label:  # click OK
                save_path = self.target_dir / f"{channel}-{checked_label}-{self.exist_count + idx}.png"
                shutil.move(file_path, save_path)
            elif not ok:
                os.remove(file_path)
            elif checked_label == "":
                self.close()
                return
            self.next_img()
        except StopIteration:
            return


def manual_tag(files_dir, target_dir):
    """make tag manually"""
    app = QApplication()
    win = TagWindow(files_dir, target_dir)
    win.show()
    app.exec_()


@click.command()
@click.option('--dataset-dir', type=str, default='origin')
@click.option('--output-dir', type=str, default='labeled')
def main(dataset_dir, output_dir):
    dataset_dir = root_dir / 'dataset' / dataset_dir
    save_dir = root_dir / 'dataset' / output_dir
    manual_tag(dataset_dir, save_dir)


if __name__ == '__main__':
    main()
