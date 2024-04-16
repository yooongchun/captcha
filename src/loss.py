#! -*- coding: utf-8 -*-

import paddle


class CTCLoss(paddle.nn.Layer):
    def __init__(self, num_classes):
        super().__init__()
        self.ctc_loss = paddle.nn.CTCLoss(blank=num_classes)

    def forward(self, inputs: paddle.Tensor, labels: paddle.Tensor):
        inputs = inputs.transpose(perm=[1, 0, 2])
        in_lengths = paddle.full(shape=[inputs.shape[1]], fill_value=inputs.shape[0], dtype="int64")
        label_lengths = paddle.sum(labels != -1, axis=-1, dtype="int64")
        loss = self.ctc_loss(inputs, labels, in_lengths, label_lengths)
        return loss
