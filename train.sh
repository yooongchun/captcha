#! /bin/bash

PROJ_DIR=$(dirname $(readlink -f "$0"))

# 指定训练参数
# 默认情况下可以不指定数据集路径，采用自动生成策略
DATASET_DIR=dataset/split-labeled
# 自动生成时数据集大小（实际上每个epoch生成的数据集都是不同的）
AUTO_NUM=1000000
# 可以指定预训练模型来实现增量训练
PRETRAINED=
# 评测评率和保存频率
EVAL_FREQ=10
SAVE_FREQ=5
# 生成的验证码最大长度
MAX_LEN=6
# 模型是识别具体的某个颜色还是识别所有颜色
CHANNEL=text
# 是否采用简单模式，简单模式下验证码没有汉字
SIMPLE_MODE=
# 采用哪个模型作为特征提取网络
MODEL="custom"
# 批次大小
BATCH_SIZE=32
# 训练轮数
NUM_EPOCH=100
# 学习率
LR=0.001
# 多进程读取数据
NUM_WORKERS=20
# 是否使用wandb来可视化(指定online或offline，不用wandb可视化则不填写)
WANDB_MODE=online
WANDB_NAME=0421-t1
# 用几卡训练
GPUS=

if [[ -n $DATASET_DIR ]]; then
    DATASET_DIR="--dataset-dir $DATASET_DIR"
fi

if [[ -n $PRETRAINED ]]; then
    PRETRAINED="--pretrained $PRETRAINED"
fi

if [[ -n $SIMPLE_MODE ]]; then
    SIMPLE_MODE="--simple-mode"
fi

if [[ -n $WANDB_MODE ]]; then
    WANDB_MODE="--wandb-mode $WANDB_MODE"
fi

if [[ -n $WANDB_NAME ]]; then
    WANDB_NAME="--wandb-name $WANDB_NAME"
fi

if [[ ${#GPUS} -gt 1 ]]; then
    DISTRIBUTED_LAUNCH="-m paddle.distributed.launch"
fi
# 运行train.py文件
CUDA_VISIBLE_DEVICES=$GPUS python $DISTRIBUTED_LAUNCH src.train.train \
                --auto-num $AUTO_NUM \
                --eval-freq $EVAL_FREQ \
                --save-freq $SAVE_FREQ \
                --max-len $MAX_LEN \
                --channel $CHANNEL \
                --model $MODEL \
                --batch-size $BATCH_SIZE \
                --num-epoch $NUM_EPOCH \
                --lr $LR \
                --num-workers $NUM_WORKERS \
                $WANDB_NAME $WANDB_MODE $SIMPLE_MODE $DATASET_DIR $PRETRAINED
