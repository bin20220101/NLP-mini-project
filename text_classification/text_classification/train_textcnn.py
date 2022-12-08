#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 从 argparse 和 transformers 中导入所需模块
import argparse

from transformers import TrainingArguments, Trainer
from transformers import AutoTokenizer, AutoConfig, AutoModel

# 以下是自己的模块
from utils import compute_metrics  # utils.py 导入 compute_metrics 函数, 用于计算指标
from data_loaders import NewsLoader  # 从 data_loaders 导入 NewsLoader 类, 用于加载数据及预处理等
from models import TextCNNConfig, TextCNN  # 从 models 导入 TextCNNConfig, TextCNN

# 在 transformers 中注册自定义模型
AutoConfig.register("TextCNN", TextCNNConfig)
AutoModel.register(TextCNNConfig, TextCNN)


def main_task(args, train_args):
    # 加载 bert-base-uncased 模型的 tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    # 加载数据集
    news_loader = NewsLoader(
        data_file='News_Category_Dataset.json', 
        tokenizer=tokenizer, 
        test_size=args.test_size,
        random_seed=args.random_seed
    )
    dataset = news_loader.get_dataset()

    # 初始化模型配置
    config = TextCNNConfig(
        vocab_size=tokenizer.vocab_size,
        num_classes=len(news_loader.target_names)
    )
    # 初始化模型
    model = AutoModel.from_config(config)
    # 初始化 Trainer
    trainer = Trainer(
        model,
        train_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        compute_metrics=compute_metrics
    )

    train_result = trainer.train()  # 训练
    trainer.save_model(args.output_dir)  # 保存最优模型

    train_metrics = train_result.metrics
    trainer.save_metrics(split='train', metrics=train_metrics)  # 保存训练集的指标
    eval_metrics = trainer.evaluate()
    trainer.save_metrics(split='eval', metrics=eval_metrics)  # 保存验证集的指标


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='News Classification')
    parser.add_argument('--output_dir', type=str, default='textcnn', help='模型保存路径')
    parser.add_argument('--test_size', type=float, default=0.2, help='测试集比例')
    parser.add_argument('--random_seed', type=int, default=42, help='随机数种子')
    args = parser.parse_args()

    # 训练参数
    train_args = TrainingArguments(
        args.output_dir,  # 模型保存位置
        evaluation_strategy="steps",  # 验证方式, 按照 steps 来验证
        save_strategy="steps",  # 保存方式
        logging_steps=1000,  # 每 1000 steps 时, 记录 log
        save_steps=1000,  # 每 1000 steps 时, 保存模型
        learning_rate=1e-3,  # 学习率
        per_device_train_batch_size=64,  # 训练时的 batch_size
        per_device_eval_batch_size=64,  # 验证时的 batch_size
        num_train_epochs=100,  # epoch, 训练的轮数
        weight_decay=0.01,  # 权重衰减
        load_best_model_at_end=True,  # 训练结束后加载最优模型
        overwrite_output_dir=True,  # 覆盖模型输出, 防止保存大量 checkpoint
        save_total_limit=1,  # 只保存最优的模型
        metric_for_best_model='eval_f1-score',  # 最优模型的评价指标
    )
    main_task(args, train_args)

