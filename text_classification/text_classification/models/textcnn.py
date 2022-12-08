#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 从 torch 和 transformers 中导入所需模块
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import PretrainedConfig, PreTrainedModel


# TextCNN 的模型配置类
class TextCNNConfig(PretrainedConfig):
    model_type = "TextCNN"
    def __init__(
            self,
            vocab_size: int = 1000,  # 词表大小
            embed_dim: int = 300,  # embedding 维数
            num_classes: int = 2,  # 模型输出的类别
            kernel_sizes: tuple = (2, 3, 4, 5, 6),  # 卷积核大小
            kernel_nums: tuple = (128, 128, 128, 128, 128),  # 不同卷积核对应的数量
            dropout: float = 0.5,  # dropout
            **kwargs
    ):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.kernel_sizes = kernel_sizes
        self.kernel_nums = kernel_nums
        self.dropout = dropout
        super().__init__(**kwargs)


class TextCNN(PreTrainedModel):
    config_class = TextCNNConfig
    def __init__(self, config):
        super().__init__(config)
        # embedding 层
        self.embed = nn.Embedding(config.vocab_size, config.embed_dim, padding_idx=0)
        # 卷积层
        self.convs = nn.ModuleList([
            nn.Conv2d(1, kernel_num, (kernel_size, config.embed_dim))
            for kernel_num, kernel_size in zip(config.kernel_nums, config.kernel_sizes)
        ])
        # dropout 层
        self.dropout = nn.Dropout(config.dropout)
        # 全连接层
        self.fc = nn.Linear(sum(config.kernel_nums), config.num_classes)
        # loss 函数
        self.loss_fn = torch.nn.CrossEntropyLoss()
        
    def forward(self, input_ids, labels=None):
        x = self.embed(input_ids)  # (batch_size, token_num, embed_dim)
        x = x.unsqueeze(1)  # (batch_size, 1, token_num, embed_dim)
        convs = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        # [(batch_size, kernel_num, token_num) * len(kernel_sizes)]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in convs]
        # [(batch_size, kernel_num) * len(kernel_sizes)]
        x = torch.cat(x, 1)  # (batch_size, kernel_num * len(kernel_sizes))
        x = self.dropout(x)  # (batch_size, kernel_num * len(kernel_sizes))
        logits = self.fc(x)  # (batch_size, num_classes)

        if labels is not None:  # 如果传入 labels, 则计算 loss
            loss = self.loss_fn(logits, labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}
