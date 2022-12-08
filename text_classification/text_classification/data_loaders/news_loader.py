#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 从 json 和 datasets 中导入所需模块
import json

from datasets import Dataset, ClassLabel


class NewsLoader(object):
    def __init__(self, data_file, tokenizer, test_size, random_seed):
        self.data_file = data_file

        dataset, self.label2id = self.load_data()  # 加载数据集
        
        # 将数据标签转换为 ClassLabel 类, 用于后续训练集和测试集的划分
        features = dataset.features.copy()
        target_names = ['' for _ in range(len(self.label2id))]
        for label, _id in self.label2id.items():
            target_names[_id] = label
        features['labels'] = ClassLabel(names=target_names)
        dataset = dataset.cast(features)

        # 用 tokenizer 将文本转换为数字, text to id, 最大长度为 512
        dataset = dataset.map(
            lambda examples: tokenizer(examples["text"], padding='max_length', max_length=512, truncation=True), 
            batched=True
        )
        # 按照标签分布划分训练测试集
        self.dataset = dataset.train_test_split(test_size=test_size, stratify_by_column='labels', seed=random_seed)
        self.target_names = target_names

    def get_dataset(self):
        return self.dataset

    def load_data(self):
        with open(self.data_file, 'r', encoding='utf-8') as fr:
            lines = fr.readlines()
        dataset = {'text': [], 'labels': []}
        label2id = {}  # 将标签转换为数值
        for line in lines:
            data_dict = json.loads(line.strip('\n'))
            headline = data_dict.get('headline', '')  # 新闻标题
            short_description = data_dict.get('short_description', '')  # 新闻的简短描述
            text = headline + short_description  # 标题 + 描述, 作为文本输入
            label = data_dict.get('category', '')  # 新闻类别作为文本标签
            if label not in label2id:  # 如果标签没见过, 记录该标签
                label2id[label] = len(label2id)
            label = label2id[label]  # label to id
            dataset['text'].append(text)
            dataset['labels'].append(label)
        return Dataset.from_dict(dataset), label2id
        