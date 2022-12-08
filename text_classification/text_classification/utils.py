#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 从 numpy 和 sklearn 中导入所需模块
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, classification_report


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)  # 概率最大的作为预测类别
    # 计算模型预测的准确率, 召回率, f1-score
    report = classification_report(labels, predictions, output_dict=True)  
    precision = report['weighted avg']['precision']
    recall = report['weighted avg']['recall']
    f1_score = report['weighted avg']['f1-score']
    return {'precision': precision, 'recall': recall, 'f1-score': f1_score}
