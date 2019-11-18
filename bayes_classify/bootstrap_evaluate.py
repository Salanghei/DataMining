#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author: zhaoyang
@license: (C) Copyright 2001-2019 Python Software Foundation. All rights reserved.
@contact: 1805453683@qq.com
@file: bootstrap_evaluate.py
@time: 2019/11/18 16:55
"""
import bayes_classify.pre_process_method as pre
import bayes_classify.bayes_method as bayes
import random
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

print("正在加载数据，生成词库......")
data_file = "../data/data.txt"
sms_words_list, class_category = pre.load_sms_data(data_file)
vocabulary_list = pre.create_vocabulary_list(sms_words_list)
print("词库生成完毕\n")

print("正在生成词矩阵......")
words_matrix = pre.create_words_matrix(vocabulary_list, sms_words_list)  # 词矩阵
print("词矩阵生成完成\n")

print("正在使用bootstrap方法对数据进行划分......")
num_lines = len(sms_words_list)      # 行数
num_words = len(sms_words_list[0])   # 单词数
index_list = [random.randint(0, num_lines) for _ in range(num_lines)]    # 随机生成n个数
index_list = set(index_list)         # 去重
train_words_matrix = []              # 用于训练的单词矩阵
test_words_matrix = []               # 用于测试的单词矩阵
train_class_category = []            # 用于训练的类别标签
test_class_category = []             # 用于测试的类别标签
for i in range(len(words_matrix)):
    if i in index_list:
        train_words_matrix.append(words_matrix[i])
        train_class_category.append(class_category[i])
    else:
        test_words_matrix.append(words_matrix[i])
        test_class_category.append(class_category[i])
print("数据划分完毕\n")
print("训练集样本数量：", len(train_words_matrix), ", 测试集样本数量：", len(test_words_matrix), "\n")

print("正在计算概率p(s)，p(wi|s)，p(wi|ns)......")
p_spam, p_word_spam, p_word_nonspam = bayes.get_probability(train_words_matrix, train_class_category)
print("概率计算完毕\n")

print("正在进行测试......")
class_result = bayes.classify(test_words_matrix, p_spam, p_word_spam, p_word_nonspam)
correct = 0.0
for i in range(len(class_result)):
    if test_class_category[i] == class_result[i]:
        correct += 1
accuracy = correct / len(class_result)
precision = precision_score(test_class_category, class_result, average='macro')
recall = recall_score(test_class_category, class_result, average='macro')
f1 = f1_score(test_class_category, class_result, average='macro')
print("测试完毕\n")
print("precision = ", precision, ", recall = ", recall, ", accuracy = ", accuracy, ", f1 = ", f1)
