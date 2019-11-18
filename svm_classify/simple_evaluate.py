#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author: zhaoyang
@license: (C) Copyright 2001-2019 Python Software Foundation. All rights reserved.
@contact: 1805453683@qq.com
@file: simple_evaluate.py
@time: 2019/11/18 22:01
"""
import svm_classify.pre_process_method as pre
import numpy as np
from sklearn import svm
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

print("正在加载并处理训练数据......")
data_file = "../data/train_data.txt"
sms_words_list, class_category = pre.load_sms_data(data_file)
spam_words_list = pre.get_spam_words_list(sms_words_list, class_category)
vocabulary_list = pre.create_vocabulary_list(spam_words_list)
words_matrix = pre.create_words_matrix(vocabulary_list, sms_words_list)
print("训练数据处理完成\n")

words_matrix = np.array(words_matrix)        # words_matrix转换为适合svm输入的格式
class_category = np.array(class_category)    # class_category转换为适合svm输入的格式

print("正在进行模型训练......")
svc = svm.SVC(gamma='auto')
svc.fit(words_matrix, class_category)
print("模型训练完成\n")
print('Training accuracy = ', svc.score(words_matrix, class_category), '\n')

print("正在加载并处理测试数据......")
test_file = "../data/test_data.txt"
test_words_list, test_class_category = pre.load_sms_data(test_file)
test_words_matrix = pre.create_words_matrix(vocabulary_list, test_words_list)
print("测试数据处理完成\n")

test_words_matrix = np.array(test_words_matrix)
test_class_category = np.array(test_class_category)

print("正在进行模型评估......")
test_predict = svc.predict(test_words_matrix)
print("模型评估完成\n")

print('Test accuracy = ', svc.score(test_words_matrix, test_class_category))
print('precision = ', precision_score(test_class_category, test_predict, average='macro'))
print('recall = ', recall_score(test_class_category, test_predict, average='macro'))
