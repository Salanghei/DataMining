#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author: zhaoyang
@license: (C) Copyright 2001-2019 Python Software Foundation. All rights reserved.
@contact: 1805453683@qq.com
@file: k_fold_evaluate.py
@time: 2019/11/20 13:01
"""
import svm_classify.pre_process_method as pre
from sklearn import svm
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

times = 20    # 词库中单词在垃圾邮件中出现的总次数 >= 20

print("正在加载并处理数据......")
data_file = "../data/data.txt"
sms_words_list, class_category = pre.load_sms_data(data_file)
spam_words_list = pre.get_spam_words_list(sms_words_list, class_category)
vocabulary_list = pre.create_vocabulary_list(spam_words_list, times)
words_matrix = pre.create_words_matrix(vocabulary_list, sms_words_list)
print("数据处理完成\n")

k = 10
fold_len = int(len(words_matrix) / k)    # 每一折包含的样本数
print("使用k-fold方法对模型进行评估，k = ", k, ", fold_len = ", fold_len, "\n")
precision = []
recall = []
accuracy = []
f1 = []
for i in range(k):
    print("正在执行第", i + 1, "次训练及测试......")
    test_words_matrix = words_matrix[i * fold_len: (i + 1) * fold_len]  # 用于测试的单词矩阵
    train_words_matrix = words_matrix[: i * fold_len]  # 用于训练的单词矩阵
    train_words_matrix.extend(words_matrix[(i + 1) * fold_len:])

    test_class_category = class_category[i * fold_len: (i + 1) * fold_len]  # 用于测试的类别标签
    train_class_category = class_category[: i * fold_len]  # 用于训练的类别标签
    train_class_category.extend(class_category[(i + 1) * fold_len:])

    svc = svm.SVC(gamma='auto')
    svc.fit(train_words_matrix, train_class_category)
    test_predict = svc.predict(test_words_matrix)

    accuracy_i = svc.score(test_words_matrix, test_class_category)
    precision_i = precision_score(test_class_category, test_predict, average='macro')
    recall_i = recall_score(test_class_category, test_predict, average='macro')
    f1_i = f1_score(test_class_category, test_predict, average='macro')
    accuracy.append(accuracy_i)
    precision.append(precision_i)
    recall.append(recall_i)
    f1.append(f1_i)
    print("第", i + 1, "次训练及测试完毕")
    print("precision(", i + 1, ") = ", precision_i, ", recall(", i + 1, ") = ", recall_i,
          ", accuracy(", i + 1, ") = ", accuracy_i, ", f1(", i + 1, ") = ", f1_i)

precision_avg = sum(precision) / k
recall_avg = sum(recall) / k
accuracy_avg = sum(accuracy) / k
f1_avg = sum(f1) / k
print("\n模型评估完毕")
print("precision = ", precision_avg, ", recall = ", recall_avg, ", accuracy = ", accuracy_avg, ", f1 = ", f1_avg)
