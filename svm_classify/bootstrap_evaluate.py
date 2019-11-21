#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author: zhaoyang
@license: (C) Copyright 2001-2019 Python Software Foundation. All rights reserved.
@contact: 1805453683@qq.com
@file: bootstrap_evaluate.py
@time: 2019/11/20 13:01
"""
import svm_classify.pre_process_method as pre
import random
from sklearn import svm
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

times = 20    # 词库中单词在垃圾邮件中出现的总次数 >= 20

print("正在加载并处理数据......")
data_file = "../data/data.txt"
sms_words_list, class_category = pre.load_sms_data(data_file)
spam_words_list = pre.get_spam_words_list(sms_words_list, class_category)
vocabulary_list = pre.create_vocabulary_list(spam_words_list, times)
words_matrix = pre.create_words_matrix(vocabulary_list, sms_words_list)
print("数据处理完成\n")

print("正在使用bootstrap方法对数据进行划分......")
num_lines = len(words_matrix)
num_words = len(words_matrix[0])
index_list = [random.randint(0, num_lines) for _ in range(num_lines)]    # 随机生成num_lines个数
index_list = set(index_list)         # 去重
train_words_matrix = []              # 用于训练的单词矩阵
test_words_matrix = []               # 用于测试的单词矩阵
train_class_category = []            # 用于训练的类别标签
test_class_category = []             # 用于测试的类别标签
for i in range(num_lines):
    if i in index_list:
        train_words_matrix.append(words_matrix[i])
        train_class_category.append(class_category[i])
    else:
        test_words_matrix.append(words_matrix[i])
        test_class_category.append(class_category[i])
print("数据划分完毕\n")
print("训练集样本数量：", len(train_words_matrix), ", 测试集样本数量：", len(test_words_matrix), "\n")

print("正在进行模型训练......")
svc = svm.SVC(gamma='auto')
svc.fit(train_words_matrix, train_class_category)
print("模型训练完成\n")
print('Training accuracy = ', svc.score(train_words_matrix, train_class_category), '\n')

print("正在进行模型评估......")
test_predict = svc.predict(test_words_matrix)
accuracy = svc.score(test_words_matrix, test_class_category)
precision = precision_score(test_class_category, test_predict, average='macro')
recall = recall_score(test_class_category, test_predict, average='macro')
f1 = f1_score(test_class_category, test_predict, average='macro')
print("模型评估完成\n")
print("Test accuracy = ", precision, ", recall = ", recall, ", accuracy = ", accuracy, ", f1 = ", f1)

p_list = svc.decision_function(test_words_matrix)
fpr, tpr, threshold = roc_curve(test_class_category, p_list)      # 计算真阳性率和假阳性率
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc)    # 绘制ROC曲线
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
