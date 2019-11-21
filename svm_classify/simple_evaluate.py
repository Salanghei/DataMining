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
from sklearn import svm
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# 选出最合适的阈值，作为词库中单词在垃圾邮件中出现的最少次数
times_range = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]    # 词库中单词在垃圾邮件中出现的总次数列表

print("正在加载并处理数据......")
data_file = "../data/train_data.txt"
sms_words_list, class_category = pre.load_sms_data(data_file)
spam_words_list = pre.get_spam_words_list(sms_words_list, class_category)

test_file = "../data/test_data.txt"
test_words_list, test_class_category = pre.load_sms_data(test_file)
print("数据处理完成\n")

precision = []
recall = []
accuracy = []
f1 = []
for i in range(len(times_range)):
    print("正在执行第", i + 1, "次训练及测试，词库中单词在垃圾邮件中出现的总次数 >= ", times_range[i], "......")
    vocabulary_list = pre.create_vocabulary_list(spam_words_list, times_range[i])
    words_matrix = pre.create_words_matrix(vocabulary_list, sms_words_list)
    print("词库中包含", len(vocabulary_list), "个单词")

    svc = svm.SVC(gamma='auto')
    svc.fit(words_matrix, class_category)

    # print('Training accuracy = ', svc.score(words_matrix, class_category))
    test_words_matrix = pre.create_words_matrix(vocabulary_list, test_words_list)

    test_predict = svc.predict(test_words_matrix)
    accuracy_i = svc.score(test_words_matrix, test_class_category)
    precision_i = precision_score(test_class_category, test_predict, average='macro')
    recall_i = recall_score(test_class_category, test_predict, average='macro')
    f1_i = f1_score(test_class_category, test_predict, average='macro')
    accuracy.append(accuracy_i)
    precision.append(precision_i)
    recall.append(recall_i)
    f1.append(f1_i)
    print("precision(", i + 1, ") = ", precision_i, ", recall(", i + 1, ") = ", recall_i,
          ", accuracy(", i + 1, ") = ", accuracy_i, ", f1(", i + 1, ") = ", f1_i)

    p_list = svc.decision_function(test_words_matrix)
    fpr, tpr, threshold = roc_curve(test_class_category, p_list)    # 计算真阳性率和假阳性率
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='{} ROC curve (area = {})'.format(i + 1, roc_auc))  # 绘制ROC曲线
print("\n模型评估完毕")

plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

"""
x_axis = times_range
plt.plot(x_axis, accuracy, label='accuracy')
plt.plot(x_axis, precision, label='precision')
plt.plot(x_axis, recall, label='recall')
plt.plot(x_axis, f1, label='f1')
plt.legend()
plt.show()
"""
