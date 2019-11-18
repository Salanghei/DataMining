#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author: zhaoyang
@license: (C) Copyright 2001-2019 Python Software Foundation. All rights reserved.
@contact: 1805453683@qq.com
@file: simple_evaluate.py
@time: 2019/11/18 16:00
"""
import bayes_classify.pre_process_method as pre
import bayes_classify.bayes_method as bayes
import numpy as np
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

print("正在生成词库......")
train_data_file = "../data/train_data.txt"
sms_words_list, class_category = pre.load_sms_data(train_data_file)
vocabulary_list = pre.create_vocabulary_list(sms_words_list)
vocabulary_list_txt = open("vocabulary_list.txt", 'w')
for i in range(len(vocabulary_list)):
    vocabulary_list_txt.write(vocabulary_list[i] + '\t')
vocabulary_list_txt.flush()
vocabulary_list_txt.close()
print("词库生成完毕\n")

print("正在生成训练集词矩阵......")
words_matrix = pre.create_words_matrix(vocabulary_list, sms_words_list)
print("训练集词矩阵生成完毕\n")

print("正在计算概率......")
p_spam, p_word_spam, p_word_nonspam = bayes.get_probability(words_matrix, class_category)
np.savetxt("p_word_spam.txt", p_word_spam, delimiter='\t')
np.savetxt("p_word_nonspam.txt", p_word_nonspam, delimiter='\t')
p_spam_txt = open("p_spam.txt", 'w')
p_spam_txt.write(p_spam.__str__())
p_spam_txt.close()
print("概率计算完毕\n")

print("正在生成测试集词矩阵......")
test_data_file = "../data/test_data.txt"
test_words_list, test_class_category = pre.load_sms_data(test_data_file)
test_words_matrix = pre.create_words_matrix(vocabulary_list, test_words_list)
print("测试集词矩阵生成完毕\n")

print("正在执行测试......")
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
print("accuracy = ", accuracy, ", precision = ", precision, ", recall = ", recall, ", f1 = ", f1)
