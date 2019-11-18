#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author: zhaoyang
@license: (C) Copyright 2001-2019 Python Software Foundation. All rights reserved.
@contact: 1805453683@qq.com
@file: test.py
@time: 2019/11/18 16:05
"""
import bayes_classify.pre_process_method as pre
import bayes_classify.bayes_method as bayes
import numpy as np

print("正在加载词库......")
vocabulary_list_file = "vocabulary_list.txt"
vocabulary_list = pre.load_vocabulary_list(vocabulary_list_file)
print("词库加载完毕\n")

print("正在生成词矩阵......")
test_data_file = "../data/test_data.txt"
sms_words_list, class_category = pre.load_sms_data(test_data_file)
words_matrix = pre.create_words_matrix(vocabulary_list, sms_words_list)
print("词矩阵生成完毕\n")

print("正在加载模型参数......")
p_spam_file = "p_spam.txt"
p_word_spam_file = "p_word_spam.txt"
p_word_nonspam_file = "p_word_nonspam.txt"
p_spam, p_word_spam, p_word_nonspam = bayes.load_probability(p_spam_file, p_word_spam_file, p_word_nonspam_file)
print("模型参数加载完毕\n")

print("正在进行分类......")
class_result = bayes.classify(words_matrix, p_spam, p_word_spam, p_word_nonspam)
print("分类执行完毕\n")

correct_num = 0.0
for i in range(len(class_result)):
    if class_result[i] == class_category[i]:
        correct_num += 1
correct_rate = correct_num / len(class_result)
print("正确率为：", correct_rate)
