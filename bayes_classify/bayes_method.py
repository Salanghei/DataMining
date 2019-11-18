#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author: zhaoyang
@license: (C) Copyright 2001-2019 Python Software Foundation. All rights reserved.
@contact: 1805453683@qq.com
@file: bayes_method.py
@time: 2019/11/18 14:32
"""
import numpy as np


def get_probability(words_matrix, class_category):
    """
    计算垃圾邮件的先验概率p(s)，以及每个单词在垃圾邮件中和非垃圾邮件中出现的概率p(wi|s)和p(wi|ns)
    :param words_matrix: 词矩阵
    :param class_category: 类别标签
    :return: p(s)，p(wi|s)，p(wi|ns)
    """
    num_lines = len(words_matrix)        # 行数，即邮件数
    num_words = len(words_matrix[0])     # 单词总数

    p_spam = sum(class_category) / float(num_lines)    # 垃圾邮件的先验概率p(s)

    num_words_in_spam = np.ones(num_words)        # 统计垃圾邮件中每个单词出现的次数，初始化为1
    num_words_in_nonspam = np.ones(num_words)     # 统计非垃圾邮件中每个单词出现的次数，初始化为1
    num_class = 2.0                               # 类别数量，即垃圾邮件、非垃圾邮件

    sum_words_in_spam = num_class                 # 垃圾邮件中出现的单词的总个数
    sum_words_in_nonspam = num_class              # 非垃圾邮件中出现的单词的总个数

    for i in range(num_lines):
        if class_category[i] == 1:    # 如果是垃圾邮件
            num_words_in_spam += words_matrix[i]
            sum_words_in_spam += sum(words_matrix[i])
        else:                         # 如果是非垃圾邮件
            num_words_in_nonspam += words_matrix[i]
            sum_words_in_nonspam += sum(words_matrix[i])

    p_word_spam = np.log(num_words_in_spam / sum_words_in_spam)            # 垃圾邮件中每个单词出现的概率，即p(wi|s)
    p_word_nonspam = np.log(num_words_in_nonspam / sum_words_in_nonspam)   # 非垃圾邮件中每个单词出现的概率，即p(wi|ns)
    return p_spam, p_word_spam, p_word_nonspam


def load_probability(p_spam_file, p_word_spam_file, p_word_nonspam_file):
    """
    加载模型参数
    :param p_spam_file: p(s)的文件名
    :param p_word_spam_file: p(wi|s)的文件名
    :param p_word_nonspam_file: p(wi|ns)的文件名
    :return: p(s)，p(wi|s)，p(wi|ns)
    """
    p_word_spam = np.loadtxt(p_word_spam_file, delimiter='\t')
    p_word_nonspam = np.loadtxt(p_word_nonspam_file, delimiter='\t')
    p_spam_txt = open(p_spam_file, 'r', encoding='UTF-8')
    p_spam = float(p_spam_txt.readline().strip())
    p_spam_txt.close()
    return p_spam, p_word_spam, p_word_nonspam


def classify(test_words_matrix, p_spam, p_word_spam, p_word_nonspam):
    """
    计算联合概率，并进行分类
    :param test_words_matrix: 由测试集得到的词矩阵
    :param p_spam: 垃圾邮件的先验概率p(s)
    :param p_word_spam: 单词在垃圾邮件中出现的概率p(wi|s)
    :param p_word_nonspam: 单词在非垃圾邮件中出现的概率p(wi|ns)
    :return: 分类结果
    """
    p_test_spam_list = sum(test_words_matrix * p_word_spam) + np.log(p_spam)
    p_test_nonspam_list = sum(test_words_matrix * p_word_nonspam) + np.log(1 - p_spam)
    class_result = []
    for i in range(len(test_words_matrix)):
        if p_test_spam_list[i] > p_test_nonspam_list[i]:
            class_result.append(1)
        else:
            class_result.append(0)
    return class_result

