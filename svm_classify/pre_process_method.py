#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author: zhaoyang
@license: (C) Copyright 2001-2019 Python Software Foundation. All rights reserved.
@contact: 1805453683@qq.com
@file: pre_process_method.py
@time: 2019/11/18 20:09
"""
import re
import pandas as pd


def text_parser(text):
    """
    对文本的行进行处理，去除空字符，并统一小写
    :param text: 行文本
    :return: 行单词列表
    """
    reg_ex = re.compile(r'[^a-zA-Z]|\d')    # 去掉非字母或数字，只留下单词
    words = reg_ex.split(text)
    words = [word.lower() for word in words if len(word) > 0]    # 转换为小写
    return words


def load_sms_data(file_name):
    """
    加载文本数据，并对每行文本进行单词切分
    :param file_name: 文本数据的文件名
    :return：切分后的文本单词，类别标签
    """
    class_category = []    # 类别标签
    sms_words_list = []     # 文本单词，包含所有行，每行为一个单词列表
    f = open(file_name, 'r', encoding='UTF-8')
    for line in f.readlines():
        line_data = line.strip().split("\t")    # strip删除行两侧的空白字符，split按照空白字符对文本进行分割
        if line_data[0] == "ham":
            class_category.append(0)
        elif line_data[0] == "spam":
            class_category.append(1)
        words = text_parser(line_data[1])
        sms_words_list.append(words)
    f.close()
    return sms_words_list, class_category


def get_spam_words_list(sms_words_list, class_category):
    """
    获取垃圾邮件的单词
    :param sms_words_list: 切分后的文本单词
    :param class_category: 类别标签
    :return: 垃圾邮件的单词
    """
    spam_words_list = []
    for i in range(len(class_category)):
        if class_category[i] == 1:
            spam_words_list.append(sms_words_list[i])
    return spam_words_list


def create_vocabulary_list(spam_words_list):
    """
    创建词库，词库中单词为在垃圾邮件中出现5次以上的单词
    :param spam_words_list: 垃圾邮件中的单词
    :return: 词库
    """
    spam_words_list = pd.DataFrame(spam_words_list)
    num_spam_words = spam_words_list.apply(pd.value_counts).apply(lambda x: x.sum(), axis=1)
    vocabulary_list = []
    for index, value in num_spam_words.iteritems():
        if value >= 5:
            vocabulary_list.append(index)
    return vocabulary_list


def create_words_vector(vocabulary_list, sms_words):
    """
    构建行的词向量
    :param vocabulary_list: 词库
    :param sms_words: 行单词列表
    :return: 行的词向量
    """
    sms_words_vector = [0] * len(vocabulary_list)
    for sms_word in sms_words:
        if sms_word in vocabulary_list:
            sms_words_vector[vocabulary_list.index(sms_word)] = 1
    return sms_words_vector


def create_words_matrix(vocabulary_list, sms_words_list):
    """
    将切分后的文本单词转换为词矩阵
    :param vocabulary_list: 词库
    :param sms_words_list: 切分后的文本单词
    :return: 词矩阵
    """
    words_matrix = []
    for i in range(len(sms_words_list)):
        words_vector = create_words_vector(vocabulary_list, sms_words_list[i])
        words_matrix.append(words_vector)
    return words_matrix
