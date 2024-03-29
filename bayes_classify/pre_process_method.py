#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author: zhaoyang
@license: (C) Copyright 2001-2019 Python Software Foundation. All rights reserved.
@contact: 1805453683@qq.com
@file: pre_process_method.py
@time: 2019/11/18 13:16
"""
import re


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


def create_vocabulary_list(sms_words_list):
    """
    创建词库，词库中的单词不重复
    :param sms_words_list: 文本行单词列表
    :return: 词库
    """
    vocabulary_list = set([])
    for sms_words in sms_words_list:
        vocabulary_list = vocabulary_list | set(sms_words)
    vocabulary_list = list(vocabulary_list)
    return vocabulary_list


def load_vocabulary_list(file_name):
    """
    从文本文件中加载词库
    :param file_name: 词库文件的文件名
    :return: 词库
    """
    f = open(file_name, 'r', encoding='UTF-8')
    vocabulary_list = f.readline().strip().split('\t')
    f.close()
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
            sms_words_vector[vocabulary_list.index(sms_word)] += 1
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

