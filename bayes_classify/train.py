#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author: zhaoyang
@license: (C) Copyright 2001-2019 Python Software Foundation. All rights reserved.
@contact: 1805453683@qq.com
@file: train.py
@time: 2019/11/18 16:00
"""
import bayes_classify.pre_process_method as pre
import bayes_classify.bayes_method as bayes
import numpy as np

print("正在生成词库......")
train_data_file = "../data/train_data.txt"
sms_words_list, class_category = pre.load_sms_data(train_data_file)
vocabulary_list = pre.create_vocabulary_list(sms_words_list)
vocabulary_list_txt = open("vocabulary_list.txt", 'w')
for i in range(len(vocabulary_list)):
    vocabulary_list_txt.write(vocabulary_list[i] + '\t')
vocabulary_list_txt.flush()
vocabulary_list_txt.close()
print("词库生成完成\n")

print("正在生成词矩阵......")
words_matrix = pre.create_words_matrix(vocabulary_list, sms_words_list)
print("词矩阵生成完成\n")

print("正在计算概率......")
p_spam, p_word_spam, p_word_nonspam = bayes.get_probability(words_matrix, class_category)
np.savetxt("p_word_spam.txt", p_word_spam, delimiter='\t')
np.savetxt("p_word_nonspam.txt", p_word_nonspam, delimiter='\t')
p_spam_txt = open("p_spam.txt", 'w')
p_spam_txt.write(p_spam.__str__())
p_spam_txt.close()
print("概率计算完成\n")
