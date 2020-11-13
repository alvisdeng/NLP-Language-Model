#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Zimeng Qiu <zimengq@andrew.cmu.edu>

"""
F19 11-411/611 NLP Assignment 3 Task 1
N-gram Language Model Utils
Zimeng Qiu Sep 2019

Define model utils here

You don't need to modify this file.
"""

import operator
from collections import Counter
from functools import reduce
import numpy as np

def read_file(file):
    """
    Read text from file
    :param file: input file path
    :return: text - text content in input file
    """
    with open(file, 'r') as f:
        text = f.readlines()
    return text


def load_dataset(files):
    """
    Load dataset from file list
    :param files:
    :return: data - text dataset
    """
    data = []
    for file in files:
        data.extend(read_file(file))
    return data


def preprocess(corpus):
    """
    Extremely simple preprocessing
    You can not suppose use a preprocessor this simple in real world
    :param corpus: input text corpus
    :return: tokens - preprocessed text
    """
    tokens = []
    for line in corpus:
        tokens.append([tok.lower() for tok in line.split()])
    return tokens

def find_infrequent_words(corpus,min_freq):
    corpus_1d = reduce(operator.add,corpus)
    infrequent_words = []

    for pair in Counter(corpus_1d).most_common():
        if pair[1] < min_freq:
            infrequent_words.append(pair[0])
    
    return infrequent_words

def replace_infrequent_words(corpus,words):
    for i in range(len(corpus)):
        for j in range(len(corpus[i])):
            if corpus[i][j] in words:
                corpus[i][j] = 'UNK'

def get_vocabulary(corpus):
    corpus_1d = reduce(operator.add,corpus)
    vocabulary = set(corpus_1d)
    V = len(vocabulary)
    N = len(corpus_1d)

    return corpus_1d,list(vocabulary),V,N

def get_counter(corpus_1d):
    return Counter(corpus_1d)

def get_word_mappings(vocabulary):
    word_to_idx = {}
    idx_to_word = {}

    for idx, word in enumerate(vocabulary):
        word_to_idx[word] = idx
        idx_to_word[idx] = word
    
    return word_to_idx,idx_to_word

def get_uniform_tables(V):
    frequency_table = np.ones((1,V),dtype=np.float64)
    probability_table = frequency_table/V
    return probability_table

def get_unigram_tables(V,N,counter_1gram,word_to_idx):
    probability_table = np.zeros((1,V),dtype=np.float64)
    # update table
    for pair in counter_1gram.most_common():
        tok = pair[0]
        freq = pair[1]
        probability_table[0][word_to_idx[tok]] = freq/N
    return probability_table

def get_bigram_tables(V,counter_1gram,counter_2gram,word_to_idx,idx_to_word):
    # frequency_table = np.zeros((V,V),dtype=np.float64)
    probability_table = np.zeros((V,V),dtype=np.float64)
    # update table
    for pair in counter_2gram.most_common():
        idx1 = word_to_idx[pair[0][0]]
        idx2 = word_to_idx[pair[0][1]]
        freq = pair[1]
        
        probability_table[idx1][idx2] = (freq+1)/(counter_1gram[pair[0][0]])
        # frequency_table[idx1][idx2] = freq
    
    # for i in range(V):
    #     for j in range(V):
    #         frequency_table[i][j] = (frequency_table[i][j]+1)*counter_1gram[idx_to_word[i]]/(counter_1gram[idx_to_word[i]]+V)
    #         probability_table[i][j] = frequency_table[i][j]/counter_1gram[idx_to_word[i]]

    return probability_table

def get_trigram_tables(V,counter_2gram,counter_3gram,word_to_idx):
    probability_table = {}
    # update table
    for pair in counter_3gram.most_common():
        word1 = pair[0][0]
        word2 = pair[0][1]
        word3 = pair[0][2]

        freq = pair[1]
        probability_table[(word1,word2,word3)] = (freq+1)/(counter_2gram[(pair[0][0],pair[0][1])])
    return probability_table