# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 19:05:28 2018

@author: DanielYeh
"""

import json
import re
import numpy as np
from os import listdir
from os.path import join
from random import randrange
import random

import string
from collections import defaultdict


############################# Loading Data #############################
def load_data(data_path):
    feats = []
    seq = []
    ids = []
    feat_dir = []
    for file in listdir(data_path):
        idss = '.'.join(file.split('.')[:-1])
        path = join(data_path, file)
        feat_dir.append(path)
        ids.append(idss)
        
    for i_dir in range(0,len(feat_dir)):
        temp_feat = np.load(feat_dir[i_dir])
        feats.append(temp_feat)
        seq.append(feats[i_dir].shape[0])
    
            
    feats = np.array(feats, dtype=np.float32)
    seq = np.array(seq, dtype=np.int32)
    
    return feats, seq, ids


def vocab_process(label_train_path):

    label_dict = {}

    with open(label_train_path, 'r', encoding='utf-8') as f:
        label_train = json.load(f)
            
    for label in label_train:
        label_dict[label['id']] = label['caption']
    
    
    sentences = sum(label_dict.values(), [])    
    word_count = defaultdict(int)
    word_count2 = defaultdict(int)
    
    
    num_sentense = 0

    for sentence in sentences:
        num_sentense += 1
        temp_sen = sentence.translate(str.maketrans('', '', string.punctuation + '“' + '”')).lower().split()
        
    #    out = "".join(c for c in sentence if c not in ('!','.',':','', '',string.punctuation + '“' + '”'))
    #    out = out.lower().split()
        
        sentence = re.sub(r"\.",r"", sentence)
        finalXdot = sentence.strip().lower()
        temp_sen2 = finalXdot.split()

            
        for word2 in temp_sen2:
            word_count2[word2] += 1
    
        for word in temp_sen:
            word_count[word] += 1
            
                    
    vocab = [word for word in word_count if word_count[word] >= 1]
   # print('Filtered words from {} to {}.'.format(len(word_count), len(vocab)))
        
    
    vocab_dict = []
    for id_v, word_v in enumerate(sorted(vocab)): 
        vocab_dict.append(word_v) 
    
    vocab_dict = ['<pad>', '<bos>', '<eos>', '<unk>'] + vocab_dict


    return vocab_dict, label_dict


def set_captions(vocab_dict, label_dict, ids_train):
    y_train = []
    y_seq_len = []
    training_max_time_steps = 15
#    y_train = np.zeros((1450, training_max_time_steps + 1), dtype=np.int32)
#    y_seq_len = np.zeros((1450), dtype=np.int32)
    
    for index in range(len(ids_train)):
        id = ids_train[index]
        # Avoid the exception that words in sentence is above max 15, so we set threshold (10 words) 
        while True:
            choice = randrange(0, len(label_dict[id]))
            temp_choi = label_dict[id][choice].translate(str.maketrans('', '', string.punctuation + '“' + '”')).lower().split()
            if (len(temp_choi)<=10):
                break
        
        temp_label = []
        label = []
        for i_choi in range(0, len(temp_choi)):
            for i_vo in range(0, len(vocab_dict)):
                if(temp_choi[i_choi] == vocab_dict[i_vo]):
                    temp_label.append(i_vo)
        
        temp_label.insert(0,1)
        temp_label.append(2)
        
        label.append(temp_label)
        
        
        label = np.array(temp_label)
        
    #    label = sentenceEncoder.transform(label_dict[id][choice])
        label_len = len(label) - 1
        
        label_seq = []
    
        for i_label in range(0,len(label)):
            label_seq.append(int(label[i_label]))
    #    for i_max in range(0, training_max_time_steps):
        adding_seq = training_max_time_steps - len(label_seq) + 1 
        for i_adding in range(0, adding_seq):
            label_seq.append(0)
        
        y_train.append(label_seq)
        y_seq_len.append(label_len)
#        y_train[index] = label_seq
#        y_seq_len[index] = label_len
    
    y_train_np = np.array(y_train, dtype = np.int32)
    y_seq_len_np = np.array(y_seq_len, dtype = np.int32)  
    
    return y_train_np, y_seq_len_np
    

def next_batch(x_train,y_train,batch_size,data_s):
    
    ram = random.randint(0,data_s)
    while(ram+2*batch_size > data_s):
        ram = random.randint(0,data_s)
    
    idx = ram    
    x_batch = x_train[idx:idx+batch_size]
    y_batch = y_train[idx:idx+batch_size]
            
    return x_batch, y_batch, idx
        
    
