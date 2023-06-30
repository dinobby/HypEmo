import os
import json
import time
import pickle
import logging
import datetime
import optimizers
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score

from pytorch_metric_learning import losses
from config import parser, label_dicts, emb_dicts
from sentence_transformers import SentenceTransformer

from util_functions import *
from hypemo import HypEmo

args = parser.parse_args()
logging.basicConfig(filename=f'./exp/{args.dataset}_{args.alpha}_{args.gamma}.log', level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler())

label2idx, idx2label = label_dicts
num_classes = len(idx2label.items())
class_names = [v for k, v in sorted(idx2label.items(), key=lambda item: item[0])]
word2vec, idx2vec = emb_dicts
args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

np.random.seed(args.seed)
torch.manual_seed(args.seed)
logging.info(f'Using device {args.device}, seed={args.seed}, training on {args.dataset} dataset.')

gm = HypEmo(args.dataset, num_classes, class_names, idx2vec, args.alpha, args.gamma, batch_size=args.batch_size)
best_valid_weighted_f1, best_test_weighted_f1 = -1, -1
best_valid_loss = 1e5
train_loss, valid_loss = [], []
train_acc, valid_acc, test_acc = [], [], []
train_weighted_f1, valid_weighted_f1, test_weighted_f1 = [], [], []

total_train_time = 0
total_test_time = 0
for i in range(args.epochs):
    start_time = time.time()
    train_log = gm.train_step(i)
    end_time = time.time()
    train_time = end_time - start_time
    total_train_time += train_time
    logging.info(f"Training time for this epoch: {train_time:.2f}")
    
    valid_log = gm.valid_step(i)
    test_start_time = time.time()
    test_log = gm.test_step(i)    
    test_end_time = time.time()
    test_time = test_end_time - test_start_time
    total_test_time += test_time
    logging.info(f"Inference time for this epoch: {test_time:.2f}")

    train_acc.append(train_log['train_acc'])
    train_weighted_f1.append(train_log['train_weighted_f1'])
    train_loss.append(train_log['loss'])
    
    valid_acc.append(valid_log['valid_acc'])
    valid_weighted_f1.append(valid_log['valid_weighted_f1'])
    valid_loss.append(valid_log['valid_loss'])
    
    test_acc.append(test_log['test_acc'])
    test_weighted_f1.append(test_log['test_weighted_f1'])

    if valid_log['valid_loss'] < best_valid_loss:
        best_valid_loss = valid_log['valid_loss']
        test_acc_best_valid = test_log['test_acc']
        test_weighted_f1_best_valid = test_log['test_weighted_f1']
        logging.info(f"[valid loss new low] test | acc: {test_acc_best_valid:.04f}, f1: {test_weighted_f1_best_valid:.04f}")        
    
    if valid_log['valid_weighted_f1'] > best_valid_weighted_f1:
        best_valid_weighted_f1 = valid_log['valid_weighted_f1']
        best_valid_acc = valid_log['valid_acc']
        test_acc_best_valid = test_log['test_acc']
        test_weighted_f1_best_valid = test_log['test_weighted_f1']
        logging.info(f"[valid f1 new high] test | acc: {test_acc_best_valid:.04f}, f1: {test_weighted_f1_best_valid:.04f}")
        
    if test_log['test_weighted_f1'] > best_test_weighted_f1:
        best_test_weighted_f1 = test_log['test_weighted_f1']
        best_test_acc = test_log['test_acc']

    logging.info(f"[best] valid | acc: {best_valid_acc:.04f}, f1: {best_valid_weighted_f1:.04f}\n test | acc: {best_test_acc:.04f}, f1: {best_test_weighted_f1:.04f}") 
logging.info(f"Average training time: {total_train_time/args.epochs}")
logging.info(f"Average inference time: {total_test_time/args.epochs}")