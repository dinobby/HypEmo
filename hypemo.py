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
from torch.nn import MSELoss, CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score


from config import parser
from models.base_models import FGTCModel
from util_functions import *
from hypbert import HypBert
from transformers import AutoConfig

args = parser.parse_args()
args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)


class HypEmo():
    def __init__(self, dataset, n_classes, class_names, idx2vec, alpha, gamma, batch_size=16):
        trainset = HyoEmoDataSet(dataset, 'train')
        self.train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, collate_fn = trainset.collate)
        validset = HyoEmoDataSet(dataset, 'valid')
        self.valid_loader = DataLoader(validset, batch_size=256, shuffle=False, collate_fn = validset.collate)
        testset = HyoEmoDataSet(dataset, 'test')
        self.test_loader = DataLoader(testset, batch_size=256, shuffle=False, collate_fn = testset.collate)
        args.n_samples, args.feat_dim = len(trainset), 768
        args.n_classes = n_classes
        
        self.poincare_model = FGTCModel(args)
        self.poincare_optimizer = getattr(optimizers, 'RiemannianAdam')(params=self.poincare_model.parameters(), lr=0.01)
                                                    
        self.poincare_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.poincare_optimizer, step_size=10, gamma=0.5)
  
        self.class_names = class_names 
        self.idx2vec = idx2vec

        self.model = HypBert(num_labels=args.n_classes, alpha=alpha, gamma=gamma)
        self.model.to(args.device)
        self.poincare_model.to(args.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=1e-5, weight_decay=1e-2)
        
    def train_step(self, ith_epoch):

        self.model.train()

        train_pred, train_label = [], []
        step = 0
        total_loss, total_poincare_loss = 0.0, 0.0
        p_bar = tqdm(self.train_loader, total=len(self.train_loader))
        
        out_train_X, out_train_y = [], []
        for x, label in p_bar:
            step += 1
            input_ids, attention_mask, labels = x['input_ids'].to(args.device), x['attention_mask'].to(args.device), label.to(args.device)
           
            output = self.model(input_ids=input_ids, 
                                attention_mask=attention_mask, 
                                labels=labels, 
                                poincare_model=self.poincare_model,
                                idx2vec=self.idx2vec)
            
            out_train_X.extend(output['cls'].cpu().detach().numpy())
            out_train_y.extend(labels.cpu().detach().numpy())
                
            loss = output['total_loss']
            poincare_loss = output['poincare_loss']

            self.opt.zero_grad()
            self.poincare_optimizer.zero_grad()

            loss.backward()

            self.opt.step()
            self.poincare_optimizer.step()
            self.poincare_lr_scheduler.step()

            total_loss += loss.item()
            total_poincare_loss += poincare_loss.item()

            train_pred.extend(torch.argmax(output['logits'], dim=-1).tolist())
            train_label.extend(label.tolist())

            if step % 10 == 0:
                p_bar.set_description(f'train step {step} | loss={(total_loss/step):.4f}')

        train_acc = accuracy_score(train_pred, train_label)
        train_weighted_f1 = f1_score(train_pred, train_label, average='weighted')
        logging.info(f'''train | loss: {total_loss/step:.04f} acc: {train_acc:.04f}, f1: {train_weighted_f1:.04f}''')
        
        return {'loss': total_loss/step, 'train_acc': train_acc, 'train_weighted_f1': train_weighted_f1}


    def valid_step(self, ith_epoch):
        valid_pred = None
        valid_label = []
        with torch.no_grad():
            for x, label in self.valid_loader:
                input_ids, attention_mask, labels = x['input_ids'].to(args.device), x['attention_mask'].to(args.device), label.to(args.device)
                output = self.model(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    labels=labels, 
                                    poincare_model=self.poincare_model,
                                    idx2vec=self.idx2vec)
                logits = output['logits']
                loss = output['total_loss']
                prediction = torch.argmax(logits, dim=-1)
                if valid_pred is None:
                    valid_pred = prediction
                else:
                    valid_pred = torch.cat([valid_pred, prediction])
                valid_label.extend(label.tolist())

        valid_pred = valid_pred.detach().cpu().numpy()
        valid_acc = accuracy_score(valid_pred, valid_label)
        valid_weighted_f1 = f1_score(valid_pred, valid_label, average='weighted')
        logging.info(f'''valid | loss: {loss:.04f} acc: {valid_acc:.04f}, f1: {valid_weighted_f1:.04f}''')
        return {'valid_loss': loss, 'valid_pred': valid_pred, 'valid_acc': valid_acc, 'valid_weighted_f1': valid_weighted_f1}
    
    def test_step(self, ith_epoch):
        test_pred = None
        test_label = []
        with torch.no_grad():
            for x, label in self.test_loader:
                input_ids, attention_mask = x['input_ids'].to(args.device), x['attention_mask'].to(args.device)
                logits = self.model(input_ids=input_ids, attention_mask=attention_mask)['logits']
                prediction = torch.argmax(logits, dim=-1)
                if test_pred is None:
                    test_pred = prediction
                else:
                    test_pred = torch.cat([test_pred, prediction])
                test_label.extend(label.tolist())

        test_pred = test_pred.detach().cpu().numpy()
        test_acc = accuracy_score(test_pred, test_label)
        test_weighted_f1 = f1_score(test_pred, test_label, average='weighted')

        logging.info(f'''test | acc: {test_acc:.04f}, f1: {test_weighted_f1:.04f}''')
        return {'test_pred': test_pred, 'test_acc': test_acc, 'test_weighted_f1': test_weighted_f1}