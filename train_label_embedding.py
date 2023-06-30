"""
Runs our hyperbolic entailment cones on the synthetic data representing a uniform tree of some
fixed branching factor and some fixed depth. These trees are in data/toy/. This code will produce
an animation when embedding dimension is 2, and the animation will be opened in your browser at the
end of the training.
"""

# +
import csv
from collections import OrderedDict
import os
import pickle
import random
import re
from random import *

from smart_open import smart_open
import argparse
from utils.poincare_model import *
from utils.relations import *
from utils.eucl_cones_model import *
from utils.hyp_cones_model import *
from utils.poincare_viz import *
from utils.eval import *

import logging
import numpy as np
import sys
import plotly
from plotly.offline import plot
sys.path.insert(1, os.path.join(sys.path[0], '..'))

# TODO: fill here your plotly details before running.
plotly.tools.set_credentials_file(username='', api_key='')

from config import get_dicts

parser = argparse.ArgumentParser()
parser.add_argument("--tree", type=str, help='go_emotion | ED | ED_easy_4 | ED_hard_a | ED_hard_b | ED_hard_c | ED_hard_d ', default='ED')
parser.add_argument("--model", type=str, help='poincare | hyp_cones ', default='poincare')
parser.add_argument("--dim", type=int, default=100)
args = parser.parse_args()

default_params = {
    'model' : args.model, # poincare or hyp_cones (initialized with Poincare)
    'tree_depth' : 3, # a number between 1 and 7
    'level_branch' : 4,  # a number between 3 and 4
    'remove_root': True,

    'print_every': 5,
    'size': args.dim, # embedding dimension. If 2, this code will produce an animation.
    'lr': 0.025, # Learning rate
    'opt': 'rsgd', # rsgd or exp_map or sgd

    'burn_in': 20,  # Number of epochs to use for burn-in initialization
    'epsilon': 1e-5,
    'seed': 7, # random seed
    'num_negative': 5, # Number of negative samples to use (5 for go_emotion, 10 for empathatic dialogs)
    'neg_sampling_power': 0.75,  # 0 for uniform, 1 for unigram, 0.75 for word2vec

    'neg_sampl_strategy': 'true_neg',  # 'all', 'true_neg' , 'all_non_leaves' or 'true_neg_non_leaves'
    'where_not_to_sample': 'ancestors',  # both or ancestors or children. Has no effect if neg_sampl_strategy = 'all'.
    'neg_edges_attach': 'child',  # How to form negative edges: 'parent' (u,v') or 'child' (u', v) or 'both'
    'always_v_in_neg': True,  # always include the true edge (u,v) as negative.

    'loss_type': 'nll', # 'nll', 'neg', 'maxmargin'

    'maxmargin_margin': 2,
    'neg_r': 1,
    'neg_t': 2,
    'neg_mu': 1.0,  # Balancing factor between the positive and negative terms

    'epochs': 2000,  # Number of epochs to use
    'batch_size': 16,  # Size of batch to use for training
}

params = default_params.copy()
param_str_list = ['%s:%s' % (key, params[key]) for key in sorted(params.keys())]
figure_name = '; '.join(param_str_list)
animation = create_animation(figure_name)

if args.tree == 'go_emotion':
    data_directory = os.path.join((os.path.abspath('')), 'data', 'go_emotion')
    data_file_path = os.path.join(data_directory, 'label_tree.tsv')
    root_label = 'root'
    
elif args.tree == 'ED':
    data_directory = os.path.join((os.path.abspath('')), 'data', 'ED')
    data_file_path = os.path.join(data_directory, 'label_tree.tsv')
    root_label = 'root'    
    
elif args.tree == 'ED_easy_4':
    data_directory = os.path.join((os.path.abspath('')), 'data', 'ED_easy_4')
    data_file_path = os.path.join(data_directory, 'label_tree.tsv')
    root_label = 'root'     

elif args.tree == 'ED_hard_a':
    data_directory = os.path.join((os.path.abspath('')), 'data', 'ED_hard_a')
    data_file_path = os.path.join(data_directory, 'label_tree.tsv')
    root_label = 'root'     

elif args.tree == 'ED_hard_b':
    data_directory = os.path.join((os.path.abspath('')), 'data', 'ED_hard_b')
    data_file_path = os.path.join(data_directory, 'label_tree.tsv')
    root_label = 'root'     

elif args.tree == 'ED_hard_c':
    data_directory = os.path.join((os.path.abspath('')), 'data', 'ED_hard_c')
    data_file_path = os.path.join(data_directory, 'label_tree.tsv')
    root_label = 'root'

elif args.tree == 'ED_hard_d':
    data_directory = os.path.join((os.path.abspath('')), 'data', 'ED_hard_d')
    data_file_path = os.path.join(data_directory, 'label_tree.tsv')
    root_label = 'root'
    
# Recovers the tree from the transitive closure of a DAG
def recover_tree_from_transitive_closure(relations):
    all_nodes_set = set()
    for rel in relations:
        all_nodes_set.add(rel[0])
        all_nodes_set.add(rel[1])

    ancestors = {}
    for node in all_nodes_set:
        ancestors[node] = []
    for rel in relations:
        if rel[0] != rel[1]:
            ancestors[rel[1]].append(rel[0])

    new_relations = []
    for node in all_nodes_set:
        num_ancestors = len(ancestors[node])
        for ancestor in ancestors[node]:
            if len(ancestors[ancestor]) == num_ancestors - 1:
                new_relations.append((ancestor, node))
    return new_relations

def read_tree_data(data=args.tree):
    # Load the tree data:
    transitive_relations = Relations(file_path=data_file_path, reverse=True)
    tree_relations = recover_tree_from_transitive_closure(transitive_relations)
                                                                                 
    # All direct children of root
    transitive_relations_without_root = []
    tree_relations_without_root = []
    for rel in tree_relations:
        if rel[0] != root_label:
            tree_relations_without_root.append(rel)

    for rel in transitive_relations:
        if rel[0] != root_label:
            transitive_relations_without_root.append(rel)

    return transitive_relations_without_root, tree_relations_without_root

transitive_relations, tree_relations = read_tree_data()

label2idx, idx2label = get_dicts(args.tree, return_emb_dicts=False)
class_names = [v for k, v in sorted(idx2label.items(), key=lambda item: item[0])]

# Create the Poincare model
model = PoincareModel(train_data=transitive_relations,
                      dim=params['size'],
                      lr=params['lr'],
                      opt=params['opt'],
                      burn_in=params['burn_in'],
                      epsilon=params['epsilon'],
                      seed=params['seed'],
                      # logger=logger,
                      num_negative=params['num_negative'],
                      ### How to sample negatives for an edge (u,v)
                      neg_sampl_strategy=params['neg_sampl_strategy'],
                      # 'all' (all nodes used for negative sampling) or 'true_neg' (only not connected nodes)
                      where_not_to_sample=params['where_not_to_sample'],
                      # both or ancestors or children. Has no effect if neg_sampl_strategy = 'all'.
                      always_v_in_neg=params['always_v_in_neg'],  # always include the true edge (u,v) as negative.
                      neg_sampling_power=params['neg_sampling_power'],  # 0 for uniform, 1 for unigram, 0.75 for word2vec
                      ### How to use the negatives in the loss function
                      neg_edges_attach=params['neg_edges_attach'],
                      # How to form negative edges: 'parent' (u,v') or 'child' (u', v) or 'both'
                      loss_type=params['loss_type'],
                      maxmargin_margin=params['maxmargin_margin'],
                      neg_r=params['neg_r'],
                      neg_t=params['neg_t'],
                      neg_mu=params['neg_mu'],
                      )

for i in range(int(params['epochs'] / params['print_every'])):
    print('Starting epoch ' + str(params['print_every'] * i))

    model.train(epochs=params['print_every'], batch_size=params['batch_size'], print_every=params['print_every'])

    # Animation
    if params['size'] == 2:
        figure = poincare_2d_visualization(
            model,
            animation=animation,
            epoch=(params['print_every'] * i),
            eval_result='',
            avg_loss=0,
            avg_pos_loss=0,
            avg_neg_loss=0,
            tree=list(tree_relations),
            show_node_labels=class_names,
            figure_title=figure_name,
            num_nodes=None)

if default_params['model'] == 'poincare':
    if args.dim == 2: plot(animation)
    word2vec = {}
    for k in label2idx.keys():
        if k in model.kv:
            word2vec[k] = model.kv[k].astype(float)

    # word2vec['sad'].astype(float)
    bin_directory = os.path.join((os.path.abspath('')), 'label_tree')
    bin_file_path = os.path.join(bin_directory, args.tree + '.bin')
    pickle.dump(word2vec, open(bin_file_path, 'wb'))
    # model.save(f'saved_poincare_model_{args.tree}.pt')
    os._exit(0)


def poincare_ball_dist(u, v):
    euclidean_dists = np.linalg.norm(u - v)
    u_norm = np.linalg.norm(u)
    v_norm = np.linalg.norm(v)
    poincare_dists = np.arccosh(
        1 + 2 * (
            (euclidean_dists ** 2) / ((1 - u_norm ** 2) * (1 - v_norm ** 2))
        )
    )
    return poincare_dists

model = HypConesModel(transitive_relations,
                      dim=params['size'],
                      init_range=(-0.1, 0.1),
                      lr=0.0001,
                      seed=params['seed'],
                      logger=logger,
                      num_negative=5,
                      ### How to sample negatives for an edge (u,v)
                       neg_sampl_strategy='true_neg',
                      # 'all', 'true_neg' , 'all_non_leaves' or 'true_neg_non_leaves'
                       where_not_to_sample='ancestors',
                      # both or ancestors or children. Has no effect if neg_sampl_strategy = 'all'.
                       neg_edges_attach='child',
                      # How to form negative edges: 'parent' (u,v') or 'child' (u', v) or 'both'
                       neg_sampling_power=0,
                      # 0 for uniform, 1 for unigram, 0.75 for word2vec
                       margin=0.01,  # Margin for the loss.
                       opt=params['opt'],
                       K=0.1,
                       epsilon=1e-4,
                      cvpr_loss='sim'
                       )

vecs = model.kv.syn0
model.kv.syn0 = model._clip_vectors(vecs * 0.8)

print('Finished initialization. Now training the hyperbolic cones..')

# Train the model
for i in range(int(params['epochs'] / params['print_every'])):
    print('Starting epoch ' + str(params['print_every'] * i))
    # Train
    if i < 1:
        # No training here, just to plot the initial state.
        avg_loss, avg_pos_loss, avg_neg_loss = \
            model.train(epochs=0, batch_size=params['batch_size'], print_every=params['print_every'])
    else:
        avg_loss, avg_pos_loss, avg_neg_loss = \
            model.train(epochs=params['print_every'], batch_size=params['batch_size'], print_every=params['print_every'])

    # Animation
    if params['size'] == 2:
        figure = poincare_2d_visualization(
            model,
            animation=animation,
            epoch=(params['epochs'] + params['print_every'] * (i+1)),
            eval_result='',
            avg_loss=avg_loss,
            avg_pos_loss=avg_pos_loss,
            avg_neg_loss=avg_neg_loss,
            tree=list(tree_relations),
            show_node_labels=show_node_labels,
            figure_title=figure_name,
            num_nodes=None)

if params['size'] == 2:
    plot(animation)

word2vec = {}
for k in label2idx.keys():
    if k in model.kv:
        word2vec[k] = model.kv[k].astype(float)
        
if 'sad' in word2vec:
    # little buggy in ED dataset
    word2vec['sad'].astype(float)
pickle.dump(word2vec, open(os.path.join((os.path.abspath('..'), "label_tree", args.tree + ".bin") , 'wb')))

# ### If one wants to save and reload a model later:
#     pickle.dump(model.kv, open(args.root_path + "model_" + str(dim) + "D.bin", 'wb'))
#
#     model = PoincareModel(train_data=transitive_relations, dim=dim, logger=logger)
#     model.kv = pickle.load(open(args.root_path + "model_" + str(dim) + "D.bin", 'rb'))
#
#     if dim == 2:
#         figure = poincare_2d_visualization(
#             model,
#             animation=animation,
#             epoch=(0),
#             eval_result='',
#             avg_loss=0,
#             avg_pos_loss=0,
#             avg_neg_loss=0,
#             tree=list(),
#             show_node_labels=show_node_labels,
#             figure_title=figure_name,
#             num_nodes=None)
#     plot(animation)


