import os
import pickle
import argparse
from utils.train_utils import add_flags_from_config

all_dataset_list = ['go_emotion', 'ED', 'ED_easy_4', 'ED_hard_a', 'ED_hard_b', 'ED_hard_c', 'ED_hard_d']
ENCODER_TYPE = 'roberta-base'
temperature = 0.3

config_args = {
    'training_config': {
        'batch_size': (100, 'batch size for training'),
        'epochs': (20, 'maximum number of epochs to train for'),
        'seed': (1234, 'seed for training'),
        'alpha': (0.9, 'weight for BCE*poincare_dist loss'),
        'gamma': (0.1, 'weight for poincare loss')
    },
    'poincare_model_config': {
        'dim': (100, 'hyperbolic label embedding dimension'),
        'feat_dim': (768, 'hidden size of text embedding'),
        'manifold': ('PoincareBall', 'which manifold to use, can be any of [Euclidean, Hyperboloid, PoincareBall]'),
        'c': (1.0, 'hyperbolic radius, set to None for trainable curvature'),
        'r': (2., 'fermi-dirac decoder parameter for lp'),
        't': (1., 'fermi-dirac decoder parameter for lp'),
        'dropout': (0.1, 'dropout probability'),
        'pos-weight': (0, 'whether to upweight positive class in node classification tasks'),
        'num-layers': (2, 'number of hidden layers in encoder'),
        'bias': (1, 'whether to use bias (1) or not (0)'),
        'act': ('relu', 'which activation function to use (or None for no activation)'),
        'n-heads': (4, 'number of attention heads for graph attention networks, must be a divisor dim'),
        'a': (0.2, 'alpha for leakyrelu in graph attention networks'),
    },
    'data_config': {
        'dataset': ('go_emotion', 'which dataset to use'),
    }
}

def get_dicts(dataset, return_emb_dicts=False):
    assert dataset in all_dataset_list
    if dataset == 'ED':
        label2idx = {'sad': 0, 'trusting': 1, 'terrified': 2, 'caring': 3, 'disappointed': 4,
             'faithful': 5, 'joyful': 6, 'jealous': 7, 'disgusted': 8, 'surprised': 9,
             'ashamed': 10, 'afraid': 11, 'impressed': 12, 'sentimental': 13, 
             'devastated': 14, 'excited': 15, 'anticipating': 16, 'annoyed': 17, 'anxious': 18,
             'furious': 19, 'content': 20, 'lonely': 21, 'angry': 22, 'confident': 23,
             'apprehensive': 24, 'guilty': 25, 'embarrassed': 26, 'grateful': 27,
             'hopeful': 28, 'proud': 29, 'prepared': 30, 'nostalgic': 31}
        
    elif dataset == 'go_emotion':
        label2idx = {'admiration': 0, 'amusement': 1, 'anger': 2,
              'annoyance': 3, 'approval': 4, 'caring': 5,
              'confusion': 6, 'curiosity': 7, 'desire': 8,
              'disappointment': 9, 'disapproval': 10, 'disgust': 11,
              'embarrassment': 12, 'excitement': 13, 'fear': 14,
              'gratitude': 15, 'grief': 16, 'joy': 17,
              'love': 18, 'nervousness': 19, 'optimism': 20,
              'pride': 21, 'realization': 22, 'relief': 23,
              'remorse': 24, 'sadness': 25, 'surprise': 26}
    
    elif dataset == 'ED_easy_4':
        label2idx = {'sad': 0, 'joyful': 1, 'angry': 2, 'afraid': 3}
    
    elif dataset == 'ED_hard_a':
        label2idx = {'anxious': 0, 'apprehensive': 1, 'afraid': 2, 'terrified': 3}
    
    elif dataset == 'ED_hard_b':
        label2idx = {'sad': 0, 'devastated': 1, 'sentimental': 2, 'nostalgic': 3}
    
    elif dataset == 'ED_hard_c':
        label2idx = {'angry': 0, 'ashamed': 1, 'furious': 2, 'guilty': 3}
    
    elif dataset == 'ED_hard_d':
        label2idx = {'anticipating': 0, 'excited': 1, 'hopeful': 2, 'guilty': 3}

    idx2label = {v: k for k, v in label2idx.items()}
    
    if return_emb_dicts:
        word2vec = pickle.load(open(os.path.join((os.path.abspath('')), 'label_tree', f'{dataset}.bin'), 'rb'))
        idx2vec = {k: word2vec[v] for k, v in idx2label.items()}
        return (label2idx, idx2label), (word2vec, idx2vec)
        
    return label2idx, idx2label
    
d, _ = config_args['data_config']['dataset']
label_dicts, emb_dicts = get_dicts(d, return_emb_dicts=True)
parser = argparse.ArgumentParser()
for _, config_dict in config_args.items():
    parser = add_flags_from_config(parser, config_dict)
