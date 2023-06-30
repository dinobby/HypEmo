"""Base model class."""

import pickle
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from layers.layers import FermiDiracDecoder
import layers.hyp_layers as hyp_layers
import manifolds
import models.encoders as encoders
from models.decoders import model2decoder
from utils.eval_utils import acc_f1


class BaseModel(nn.Module):
    """
    Base model for graph embedding tasks.
    """

    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.manifold_name = args.manifold
        if args.c is not None:
            self.c = torch.tensor([args.c])
            if not args.cuda == -1:
                self.c = self.c.to(args.device)
        else:
            self.c = nn.Parameter(torch.Tensor([1.]))
        self.manifold = getattr(manifolds, self.manifold_name)()
        if self.manifold.name == 'Hyperboloid':
            args.feat_dim = args.feat_dim + 1
        self.nnodes = args.n_nodes
        self.encoder = getattr(encoders, 'HNN')(self.c, args)
        print(self.encoder)

    def encode(self, x, adj):
        if self.manifold.name == 'Hyperboloid':
            o = torch.zeros_like(x)
            x = torch.cat([o[:, 0:1], x], dim=1)
        h = self.encoder.encode(x, adj)
        return h

    def compute_metrics(self, embeddings, data, split):
        raise NotImplementedError

    def init_metric_dict(self):
        raise NotImplementedError

    def has_improved(self, m1, m2):
        raise NotImplementedError

class TextBaseModel(nn.Module):
    """
    Base model for text classification tasks.
    """

    def __init__(self, args):
        super(TextBaseModel, self).__init__()
        self.manifold_name = args.manifold
        if args.c is not None:
            self.c = torch.tensor([args.c])
            self.c = self.c.to(args.device)
        else:
            self.c = nn.Parameter(torch.Tensor([1.]))
        self.manifold = getattr(manifolds, self.manifold_name)()
        if self.manifold.name == 'Hyperboloid':
            args.feat_dim = args.feat_dim + 1
        self.n_samples = args.n_samples
        self.encoder = getattr(encoders, 'HNN')(self.c, args)

    def encode(self, x):
        if self.manifold.name == 'Hyperboloid':
            o = torch.zeros_like(x)
            x = torch.cat([o[:, 0:1], x], dim=1)
        h = self.encoder.encode(x)
        return h

    def compute_metrics(self, embeddings, data, split):
        raise NotImplementedError

    def init_metric_dict(self):
        raise NotImplementedError

    def has_improved(self, m1, m2):
        raise NotImplementedError

class FGTCModel(TextBaseModel):
    """
    Base model for fine grained text classification task.
    """

    def __init__(self, args):
        super(FGTCModel, self).__init__(args)
        self.decoder = model2decoder['HNN'](self.c, args)
        self.device = args.device
        self.dim = args.dim
        if args.n_classes > 2:
            self.f1_average = 'micro'
        else:
            self.f1_average = 'binary'
        if args.pos_weight:
            self.weights = torch.Tensor([1., 1. / data['labels'][idx_train].mean()])
        else:
            self.weights = torch.Tensor([1.] * args.n_classes)
            self.weights = self.weights.to(args.device)
    
    def poincare_ball_dist(self, u, v):
        u = u.detach().cpu().numpy()
        v = v.detach().cpu().numpy()
        euclidean_dists = np.linalg.norm(u - v)
        u_norm = np.linalg.norm(u)
        v_norm = np.linalg.norm(v)
        poincare_dists = np.arccosh(
            1 + 2 * (
                (euclidean_dists ** 2) / ((1 - u_norm ** 2) * (1 - v_norm ** 2))
            )
        )
        return poincare_dists

    def decode(self, h):
        output = self.decoder.decode(h)
        return F.log_softmax(output, dim=-1)

    def compute_metrics(self, embeddings, label, idx2vec):
        # embeddnigs is encoded by encoder. shape: [b, poincare_dim]
        label_emb = torch.FloatTensor(list(idx2vec.values())).to(self.device) # [num_class, poincare_dim]
        output = self.decode(embeddings)
        loss_ = F.nll_loss(output, label, self.weights) # [b, ]
        acc, f1 = acc_f1(output, label, average=self.f1_average)
        label_ = label.clone().detach().cpu().numpy()
        label_vec = torch.tensor(list(map(idx2vec.get, label_)), dtype=torch.float).to(self.device) # [b, p_dim]
        attention_score = torch.softmax(torch.matmul(embeddings.view(-1, 1, self.dim), label_emb.transpose(0, 1)), dim=1) # [b, num_class]
        # attend_feat = torch.matmul(attention_score.transpose(0, 1), embeddings) # B x n_labels x hidden_dimension

        dot_product = torch.bmm(embeddings.view(-1, 1, self.dim), label_vec.view(-1, self.dim, 1)) # [b, p_dim] x [p_dim, b] => [b, b]
        dot_product = torch.sum(dot_product, dim=0) # [b, ]
        loss = (loss_ - dot_product).sum()
        metrics = {'loss': loss, 'acc': acc, 'f1': f1, 'attention_score': attention_score}
        return metrics

    def init_metric_dict(self):
        return {'acc': -1, 'f1': -1}

    def has_improved(self, m1, m2):
        return m1["f1"] < m2["f1"]

