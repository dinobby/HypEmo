import torch
import numpy as np
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from config import ENCODER_TYPE
from torch.nn import functional as F
from transformers import AutoModel
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
def poincare_ball_dist(u, v):
    if type(u) is not np.ndarray:
        u = u.detach().cpu().numpy()
        v = v.detach().cpu().numpy()
    if len(u.shape) ==1:
        euclidean_dists = np.linalg.norm(u - v)
        u_norm = np.linalg.norm(u)
        v_norm = np.linalg.norm(v)
        poincare_dists = np.arccosh(
            1 + 2 * (
                (euclidean_dists ** 2) / ((1 - u_norm ** 2) * (1 - v_norm ** 2))
            )
        )
    return poincare_dists

def batch_poincare_ball_dist_loss(batch_u, batch_v):
    if type(batch_u) is not np.ndarray:
        batch_u = batch_u.detach().cpu().numpy()
        batch_v = batch_v.detach().cpu().numpy()
    results = []
    for i in range(batch_u.shape[0]):
        d = poincare_ball_dist(batch_u[i], batch_v[i])
        results.append(d)
        
    results = torch.FloatTensor(results)
    return results

def one_to_batch_poincare_dist(vector_1, vectors_all):
    if type(vector_1) is not np.ndarray:
        vector_1 = vector_1.detach().cpu().numpy()
        vectors_all = vectors_all.detach().cpu().numpy()
    euclidean_dists = np.linalg.norm(vector_1 - vectors_all, axis=1)
    norm = np.linalg.norm(vector_1)
    all_norms = np.linalg.norm(vectors_all, axis=1)
    results = np.arccosh(
        1 + 2 * (
            (euclidean_dists ** 2) / ((1 - norm ** 2) * (1 - all_norms ** 2))
        )
    )
    results = torch.FloatTensor(results)
    return results
        
class HypBert(nn.Module):
    def __init__(self, num_labels, alpha, gamma):
        super().__init__()
        self.num_labels = num_labels
        self.alpha = alpha
        self.gamma = gamma
        self.encoder = AutoModel.from_pretrained(ENCODER_TYPE)
        self.config = self.encoder.config
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.pooler_fc = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.classifier = nn.Linear(self.config.hidden_size, self.num_labels)
        # self.init_weights()
        
    def get_emedding(self, features):
        x = features[:, 0, :]
        x = self.pooler_fc(x)
        x = self.dropout(x)
        x = F.relu(x)
        return x
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        poincare_model=None,
        idx2vec=None
    ):
        
        outputs = self.encoder(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=True
        )

        if ENCODER_TYPE == 'bert-base-uncased':
            sequence_output = outputs[0]
            pooled_output   = outputs[1]
        else:
            pooled_output   = self.get_emedding(outputs.hidden_states[-1])
           
        cls = pooled_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if poincare_model is not None:
            poincare_emb = poincare_model.encode(pooled_output)
            train_metrics = poincare_model.compute_metrics(poincare_emb, labels, idx2vec=idx2vec)
            poincare_loss = train_metrics['loss']
        else:
            poincare_loss = 0.0
        
        loss, poincare_dist, CE_loss = None, None, None
        if labels is not None:
                
            loss_fct = CrossEntropyLoss(reduction='none')
            CE_loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                        
            label_ = labels.clone().detach().cpu().numpy()
            label_vec = torch.tensor(list(map(idx2vec.get, label_)), dtype=torch.float).to(device) # (b, 100)
            poincare_weight = batch_poincare_ball_dist_loss(poincare_emb, label_vec).to(device) # (b,)
            
            loss = self.alpha * torch.mean(poincare_weight * CE_loss)
            loss += self.gamma * poincare_loss
            
        return {'total_loss': loss, 'CE_loss': CE_loss, 'poincare_loss': poincare_loss, 'poincare_dist': poincare_dist, 'logits': logits, 'cls': cls}