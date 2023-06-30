#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Python implementation of Hyperbolic Angular Cones"""

from .dag_emb_model import *

try:
    from autograd import grad  # Only required for optionally verifying gradients while training
    from autograd import numpy as grad_np
    AUTOGRAD_PRESENT = True
except ImportError:
    AUTOGRAD_PRESENT = False

# Cosine clipping epsilon
EPS = 1e-7

class HypConesModel(DAGEmbeddingModel):
    """Class for training, using and evaluating Order Embeddings."""
    def __init__(self,
                 train_data,
                 dim=5,
                 init_range=(-0.1, 0.1),
                 lr=0.1,
                 seed=0,
                 logger=None,

                 opt= 'rsgd',  # rsgd or exp_map
                 num_negative=1,
                 ### How to sample negatives for an edge (u,v)
                 neg_sampl_strategy='true_neg',  # 'all' (all nodes for negative sampling) or 'true_neg' (only nodes not connected)
                 where_not_to_sample='children',  # both or ancestors or children. Has no effect if neg_sampl_strategy = 'all'.
                 neg_edges_attach='both',  # How to form negative edges: 'parent' (u,v') or 'child' (u', v) or 'both'
                 neg_sampling_power=0.0,  # 0 for uniform, 1 for unigram, 0.75 for word2vec

                 margin=0.1,  # Margin for the loss.
                 K = 0.1,  # Minimum norm of vectors
                 epsilon=1e-5,  # Eps for projecting outside of the inner K-ball and inside the outer unit ball.
                 cvpr_loss=False
                ):
        super().__init__(train_data=train_data,
                         dim=dim,
                         init_range=init_range,
                         lr=lr,
                         opt=opt,
                         burn_in=0,
                         seed=seed,
                         logger=logger,
                         BatchClass=HypConesBatch,
                         KeyedVectorsClass=HypConesKeyedVectors,
                         num_negative=num_negative,
                         neg_sampl_strategy=neg_sampl_strategy,
                         where_not_to_sample=where_not_to_sample,
                         always_v_in_neg=False,
                         neg_sampling_power=neg_sampling_power,
                         neg_edges_attach=neg_edges_attach)
        self.margin = margin
        self.epsilon = epsilon
        self.K = K
        self.inner_radius = 2 * K / (1 + np.sqrt(1 + 4 * K * K))

        assert self.opt in ['rsgd', 'exp_map']

        # Initialize outside of the K ball, but inside the unit ball.
        self.kv.syn0 *= 1.0 / np.linalg.norm(self.kv.syn0, axis=1)[:,np.newaxis] # Normalize to unit length
        self.kv.syn0 *= self._np_rand.uniform(self.inner_radius + self.epsilon, self.inner_radius + 0.5, (self.kv.syn0.shape[0], 1)) # Renormalize
        assert not np.any(np.linalg.norm(self.kv.syn0, axis=1) <= self.inner_radius + self.epsilon)
        self.cvpr_loss = cvpr_loss
        if self.cvpr_loss == 'sim':
            self.cvpr_table = self._build_sim_table()
        elif self.cvpr_loss == 'trav':
            self.cvpr_table = self._build_table()
        
    def _build_sim_table(self):
        
        adj = self.adjacent_nodes
        table = np.zeros((len(self.kv.index2word), len(self.kv.index2word)))
        table = table - 1 # init to -1
        tree_depth = max([len(v) for v in self.adjacent_nodes.values()])
        print('tree depth', tree_depth)
        
        root = -1
        
        for i in range(len(self.kv.index2word)):
            for j in range(len(self.kv.index2word)):
                if i == j:
                    table[i][j] = 1
                    table[j][i] = 1
                    continue
                x, y = adj[i], adj[j]
                l_x, l_y = list(x), list(y)
                l_x.append(root)
                l_y.append(root)
                
                if i in l_y or j in l_x: # one of them is root
                    sim_level = abs(len(l_y) - len(l_x))
                elif len(l_x) == len(l_y):
                    sim_level = len(list(x|y)) - tree_depth
                else:
                    sim_level = tree_depth
                
                table[i][j] = sim_level
                table[j][i] = sim_level
                
        return table
        
    def _build_table(self):
        
        adj = self.adjacent_nodes
        table = np.zeros((len(self.kv.index2word), len(self.kv.index2word)))
        table = table - 1 # init to -1
        tree_depth = max([len(v) for v in self.adjacent_nodes.values()])
        print('tree depth', tree_depth)
        
        for i in range(len(self.kv.index2word)):
            for j in range(len(self.kv.index2word)):
                if table[i][j] != -1 and table[j][i] != -1:
                    continue
                if i == j:
                    table[i][j] = 1
                    table[j][i] = 1
                else:
                    x, y = adj[i], adj[j]
                    if len(x) == len(y):
                        dist = tree_depth - len(list(x&y)) # if they have
                    else:
                        if len(list(x|y)) == max(len(list(x)), len(list(y))):
                            dist = 1
                        else:
                            dist = tree_depth - len(list(x&y)) + abs(len(list(x)) - len(list(y)))
                    
                    table[i][j] = dist
                    table[j][i] = dist
        return table
                
    def _clip_vectors(self, vectors):
        """Clip vectors to have a norm of less than 1 - eps and more than inner_radius + eps.

        Parameters
        ----------
        vectors : numpy.array
            Can be 1-D,or 2-D (in which case the norm for each row is checked).

        Returns
        -------
        numpy.array
            Array with norms clipped below 1.
        """

        # Project vectors outside of the inner ball.
        # Clip vectors to have a norm at least inner_radius + epsilon.
        thresh = self.inner_radius + self.epsilon
        one_d = len(vectors.shape) == 1
        if one_d:
            norm = np.linalg.norm(vectors)
            if norm < thresh:
                vectors *= thresh / norm
        else:
            norms = np.linalg.norm(vectors, axis=1)
            if not (norms >= thresh).all():
                vectors[norms < thresh] *= (thresh / norms[norms < thresh])[:, np.newaxis]

        # Project vectors outside of the inner ball.
        # Clip vectors to have a norm at least inner_radius + epsilon.
        thresh = 1.0 - self.epsilon
        one_d = len(vectors.shape) == 1
        if one_d:
            norm = np.linalg.norm(vectors)
            if norm < thresh:
                return vectors
            else:
                return thresh * vectors / norm
        else:
            norms = np.linalg.norm(vectors, axis=1)
            if (norms < thresh).all():
                return vectors
            else:
                vectors[norms >= thresh] *= (thresh / norms[norms >= thresh])[:, np.newaxis]
                return vectors


    ### For autograd
    def _loss_fn(self, matrix, rels_reversed):
        """Given a numpy array with vectors for u, v and negative samples, computes loss value.

        Parameters
        ----------
        matrix : numpy.array
            Array containing vectors for u, v and negative samples, of shape (2 + negative_size, dim).
        rels_reversed : bool

        Returns
        -------
        float
            Computed loss value.

        Warnings
        --------
        Only used for autograd gradients, since autograd requires a specific function signature.
        """
        vector_u = matrix[0]
        vectors_v = matrix[1:]

        norm_u = grad_np.linalg.norm(vector_u)
        norms_v = grad_np.linalg.norm(vectors_v, axis=1)
        euclidean_dists = grad_np.linalg.norm(vector_u - vectors_v, axis=1)
        dot_prod = (vector_u * vectors_v).sum(axis=1)

        if not rels_reversed:
            # u is x , v is y
            cos_angle_child = (dot_prod * (1 + norm_u ** 2) - norm_u ** 2 * (1 + norms_v ** 2)) /\
                              (norm_u * euclidean_dists * grad_np.sqrt(1 + norms_v ** 2 * norm_u ** 2 - 2 * dot_prod))
            angles_psi_parent = grad_np.arcsin(self.K * (1 - norm_u**2) / norm_u) # scalar
        else:
            # v is x , u is y
            cos_angle_child = (dot_prod * (1 + norms_v ** 2) - norms_v **2 * (1 + norm_u ** 2) ) /\
                              (norms_v * euclidean_dists * grad_np.sqrt(1 + norms_v**2 * norm_u**2 - 2 * dot_prod))
            angles_psi_parent = grad_np.arcsin(self.K * (1 - norms_v**2) / norms_v) # 1 + neg_size

        # To avoid numerical errors
        clipped_cos_angle_child = grad_np.maximum(cos_angle_child, -1 + EPS)
        clipped_cos_angle_child = grad_np.minimum(clipped_cos_angle_child, 1 - EPS)
        angles_child = grad_np.arccos(clipped_cos_angle_child)  # 1 + neg_size

        energy_vec = grad_np.maximum(0, angles_child - angles_psi_parent)
        positive_term = energy_vec[0]
        negative_terms = energy_vec[1:]
        return positive_term + grad_np.maximum(0, self.margin - negative_terms).sum()


class HypConesBatch(DAGEmbeddingBatch):
    """Compute gradients and loss for a training batch."""
    def __init__(self,
                 vectors_u, # (1, dim, batch_size)
                 vectors_v, # (1 + neg_size, dim, batch_size)
                 indices_u,
                 indices_v,
                 rels_reversed,
                 hyp_cones_model,
                cvpr_loss=False):
        super().__init__(
            vectors_u=vectors_u,
            vectors_v=vectors_v,
            indices_u=indices_u,
            indices_v=indices_v,
            rels_reversed=rels_reversed,
            dag_embedding_model=None)
        self.margin = hyp_cones_model.margin
        self.K = hyp_cones_model.K
        self.cvpr_loss = cvpr_loss
        self.adj = hyp_cones_model.adjacent_nodes
        self.table = hyp_cones_model.cvpr_table
        print(len(indices_v), len(indices_u))
#         print(indices_v.shape, indices_u.shape)
        print(self.vectors_u.shape)
        print(self.vectors_v.shape)
        print('table size', self.table.shape)
#         raise Exception

    def _compute_loss(self):
        """Compute and store loss value for the given batch of examples."""
        if self._loss_computed:
            return
        self._loss_computed = True
        
        self.euclidean_dists = np.linalg.norm(self.vectors_u - self.vectors_v, axis=1)  # (1 + neg_size, batch_size)
        self.dot_prods = (self.vectors_u * self.vectors_v).sum(axis=1) # (1 + neg, batch_size)

        self.g = 1 + self.norms_v_sq * self.norms_u_sq - 2 * self.dot_prods
        self.g_sqrt = np.sqrt(self.g)

        self.euclidean_times_sqrt_g = self.euclidean_dists * self.g_sqrt

        if not self.rels_reversed:
            # u is x , v is y
            # (1 + neg_size, batch_size)
            child_numerator = self.dot_prods * (1 + self.norms_u_sq) - self.norms_u_sq * (1 + self.norms_v_sq)
            self.child_numitor = self.euclidean_times_sqrt_g * self.norms_u
            self.angles_psi_parent = np.arcsin(self.K * self.one_minus_norms_sq_u / self.norms_u) # (1, batch_size)

        else:
            # v is x , u is y
            # (1 + neg_size, batch_size)
            child_numerator = self.dot_prods * (1 + self.norms_v_sq) - self.norms_v_sq * (1 + self.norms_u_sq)
            self.child_numitor = self.euclidean_times_sqrt_g * self.norms_v
            self.angles_psi_parent = np.arcsin(self.K * self.one_minus_norms_sq_v / self.norms_v) # (1, batch_size)

        self.cos_angles_child = child_numerator / self.child_numitor
        # To avoid numerical errors
        self.clipped_cos_angle_child = np.maximum(self.cos_angles_child, -1 + EPS)
        self.clipped_cos_angle_child = np.minimum(self.clipped_cos_angle_child, 1 - EPS)
        self.angles_child = np.arccos(self.clipped_cos_angle_child)  # (1 + neg_size, batch_size)

        self.angle_diff = self.angles_child - self.angles_psi_parent
        self.energy_vec = np.maximum(0, self.angle_diff) # (1 + neg_size, batch_size)
          
        # CVPR loss
        pos_vec = self.angle_diff[0]
        neg_vec = self.angle_diff[1]
        
        idx_v = np.expand_dims(np.array(self.indices_v), 0)
        idx_u = np.expand_dims(np.array(self.indices_u), 0)
        idx_v = idx_v.reshape(self.vectors_v.shape[0], self.vectors_v.shape[-1])
        idx_u = idx_u.reshape(self.vectors_u.shape[0], self.vectors_u.shape[-1])
        anc_idx = np.array(self.indices_u)
        pos_idx = idx_v[0,:]
        neg_idx = idx_v[1,:]
        pos_dist, neg_dist = [],[]
        for i in range(len(pos_idx)):
            pos_dist.append(self.table[anc_idx[i]][pos_idx[i]])
            neg_dist.append(self.table[anc_idx[i]][neg_idx[i]])
        pos_dist, neg_dist = np.array(pos_dist), np.array(neg_dist)
        vec_diff = np.absolute(pos_vec/neg_vec).sum()
        dist_diff = np.absolute(pos_dist-neg_dist).sum()
        cvpr_loss = (vec_diff - dist_diff)**2
        
        self.pos_loss = self.energy_vec[0].sum()
        self.neg_loss = np.maximum(0, self.margin - self.energy_vec[1:]).sum()
        self.loss = self.pos_loss + self.neg_loss
        
        # cvpr loss here
        self.loss = cvpr_loss

    def _compute_loss_gradients(self):
        """Compute and store gradients of loss function for all input vectors."""
        if self._loss_gradients_computed:
            return
        self._compute_loss()

        self.norms_u = self.norms_u[:, np.newaxis, :] # (1, 1, batch_size)
        self.norms_v = self.norms_v[:, np.newaxis, :] # (1 + neg, 1, batch_size)
        self.euclidean_dists = self.euclidean_dists[:, np.newaxis, :] # (1 + neg, 1, batch_size)
        self.norms_u_sq = self.norms_u_sq[:, np.newaxis, :] # (1, 1, batch_size)
        self.norms_v_sq = self.norms_v_sq[:, np.newaxis, :] # (1 + neg, 1, batch_size)
        self.child_numitor = self.child_numitor[:, np.newaxis, :] # (1 + neg, 1, batch_size)
        self.cos_angles_child = self.cos_angles_child[:, np.newaxis, :] # (1 + neg, 1, batch_size)
        self.dot_prods = self.dot_prods[:, np.newaxis, :] # (1 + neg, 1, batch_size)
        self.g_sqrt = self.g_sqrt[:, np.newaxis, :] # (1 + neg, 1, batch_size)
        self.euclidean_times_sqrt_g = self.euclidean_times_sqrt_g[:, np.newaxis, :] # (1 + neg, 1, batch_size)
        self.clipped_cos_angle_child = self.clipped_cos_angle_child[:, np.newaxis, :]

        # gradient of |u-v| w.r.t. u and v
        euclidean_dists_grad_u = (self.vectors_u - self.vectors_v) / self.euclidean_dists # (1 + neg, dim, batch_size)
        euclidean_dists_grad_v = - euclidean_dists_grad_u

        sqrt_g_grad_u = (self.vectors_u * self.norms_v_sq - self.vectors_v) / self.g_sqrt
        sqrt_g_grad_v = (self.vectors_v * self.norms_u_sq - self.vectors_u) / self.g_sqrt

        euclid_times_sqrt_g_grad_u = sqrt_g_grad_u * self.euclidean_dists + euclidean_dists_grad_u * self.g_sqrt
        euclid_times_sqrt_g_grad_v = sqrt_g_grad_v * self.euclidean_dists + euclidean_dists_grad_v * self.g_sqrt

        if not self.rels_reversed:
            # u is x , v is y
            angle_psi_parent_grad_u = - self.K * self.vectors_u * (1 + self.norms_u_sq) / \
                                      (np.sqrt(self.norms_u_sq - self.K**2 * (1 - self.norms_u_sq)**2) * self.norms_u_sq)  # (1, dim, batch_size)
            angle_psi_parent_grad_v = np.zeros(self.vectors_v.shape) # (1 + neg_size, dim, batch_size)

            child_numerator_grad_u = self.vectors_v * (1 + self.norms_u_sq) +\
                                     2 * self.vectors_u * (self.dot_prods - 1 - self.norms_v_sq)
            child_numerator_grad_v = self.vectors_u * (1 + self.norms_u_sq) - 2 * self.vectors_v * self.norms_u_sq

            child_numitor_grad_u = self.euclidean_times_sqrt_g * self.vectors_u / self.norms_u + self.norms_u * euclid_times_sqrt_g_grad_u
            child_numitor_grad_v = self.norms_u * euclid_times_sqrt_g_grad_v

        else:
            # v is x , u is y
            angle_psi_parent_grad_v= - self.K * self.vectors_v * (1 + self.norms_v_sq) / \
                                      (np.sqrt(self.norms_v_sq - self.K**2 * (1 - self.norms_v_sq)**2) * self.norms_v_sq)  # (1, dim, batch_size)
            angle_psi_parent_grad_u = np.zeros(self.vectors_u.shape) # (1 + neg_size, dim, batch_size)

            child_numerator_grad_v = self.vectors_u * (1 + self.norms_v_sq) +\
                                     2 * self.vectors_v * (self.dot_prods - 1 - self.norms_u_sq)
            child_numerator_grad_u = self.vectors_v * (1 + self.norms_v_sq) - 2 * self.vectors_u * self.norms_v_sq

            child_numitor_grad_v = self.euclidean_times_sqrt_g * self.vectors_v / self.norms_v + self.norms_v * euclid_times_sqrt_g_grad_v
            child_numitor_grad_u = self.norms_v * euclid_times_sqrt_g_grad_u

        arccos_child_grad = - 1.0 / np.sqrt(1 - self.clipped_cos_angle_child **2) # (1 + neg_size, 1, batch_size)
        update_cond = (self.clipped_cos_angle_child == self.cos_angles_child)

        # (a/b)' = (a' - a b' / b) / b
        angles_child_grad_v = arccos_child_grad * update_cond * \
                              (child_numerator_grad_v - self.cos_angles_child * child_numitor_grad_v) / self.child_numitor
        angles_child_grad_u = arccos_child_grad * update_cond * \
                              (child_numerator_grad_u - self.cos_angles_child * child_numitor_grad_u) / self.child_numitor

        energy_cond = (self.angle_diff > 0)[:, np.newaxis, :] # (1 + neg_size, dim, batch_size)
        energy_vec_grad_u = energy_cond * (angles_child_grad_u - angle_psi_parent_grad_u) # (1 + neg_size, dim, batch_size)
        energy_vec_grad_v = energy_cond * (angles_child_grad_v - angle_psi_parent_grad_v) # (1 + neg_size, dim, batch_size)

        # neg_loss gradients
        neg_update_cond = (self.margin - self.energy_vec > 0)[:, np.newaxis, :] # (1 + neg_size, dim, batch_size)
        gradients_v = (-1.0 * neg_update_cond) * energy_vec_grad_v # (1 + neg_size, dim, batch_size)
        gradients_u = ((-1.0 * neg_update_cond) * energy_vec_grad_u)[1:].sum(axis=0) # (dim, batch_size)

        # pos loss gradients
        gradients_v[0] = energy_vec_grad_v[0]
        gradients_u += energy_vec_grad_u[0]

        self.loss_gradients_u = gradients_u # (dim, batch_size)
        self.loss_gradients_v = gradients_v # (1 + neg_size, dim, batch_size)

        assert not np.isnan(self.loss_gradients_u).any()
        assert not np.isnan(self.loss_gradients_v).any()
        self._loss_gradients_computed = True


class HypConesKeyedVectors(DAGEmbeddingKeyedVectors):
    """Class to contain vectors and vocab for the :class:`~HypConesModel` training class.
    Used to perform operations on the vectors such as vector lookup, distance etc.
    Inspired from KeyedVectorsBase.
    """
    def __init__(self):
        super(HypConesKeyedVectors, self).__init__()

    def is_a_scores_vector_batch(self, K, parent_vectors, other_vectors, rel_reversed):
        norm_parent = np.linalg.norm(parent_vectors, axis=1)
        norm_parent_sq = norm_parent ** 2
        norms_other = np.linalg.norm(other_vectors, axis=1)
        norms_other_sq = norms_other ** 2
        euclidean_dists = np.maximum(np.linalg.norm(parent_vectors - other_vectors, axis=1), 1e-6) # To avoid the fact that parent can be equal to child for the reconstruction experiment
        dot_prods = (parent_vectors * other_vectors).sum(axis=1)
        g = 1 + norm_parent_sq * norms_other_sq - 2 * dot_prods
        g_sqrt = np.sqrt(g)

        if not rel_reversed:
            # parent = x , other = y
            child_numerator = dot_prods * (1 + norm_parent_sq) - norm_parent_sq * (1 + norms_other_sq)
            child_numitor = euclidean_dists * norm_parent * g_sqrt
            angles_psi_parent = np.arcsin(K * (1 - norm_parent_sq) / norm_parent)
        else:
            # parent = y , other = x
            child_numerator = dot_prods * (1 + norms_other_sq) - norms_other_sq * (1 + norm_parent_sq)
            child_numitor = euclidean_dists * norms_other * g_sqrt
            angles_psi_parent = np.arcsin(K * (1 - norms_other_sq) / norms_other)

        cos_angles_child = child_numerator / child_numitor
        assert not np.isnan(cos_angles_child).any()
        clipped_cos_angle_child = np.maximum(cos_angles_child, -1 + EPS)
        clipped_cos_angle_child = np.minimum(clipped_cos_angle_child, 1 - EPS)
        angles_child = np.arccos(clipped_cos_angle_child)  # (1 + neg_size, batch_size)

        # return angles_child # np.maximum(1, angles_child / angles_psi_parent)
        return np.maximum(0, angles_child - angles_psi_parent)
