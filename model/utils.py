import torch
import numpy as np
import scanpy as sc
import scipy as sp
import math
import os
import time
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def preprocess(gene_exp, select_genes):
    X = np.ceil(gene_exp).astype(np.int)
    count_X = X
    print(X.shape, count_X.shape, f"keeping {select_genes} genes")
    adata = sc.AnnData(X)

    adata = counts_normalize(adata,
                      copy=True,
                      highly_genes=select_genes,
                      size_factors=True,
                      normalize_input=True,
                      logtrans_input=True)
    X = adata.X.astype(np.float32)
    return X


def counts_normalize(adata, copy=True, highly_genes=None, filter_min_counts=True,
              size_factors=True, normalize_input=True, logtrans_input=True):
    if isinstance(adata, sc.AnnData):
        if copy:
            adata = adata.copy()
    elif isinstance(adata, str):
        adata = sc.read(adata)
    else:
        raise NotImplementedError
    norm_error = 'Make sure that the dataset (adata.X) contains unnormalized count data.'
    assert 'n_count' not in adata.obs, norm_error
    if adata.X.size < 50e6:
        if sp.sparse.issparse(adata.X):
            assert (adata.X.astype(int) != adata.X).nnz == 0, norm_error
        else:
            assert np.all(adata.X.astype(int) == adata.X), norm_error

    if filter_min_counts:
        sc.pp.filter_genes(adata, min_counts=1)
        sc.pp.filter_cells(adata, min_counts=1)
    if size_factors or normalize_input or logtrans_input:
        adata.raw = adata.copy()
    else:
        adata.raw = adata
    if size_factors:
        sc.pp.normalize_per_cell(adata)
        adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
    else:
        adata.obs['size_factors'] = 1.0
    if logtrans_input:
        sc.pp.log1p(adata)
    if highly_genes != None:
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes = highly_genes, subset=True)
    if normalize_input:
        sc.pp.scale(adata)
    return adata


def get_device(use_cpu= None, gpu_id=0):
    if use_cpu is None:
        if torch.cuda.is_available():
            device = torch.device(f'cuda:{gpu_id}')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')
    return device


def adjust_learning_rate(optimizer, epoch, lr):
    p = {
        'epochs': 500,
        'optimizer': 'sgd',
        'optimizer_kwargs':
            {'nesterov': False,
             'weight_decay': 0.0001,
             'momentum': 0.9,
             },
        'scheduler': 'cosine',
        'scheduler_kwargs': {'lr_decay_rate': 0.1},
    }

    new_lr = None

    if p['scheduler'] == 'cosine':
        eta_min = lr * (p['scheduler_kwargs']['lr_decay_rate'] ** 3)
        new_lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / p['epochs'])) / 2

    elif p['scheduler'] == 'step':
        steps = np.sum(epoch > np.array(p['scheduler_kwargs']['lr_decay_epochs']))
        if steps > 0:
            new_lr = lr * (p['scheduler_kwargs']['lr_decay_rate'] ** steps)

    elif p['scheduler'] == 'constant':
        new_lr = lr

    else:
        raise ValueError('Invalid learning rate schedule {}'.format(p['scheduler']))

    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

    return lr


def save_model(name, model, optimizer, current_epoch, pre_epoch):
    dir_path = os.path.join(os.getcwd(), "save", name)
    os.makedirs(dir_path, exist_ok=True)

    if pre_epoch != -1:
        pre_path = os.path.join(os.getcwd(), "save", name, "checkpoint_{}.tar".format(pre_epoch))
        os.remove(pre_path)
    cur_path = os.path.join(os.getcwd(), "save", name, "checkpoint_{}.tar".format(current_epoch))
    state = {'net': model.state_dict(), 
            'optimizer': optimizer.state_dict(), 
            'epoch': current_epoch}
    torch.save(state, cur_path)

def cluster_acc(y_true, y_pred):
    """
    Compute clustering accuracy (ACC) with optimal label mapping via the Hungarian algorithm.
    y_true, y_pred: 1-D integer arrays of the same length.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    D = max(y_pred.max(), y_true.max()) + 1
    # build contingency matrix
    w = np.zeros((D, D), dtype=int)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    # solve assignment problem
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return w[row_ind, col_ind].sum() / y_pred.size


def cluster_embedding(embedding, cluster_number, real_label, save_pred=False, cluster_methods=None):
    if cluster_methods is None:
        cluster_methods = ["KMeans"]
    result = {"t_clust": time.time()}
    if "KMeans" in cluster_methods:
        kmeans = KMeans(n_clusters=cluster_number, init="k-means++", random_state=0, n_init=10)
        pred = kmeans.fit_predict(embedding)
        if real_label is not None:
            result[f"ari"] = round(adjusted_rand_score(real_label, pred), 4)
            result[f"nmi"] = round(normalized_mutual_info_score(real_label, pred), 4)
            result[f"acc"] = round(cluster_acc(real_label, pred),4)
        result["t_k"] = time.time()
        if save_pred:
            result[f"pred"] = pred

    return result