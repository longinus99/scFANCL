import torch.nn as nn
import torch
import torch.nn.functional as F

from  clu_head import Clu_Head

class DataAug(nn.Module):
    def __init__(self, dropout=0.9):
        super(DataAug, self).__init__()
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x):
        aug_data = self.drop(x)

        return aug_data


class BaseEncoder(nn.Module):
    def __init__(self, dims):
        super(BaseEncoder, self).__init__()
        self.dims = dims
        self.n_stacks = len(self.dims)  # -1
        enc = [nn.Linear(self.dims[0], self.dims[1]), nn.BatchNorm1d(self.dims[1]), nn.ReLU(),
               nn.Linear(self.dims[1], self.dims[2]), nn.BatchNorm1d(self.dims[2]), nn.ReLU(),
               nn.Linear(self.dims[2], self.dims[3]), nn.BatchNorm1d(self.dims[3]), nn.ReLU()]

        self.encoder = nn.Sequential(*enc)
        self._reset_prams()

    def _reset_prams(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        return

    def forward(self, x):
        latent_out = self.encoder(x)
        latent_out = F.normalize(latent_out, dim=1)

        return latent_out


class MLP(nn.Module):
    def __init__(self, contrastive_dim, mlp_dim, dim):
        super(MLP, self).__init__()
        self.contrastive_dim = contrastive_dim
        self.dim = dim
        self.mlp_dim = mlp_dim
        self.encoder = nn.Sequential(
            nn.Linear(self.contrastive_dim, self.mlp_dim),
            nn.ReLU(),
            nn.Linear(self.mlp_dim, self.dim),
        )

    def forward(self, x):
        latent_out = self.encoder(x)
        return latent_out


class scFANCL(nn.Module):

    def __init__(self, encoder_q, encoder_k, instance_projector, cluster_projector, class_num,
                 m=0.2, last_activation="softmax"):
        super(scFANCL, self).__init__()

        self.cluster_num = class_num
        self.m = m

        self.encoder_q = encoder_q
        self.encoder_k = encoder_k
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        self.instance_projector = instance_projector

        self.cluster_projector = Clu_Head(cfg=cluster_projector + [self.cluster_num], last_activation="softmax")

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    def forward(self, cell_q, cell_k):
        q = self.encoder_q(cell_q)
        q_instance = F.normalize(self.instance_projector(q), dim=1)
        q_cluster = self.cluster_projector(q)

        if cell_k is None:
            return q_instance, q_cluster, None, None

        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = self.encoder_k(cell_k)
            k_instance = F.normalize(self.instance_projector(k), dim=1)
            k_cluster = self.cluster_projector(k)

        return q_instance, q_cluster, k_instance, k_cluster
