import math
import torch
import torch.nn as nn
import torch.nn.functional as F
                
EPS = 1e-8

class FN_InstanceLoss(nn.Module):
    def __init__(self, temperature, phi=0.95, debug_opt=True):
        super(FN_InstanceLoss, self).__init__()
        self.temperature = temperature
        self.phi = phi
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.debug_opt = debug_opt

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask.bool()


    def forward(self, features):
        
        device = features.device

        if len(features.shape) < 3:
            raise ValueError('features needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        
        batch_size = features.shape[0]
        N = 2 * batch_size
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        contrast_feature = F.normalize(contrast_feature, dim=-1)

        raw_sim = torch.matmul(contrast_feature, contrast_feature.T)
        raw_sim = torch.clamp(raw_sim, min=-1.0, max=1.0)

        sim = raw_sim / self.temperature

        #base negative_samples
        neg_mask = self.mask_correlated_samples(batch_size).to(device)
        negative_samples = sim[neg_mask].reshape(N, -1)  
        

        #phi_base negative_samples
        neg_raw_sim = raw_sim[neg_mask].reshape(N, -1)

        # print(neg_raw_sim.shape)
        keep = (neg_raw_sim < self.phi).detach()

        # print(neg_raw_sim)

        #positive_samples 
        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        pos = positive_samples.squeeze(1)
    
        #loss
        den_list = []
        for i in range(N):
            # 행 i에서 남긴 neg만 선택
            neg_i_kept = negative_samples[i][keep[i]]                         # shape: [Mi], Mi는 행마다 다름
            # 분모 = exp(pos_i) + Σ exp(neg_i_kept)
            den_i = torch.logsumexp(torch.cat([pos[i:i+1], neg_i_kept], dim=0), dim=0)
            den_list.append(den_i)
        den = torch.stack(den_list, dim=0)                                     # [N]

        # 5) 최종 손실: -log( exp(pos) / (exp(pos)+Σexp(neg_kept)) )
        loss = (-pos + den).mean()

        if self.debug_opt:
            # false_neg_mask: "neg 후보" 중에서 cosine >= phi 인 위치만 True
            false_neg_mask = neg_mask & (raw_sim >= self.phi)
            debug = {
                "contrast_feature": contrast_feature.detach(),
                "raw_sim": raw_sim.detach(),
                "neg_mask_full": neg_mask,         # [N,N] bool
                "false_neg_mask": false_neg_mask,       # [N,N] bool
                "batch_size": batch_size,
            }
            return loss, debug


        return loss

def off_diag(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class Corr_ClusterLoss(nn.Module):
    def __init__(self):
        super(Corr_ClusterLoss, self).__init__()

    def forward(self, q_cluster, k_cluster):

        q_cluster_t = q_cluster.t()
        k_cluster_t = k_cluster.t()

        p_i = q_cluster_t.sum(0).view(-1)
        p_i /= p_i.sum() + EPS
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i + EPS)).sum()
        p_j = k_cluster_t.sum(0).view(-1)
        p_j /= p_j.sum() + EPS
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j + EPS)).sum()
        ne_loss = ne_i + ne_j

        S = torch.mm(
            F.normalize(q_cluster.t(), dim=1, p=2),
            F.normalize(k_cluster, dim=0, p=2)
        )
        
        loss = (torch.diagonal(S) - 1).pow(2).mean() + off_diag(S).pow(2).mean()
        return ne_loss + loss

