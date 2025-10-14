import os
import scFANCL
import config
import time
import numpy as np
import torch
import loss_module
import torch.nn.functional as F
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import csv

from utils import get_device, adjust_learning_rate, save_model, cluster_embedding


def run(gene_exp, cluster_number, dataset, real_label, epochs, lr, temperature, dropout, layers, batch_size, m,
        save_pred=True, noise=None, use_cpu=None, cluster_methods=None):
    if cluster_methods is None:
        cluster_methods = []
    results = {}

    start = time.time()
    embedding, best_model = train_model(gene_exp=gene_exp, cluster_number=cluster_number, real_label=real_label,
                                        epochs=epochs, lr=lr, temperature=temperature,
                                        dropout=dropout, layers=layers, batch_size=batch_size,
                                        m=m, save_pred=save_pred, noise=noise, use_cpu=use_cpu)

    if save_pred:
        results[f"features"] = embedding
        results[f"max_epoch"] = best_model
    elapsed = time.time() - start
    res_eval = cluster_embedding(embedding, cluster_number, real_label, save_pred=save_pred,
                                 cluster_methods=cluster_methods)
    results = {**results, **res_eval, "dataset": dataset, "time": elapsed}

    return results


def train_model(gene_exp, cluster_number, real_label, epochs, lr,
                temperature, dropout, layers, batch_size, m,
                save_pred=False, noise=None, use_cpu=None, evaluate_training=True):
    device = get_device(use_cpu, gpu_id=config.args.cuda)

    print(device)

    dims = np.concatenate([[gene_exp.shape[1]], layers])
    data_aug_model = scFANCL.DataAug(dropout=dropout)
    encoder_q = scFANCL.BaseEncoder(dims)
    encoder_k = scFANCL.BaseEncoder(dims)
    instance_projector = scFANCL.MLP(layers[2], layers[2] + layers[3], layers[2] + layers[3])
    cluster_cfg = [layers[2], layers[3]]
    
    model = scFANCL.scFANCL(encoder_q, encoder_k, instance_projector, cluster_cfg, cluster_number, m=m)
    data_aug_model.to(device)
    model.to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    criterion_instance = loss_module.FN_InstanceLoss(temperature=temperature , phi=config.args.phi, debug_opt=True)
    criterion_cluster = loss_module.Corr_ClusterLoss()

    max_value, best_model = -1, -1

    idx = np.arange(len(gene_exp))
    for epoch in range(epochs):
        model.train()
        adjust_learning_rate(optimizer, epoch, lr)
        np.random.shuffle(idx)
        loss_instance_ = 0
        loss_cluster_ = 0
        loss_cons_  = 0
        for pre_index in range(len(gene_exp) // batch_size + 1):
            c_idx = np.arange(pre_index * batch_size,
                              min(len(gene_exp), (pre_index + 1) * batch_size))
            if len(c_idx) == 0:
                continue
            c_idx = idx[c_idx]
            c_inp = gene_exp[c_idx]
            input1 = data_aug_model(torch.FloatTensor(c_inp).to(device))
            input2 = data_aug_model(torch.FloatTensor(c_inp).to(device))

            if noise is None or noise == 0:
                input1 = torch.FloatTensor(input1).to(device)
                input2 = torch.FloatTensor(input2).to(device)
            else:
                noise1 = torch.normal(mean=0.0, std=noise, size=input1.shape, device=input1.device, dtype=input1.dtype)
                input1 = input1 + noise1
                noise2 = torch.normal(mean=0.0, std=noise, size=input2.shape, device=input2.device, dtype=input2.dtype)
                input2 = input2 + noise2
            q_instance, q_cluster, k_instance, k_cluster = model(input1, input2)

            features_instance = torch.cat(
                [q_instance.unsqueeze(1),
                 k_instance.unsqueeze(1)],
                dim=1)
            
            # loss_instance = criterion_instance(features_instance)

            loss_instance_out = criterion_instance(features_instance)
            if isinstance(loss_instance_out, tuple):
                loss_instance, dbg = loss_instance_out
            else:
                loss_instance, dbg = loss_instance_out, None

            loss_cluster = criterion_cluster(q_cluster, k_cluster)

            # eps = 1e-8
            # p = q_cluster.clamp_min(eps)
            # p = p / (p.sum(dim=1, keepdim=True) + eps)
            # q = k_cluster.clamp_min(eps)
            # q = q / (q.sum(dim=1, keepdim=True) + eps)

            # kl_qk = (p * (p.add(eps).log() - q.add(eps).log())).sum(dim=1).mean()
            # kl_kq = (q * (q.add(eps).log() - p.add(eps).log())).sum(dim=1).mean()
            # loss_consistency = 0.5 * (kl_qk + kl_kq)


            loss = loss_instance + loss_cluster
            loss_instance_ += loss_instance.item()
            loss_cluster_ += loss_cluster.item()
            # loss_cons_ += loss_consistency.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if dbg is not None:
                with torch.no_grad():
                    bs = dbg["batch_size"]
                    N = 2 * bs
                    false_neg_mask = dbg["false_neg_mask"]
                    if false_neg_mask.any():
                        device = false_neg_mask.device
                        c_idx_local = torch.as_tensor(c_idx, device=device)
                        row2data = c_idx_local[torch.arange(N, device=device) % bs]

                        real_label_t = torch.as_tensor(real_label, device=device)
                        row_labels = real_label_t[row2data]

                        idxs = false_neg_mask.nonzero(as_tuple=False)
                        K = idxs.size(0)
                        same_cnt = (row_labels[idxs[:,0]] == row_labels[idxs[:,1]]).sum().item()
                        ratio = same_cnt / K
                        mean_cos = float(dbg["raw_sim"][false_neg_mask].mean().item())

            
                        save_dir = os.path.join(os.getcwd(), "save", config.args.name)
                        os.makedirs(save_dir, exist_ok=True)
                        out_csv = os.path.join(save_dir, 'fn_summary.csv')

                        need_header = not os.path.exists(out_csv)
                        with open(out_csv, 'a', newline="") as f:
                            wr = csv.writer(f)
                            if need_header:
                                wr.writerow(["epoch", "batch", "phi", "FN_pairs", "same_label_cnt",
                                            "same_ratio", "mean_cosine"])
                            wr.writerow([epoch, pre_index, config.args.phi, K,
                                        same_cnt, f"{ratio:.4f}", f"{mean_cos:.6f}"])


            

        if evaluate_training and real_label is not None:
            model.eval()
            with torch.no_grad():
                q_instance, _, _, _ = model(torch.FloatTensor(gene_exp).to(device), None)
                features = q_instance.detach().cpu().numpy()
            res = cluster_embedding(features, cluster_number, real_label, save_pred=save_pred)
            print(
                f"Epoch {epoch}: Loss: {loss_instance_ + loss_cluster_}, ACC:{res['acc']} NMI: {res['nmi']}, "
                f"ARI: {res['ari']} "
            )

            if res['ari'] + res['nmi'] >= max_value:
                max_value = res['ari'] + res['nmi']
                save_model(config.args.name, model, optimizer, epoch, best_model)
                best_model = epoch

    model.eval()
    model_fp = os.path.join(os.getcwd(), "save", config.args.name, f"checkpoint_{best_model}.tar")
    model.load_state_dict(torch.load(model_fp, map_location=device)['net'])
    model.to(device)

    with torch.no_grad():
        q_instance, _, _, _ = model(torch.FloatTensor(gene_exp).to(device), None)
        q_np = q_instance.detach().cpu().numpy()

    return q_np, best_model



