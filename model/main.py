import os
import h5py
import torch
import config
import numpy as np
import train
import scipy.io as sio
import scanpy as sc
import pandas as pd
from utils import preprocess

from sklearn.preprocessing import LabelEncoder

if __name__ == "__main__":
    h5_datasets = ['10X_PBMC']
    mat_datasets = []


    path = "put_your_dataset_forder_pathway"
    dataset = config.args.name
    gene_exp = []
    real_label = []
    if dataset in h5_datasets:
        file_path = os.path.join(path, f"{dataset}.h5")
        data_h5 = h5py.File(file_path, 'r')

        gene_exp = np.array(data_h5.get('X'))
        real_label = np.array(data_h5.get('Y')).reshape(-1)
        gene_exp = preprocess(gene_exp, config.args.select_gene)

    elif dataset in mat_datasets:
        file_path = os.path.join(path, f"{dataset}.mat")
        data_mat = sio.loadmat(file_path)

        gene_exp = np.array(data_mat['feature'])
        real_label = np.array(data_mat['label']).reshape(-1)
        gene_exp = preprocess(gene_exp, config.args.select_gene)

    else:
        file_path = os.path.join(path, f"{dataset}.h5ad")
        adata = sc.read_h5ad(file_path)

        gene_expr = adata.X

        if not isinstance(gene_expr, np.ndarray):
            gene_expr = gene_expr.toarray()
        string_labels = np.array(adata.obs["cell_type"]).reshape(-1)
        encoder = LabelEncoder()
        real_label = encoder.fit_transform(string_labels)
        gene_exp = preprocess(gene_expr, config.args.select_gene)
        gene_exp = np.asarray(gene_exp)

        
    print(f"The gene expression matrix shape is {gene_exp.shape}...")
    cluster_number = np.unique(real_label).shape[0]
    print(f"The real clustering num is {cluster_number}...")

    results = train.run(gene_exp=gene_exp, 
                        cluster_number=cluster_number, 
                        dataset=config.args.name,
                        real_label=real_label, 
                        epochs=config.args.epoch, 
                        lr=config.args.lr,
                        temperature=config.args.temperature, 
                        dropout=config.args.dropout,
                        layers=[config.args.enc_1, config.args.enc_2, config.args.enc_3, config.args.mlp_dim],
                        save_pred=True, 
                        cluster_methods=config.args.cluster_methods, 
                        batch_size=config.args.batch_size,
                        m=config.args.m, 
                        noise=config.args.noise)
                        
    print("ACC:" + str(results["acc"]))
    print("NMI:" + str(results["nmi"]))
    print("ARI:" + str(results["ari"]))

