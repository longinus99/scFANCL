# scFANCL: Dual Contrastive Learning with False-negative correction at Cell level for single-cell RNA sequence Clustering


## Table of Contents

- [Model Overview](#model-overview)
- [Clustering Performance](#clustering-performnace)
- [Required Environment](#required-environment)
- [Training scFANCL](#training-scfancl)
- [Recommended Hyperparameters of scFANCL](#recommended-hyperparameters-of-scfancl)
- [Datasets](#datasets)


## Model Overview
<img width="1395" height="553" alt="Framework" src="https://github.com/user-attachments/assets/da86e61a-34a4-4c1c-b021-b8b2c7991a3d" />


## Clustering Performance

Clustering performance of different models across various datasets, evaluated over 10 consecutive runs in terms of ACC, NMI and ARI. The best clustering result for each dataset is **bolded**.


| Dataset       | scFANCL (ACC/NMI/ARI)      | Seurat (ACC/NMI/ARI)   | scDCCA (ACC/NMI/ARI)      | ScCCL (ACC/NMI/ARI)      | JojoSCL (ACC/NMI/ARI)      |
|---------------|--------------------------|--------------------------|---------------------------|--------------------------|---------------------------|
| **10X PBMC**      | **0.8345/0.7868/0.7822**        | 0.7717/0.7496/0.7242            | 0.7956/0.7322/0.7455             | 0.8327/0.7864/0.7820            | 0.8326/0.7856/0.7819             |
| **Klein**   | **0.9447/0.8535/0.888**        | 0.8362/0.7709/0.8349            | 0.8887/0.8515/0.8304             | 0.8344/0.7827/0.7872      | 0.8312/0.7844/0.7797             |
| **Camp1**   | **0.9272/0.8852/0.8679**        | 0.7245/0.7591/0.6817            | 0.7956/0.7956/0.7455             | 0.8263/0.8516/0.7953      | 0.8668/0.8147/0.8654             |
| **Adam**   | **0.9645/0.9053/0.9246**        | 0.7002/0.7192/0.6749            | 0.7001/0.6751/0.5759             | 0.9620/0.9018/0.9193      | 0.9620/0.9012/0.92             |
| **Melanoma**   | **0.7391/0.6663/0.6565**        | 0.5493/0.532/0.3628            | 0.5254/0.4926/0.3632             | 0.7113/0.6599/0.6092      | 0.7173/0.6672/0.6164             |




## Required Environment
Experiments were conducted with CUDA 12.2 (GPU) and Python 3.7 </br>
Detail packages can be downloaded by `environment.yaml`


## Training scFANCL
Our code supports both `.h5`, `.mat` and `.h5ad` file formats. To run scFANCL follow these steps:
1. Prepare Datasets File:<br>
  Save your dataset file and change the `path` to your Data directory
2. Configure the Dataset Type:<br>
  In `main.py`, add the dataset name(without extension) to the appropriate list:
  * For H5-format, add to the `h5_datasets` list.
  * For mat-format, add to the `mat_datasets` list.
  * For H5ad-format, you don't have to do anything.


## Recommended Hyperparameters of scFANCL
Many settings for training and model configuration are managed in `config.py`. This includes parameters such as the dataset name, number of training epochs, number of genes to select (`--select_gene`), learning rate, dropout rate, and more. For example, the following parameters are defined:

- `--name`: Dataset name (default: "10X_PBMC")
- `--cuda`: Cuda (default: "0")
- `--epoch`: Number of training epochs (default: 200)
- `--dropout`: Dropout rate (default: 0.9)
- `--temperature`: Sharpness of similarity scores for instance-level contrastive module (default: 0.07)

These arguments allow for flexible configuration of the training process and are parsed using Python's `argparse` module. For contrastive learning, larger batch sizes tend to improve training stability. 


## Datasets

The detailed statistics of these datasets are summarized in the table below.

| Dataset | Cell | Gene | Cell Type | Sample|
|---------|---------|---------|---------|---------|
| 10X PBMC | 4271 | 16449 | 8 | Human PBMC |
| Klein | 2717 | 24175 | 4 | Mouse Embryo Stem Cells |
| Camp1 | 777 | 19020 | 7 | Human Liver |
| Adam | 3660 | 23797 | 8 | Human Kidney |
| Melanoma | 7186 | 23686 | 10 | Human Melanoma |


1. **Open the Google Drive link**: [Dataset directory](https://drive.google.com/drive/folders/1M0VaFZtQBDUqqXLGe8OUHzG1iIFNSTZE?usp=sharing).
2. **Select the files**  and **download** them to your local machine.
