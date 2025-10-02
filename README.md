# scFANCL: Dual Contrastive Learning with False-negative correction at Cell level for single-cell RNA sequence Clustering


## Table of Contents

* [Model Overview](#model-overview)
* [Clustering Performance](#clustering-performnace)
* [Training scFANCL](#training-scfancl)
* [Recommended Hyperparameters of scFANCL](#recomended-hyperparameters-of-scfancl)
* [Datasets](#datasets)


## Model Overview
<img width="1395" height="553" alt="Framework" src="https://github.com/user-attachments/assets/da86e61a-34a4-4c1c-b021-b8b2c7991a3d" />


## Clustering Performance
(내용 작성)

## Training scFANCL
Our code supports both `.h5`, `.mat` and `.h5ad` file formats. To run scFANCL follow these steps:
1. Prepare Datasets File:<br>
  Save your dataset file and change the `path` to your Data directory
2. Configure the Dataset Type:<br>
  In `main.py`, add the dataset name(without extension) to the appropriate list:
  * For H5-format, add to the `h5_datasets` list.
  * For mat-format, add to the `mat_datasets` list.
  * For H5ad-format, you don't have have to do anything.


## Recommended Hyperparameters of scFANCL
(내용 작성)

## Datasets
(내용 작성)
