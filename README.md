# DAE-TPGM
This is a deep autoencoder network based on a two-part-gamma model for analyzing single-cell RNA-seq data, implemented by python 3.

## Introduction
DAE-TPGM uses a two-part-gamma model to capture the statistical characteristics of semi-continuous normalized data and adaptively explores the potential relationships between genes for promoting data imputation through the use of deep autoencoder.

## <a name="compilation"></a>  Installation

You can click [here](https://github.com/PUGEA/DAE-TPGM) to download the DAE-TPGM software. 


### Requirements:

*   DAE-TPGM implementation uses [Scanpy](https://github.com/theislab/scanpy) to laod and pre-process the scRNA-seq data.
*   In DAE-TPGM, the Python codes is implemented using [Keras](https://github.com/keras-team/keras) and its [TensorFlow](https://github.com/tensorflow/tensorflow) backend.

&nbsp;




### Example 1-dimensionality reduction

Here, we take Paul et al. blood differentiation data as an example to show the specific process of dimensionality reduction using DAE-TPGM.

Step 1. Load data.

import scanpy.api as sc

paul15_adata = sc.datasets.paul15()

Step 2. Data preprocessing. Convert the original count data into semi continuous form.

sc.pp.normalize_per_cell(paul15_adata)

sc.pp.log1p(paul15_adata)

Step 3. Run DAE-TPGM model.

paul15_adata = dae_tpgm(adata_tpm, threads=4, copy=True, log1p=False, return_info=True)

Step 4. Visual display. Here, [DPT](https://www.nature.com/articles/nmeth.3971) method is utilized to measure the trajectory information of cell differentiation.

4.1 Compute a neighborhood graph of the low-dimensional data from DAE-TPGM.

sc.pp.neighbors(paul15_adata, n_neighbors=20, use_rep='Encoding', method='gauss')

4.2 Infer progression of cells through geodesic distance along the graph.

sc.tl.dpt(adata_tpm, n_branchings=1)

4.3 Scatter plot in Diffusion Map basis.

![Image text](https://github.com/PUGEA/DAE-TPGM/blob/main/demo_images/paul_2.png)
Figure 1. Scatter plots of diffusion map based on PCA (first row) and DAE-TPGM (second row) colored according to the cell type assignment from [Paul et al.](https://www.cell.com/cell/fulltext/S0092-8674(15)01493-2?_returnURL=https%3A%2F%2Flinkinghub.elsevier.com%2Fretrieve%2Fpii%2FS0092867415014932%3Fshowall%3Dtrue), diffusion pseudotime (DPT) and DPT branches.


### Example 2-imputation
Here, we take Klein dataset as an example to show the specific process of imputation using DAE-TPGM. The download address of the Klein dataset is [here](https://scrnaseq-public-datasets.s3.amazonaws.com/scater-objects/klein.rds).

Step 1. Feature selection. Select the top 3000 high variable genes by using the Seurat package
source('klein_filterHVG.R')

Step 2. Data preprocessing. Convert the original count data into semi continuous form.

klein_data = sc.read_csv('./klein_filter_data.csv')

sc.pp.normalize_per_cell(klein_data)

sc.pp.log1p(klein_data)

Step 3. Run DAE-TPGM model and save the imputed data.

klein_data = dae_tpgm(klein_data, threads=4, copy=True, log1p=False, return_info=True)

klein_data_df = pd.DataFrame(dae_tpgm_data.obsm['Imputation'])

klein_data_df.to_csv('./klein_imputation_data.csv', sep=',', index=True, header=True)

Step 4. Display of visualization and clustering performance.

![Image text](https://github.com/PUGEA/DAE-TPGM/blob/main/demo_images/klein_tsne_2.png)
Figure 2.The tSNE visualizations of the Klein dataset. The fighures illustrates the results obtained from the Klein dataset, with the dropout outputs imputed by DAE-TPGM, respectively.


![Image text](https://github.com/PUGEA/DAE-TPGM/blob/main/demo_images/klein_evaluation.png)

Figure 3. Clustering evaluation metrics including ACC, ARI, NMI, and F1 for the original data and imputed data of Klein dataset.

## Authors

The DAE-TPGM algorithm is developed by Shuchang Zhao. 

## Contact information

For any query, please contact Shuchang Zhao via shuchangzhao@nuaa.edu.cn or Xuejun Liu via xuejun.liu@nuaa.edu.cn.
