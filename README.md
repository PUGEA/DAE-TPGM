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

1. Load data
import scanpy.api as sc
paul15_adata = sc.datasets.paul15()

2. Data preprocessing: convert the original count data into semi continuous form

sc.pp.normalize_per_cell(paul15_adata)
sc.pp.log1p(paul15_adata)

3. Run DAE-TPGM model
paul15_adata = dae_tpgm(adata_tpm, threads=4, copy=True, log1p=False, return_info=True)

4. Visual display. Here, [DPT](https://www.nature.com/articles/nmeth.3971) method is utilized to measure the trajectory information of cell differentiation.
4.1 Compute a neighborhood graph of the low-dimensional data from DAE-TPGM 
sc.pp.neighbors(paul15_adata, n_neighbors=20, use_rep='Encoding', method='gauss')

4.2 Infer progression of cells through geodesic distance along the graph
sc.tl.dpt(adata_tpm, n_branchings=1)

4.3 Scatter plot in Diffusion Map basis.


### Outputs

Output folder contains the low dimensional representation and imputation of input_file.
- `encoding.csv` is the low dimensional representation of each cell by encoder. The dimension is determined by the number of neurons in the middle layer of network.

- `imputation.csv` is the imputed output calculated based on the mean of TPGM according to the inferred parameters through decoder. It is formatted as a `cell x gene` matrix.






## Authors

The DAE-TPGM algorithm is developed by Shuchang Zhao. 

## Contact information

For any query, please contact Shuchang Zhao via shuchangzhao@nuaa.edu.cn or Xuejun Liu via xuejun.liu@nuaa.edu.cn.
