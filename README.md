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




### Usage

You can run the DAE-TPGM from the command line by run.py:

`python run.py input_file outputs`

where `input_file` is a CSV-formatted raw semi-continuous data with cells in rows and genes in columns. 

### Outputs

Output folder contains the low dimensional representation and imputation of input_file.
- `encoding.csv` is the low dimensional representation of each cell by encoder. The dimension is determined by the number of neurons in the middle layer of network.

- `imputation.csv` is the imputed output calculated based on the mean of TPGM according to the inferred parameters through decoder. It is formatted as a `cell x gene` matrix.






## Authors

The DAE-TPGM algorithm is developed by Shuchang Zhao. 

## Contact information

For any query, please contact Shuchang Zhao via shuchangzhao@nuaa.edu.cn or Xuejun Liu via xuejun.liu@nuaa.edu.cn.