import pandas as pd
import scanpy.api as sc
from dae_tpgm.dae_tpgm_api import dae_tpgm

import sys
input_file = sys.argv[1]
outputs_path = sys.argv[2]

input = sc.read_csv(input_file)

input_adata = dae_tpgm(input, threads=4, copy=True, log1p=False, return_info=True)
input_adata = input_adata.copy()


encoding = pd.DataFrame(input_adata.obsm['Encoding'])
encoding.index = input_adata.obs_names
encoding.to_csv(outputs_path+'/'+'encoding.csv', sep=',', index=True, header=False)


imputation = pd.DataFrame(input_adata.obsm['Imputation'])
imputation.columns = input_adata.var_names
imputation.index = input_adata.obs_names
imputation.to_csv(outputs_path+'/'+'imputaion.csv', sep=',', index=True, header=True)




