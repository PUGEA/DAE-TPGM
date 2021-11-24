from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import numpy as np
import pandas as pd
import scanpy.api as sc
from sklearn.model_selection import train_test_split

def read_dataset(adata, transpose=False, test_split=False, copy=False):

    if isinstance(adata, sc.AnnData):
        if copy:
            adata = adata.copy()
    elif isinstance(adata, str):
        adata = sc.read(adata, first_column_names=True)
    else:
        raise NotImplementedError

    if transpose: adata = adata.transpose()

    if test_split:
        train_idx, test_idx = train_test_split(np.arange(adata.n_obs), test_size=0.1, random_state=42)
        spl = pd.Series(['train'] * adata.n_obs)
        spl.iloc[test_idx] = 'test'
        adata.obs['dae_tpgm_split'] = spl.values
    else:
        print('No split')
        adata.obs['dae_tpgm_split'] = 'train'
        print('adata.obs[dae_tpgm_split]=',adata.obs['dae_tpgm_split'] )

    adata.obs['dae_tpgm_split'] = adata.obs['dae_tpgm_split'].astype('category')
    print('dae_tpgm: Successfully preprocessed {} genes and {} cells.'.format(adata.n_vars, adata.n_obs))

    return adata

def normalize(adata, filter_min_counts=True, size_factors=None, normalize_input=None, logtrans_input=None):

    if filter_min_counts:
        sc.pp.filter_genes(adata, min_counts=1)
        sc.pp.filter_cells(adata, min_counts=1)

    if size_factors or normalize_input or logtrans_input:
        adata.raw = adata.copy()
    else:
        adata.raw = adata

    if size_factors:
        sc.pp.normalize_per_cell(adata)
        adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
    else:
        adata.obs['size_factors'] = 1.0

    if logtrans_input:
        sc.pp.log1p(adata)

    if normalize_input:
        sc.pp.scale(adata)

    return adata

def read_genelist(filename):
    genelist = list(set(open(filename, 'rt').read().strip().split('\n')))
    assert len(genelist) > 0, 'No genes detected in genelist file'
    return genelist

def write_text_matrix(matrix, filename, rownames=None, colnames=None, transpose=False):
    if transpose:
        matrix = matrix.T
        rownames, colnames = colnames, rownames

    pd.DataFrame(matrix, index=rownames, columns=colnames).to_csv(filename,
                                                                  sep='\t',
                                                                  index=(rownames is not None),
                                                                  header=(colnames is not None),
                                                                  float_format='%.6f')
def read_pickle(inputfile):
    return pickle.load(open(inputfile, "rb"))