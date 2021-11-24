import random
import anndata
import numpy as np


try:
    import tensorflow as tf
except ImportError:
    raise ImportError('dae_tpgm requires tensorflow. ')


from .dae_tpgm_io import read_dataset, normalize
from .dae_tpgm_train import train
from .dae_tpgm_network import AE_types


def dae_tpgm(adata,
        mode='encoding-imputation',
        ae_type='tpgm-conddisp',
        normalize_per_cell=False,
        scale=False,
        log1p=False,
        hidden_size=(100, 50, 25, 50, 100),
        hidden_dropout=0.5,
        batchnorm=True,
        activation='relu',
        init='glorot_normal',
        network_kwds={},
        epochs=400,
        batch_size=50,
        optimizer='rmsprop',
        random_state=100,
        threads=None,
        verbose=True,
        training_kwds={},
        return_model=False,
        return_info=False,
        copy=False
        ):

    print('\n')

    assert isinstance(adata, anndata.AnnData), 'adata must be an AnnData instance'
    assert mode in ('encoding-imputation'), '%s is not a valid mode.' % mode

    # set seed for reproducibility
    random.seed(random_state)
    np.random.seed(random_state)
    tf.set_random_seed(random_state)

    adata = read_dataset(adata,
                         transpose=False,
                         test_split=False,
                         copy=copy)


    adata = normalize(adata,
                      filter_min_counts=False,
                      size_factors=normalize_per_cell,
                      normalize_input=scale,
                      logtrans_input=log1p)

    network_kwds = {**network_kwds,
        'hidden_size': hidden_size,
        'hidden_dropout': hidden_dropout,
        'batchnorm': batchnorm,
        'activation': activation,
        'init': init
    }

    input_size = output_size = adata.n_vars
    net = AE_types[ae_type](input_size=input_size,
                            output_size=output_size,
                            **network_kwds)
    net.save()
    net.build()

    training_kwds = {**training_kwds,
        'epochs': epochs,
        'batch_size': batch_size,
        'optimizer': optimizer,
        'verbose': verbose,
        'threads': threads
    }

    hist = train(adata[adata.obs.dae_tpgm_split == 'train'], net, **training_kwds)
    res = net.predict(adata, mode, return_info, copy)
    adata = res if copy else adata

    if return_info:
        adata.uns['dae_tpgm_loss_history'] = hist.history

    if return_model:
        return (adata, net) if copy else net
    else:
        return adata if copy else None
