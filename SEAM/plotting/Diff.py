import numpy as np
import scanpy as sc
from ..settings import *

def Diff(a,cls,method='SIMLR_mz_emd',show_gene_labels=False):
    mzs = np.hstack(a.uns[method])
    #cls='SIMLR'
    sc.pl.heatmap(a,var_names=mzs,groupby=cls,standard_scale='var', 
                  cmap=heatmap_cmp,dendrogram=False,save=None,swap_axes=True,
                 show_gene_labels=show_gene_labels)

