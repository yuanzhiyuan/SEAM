from ..utils import *
import scanpy as sc

def Cluster(a,cls,groups,method='mask'):
#     groups are list of label index
    if method=='mask':
        plot_label_image(a,a.obs[cls],a.uns[cls+'_colors'],mask=groups,save=None)
    elif method=='dot':
        unique_labels = np.unique(a.obs[cls])
        sc.pl.embedding(a,basis='spatial',color=cls,groups=list(unique_labels[groups]))        
        
