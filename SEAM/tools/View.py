from sklearn.manifold import TSNE
import umap
import numpy as np
from sklearn.preprocessing import *
def View(a,method='Umap'):
    
    data_all = a.uns['IMS']
    pseudo_count=1
    data_all_norm = (data_all+pseudo_count)/(np.percentile(data_all,50,axis=1,keepdims=True)+pseudo_count)
    data_all_norm = MinMaxScaler().fit_transform(data_all_norm)
    if method=='Umap':
        fg_umap = umap.UMAP(n_components=3,n_neighbors=50).fit_transform(data_all_norm)
        a.uns['IMS_Umap'] = fg_umap
    elif method=='Tsne':
        fg_tsne = TSNE(n_components=3).fit_transform(data_all_norm)
        a.uns['IMS_Tsne'] = fg_tsne
    return a
