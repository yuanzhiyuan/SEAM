import scanpy as sc

def Umap(a,rep='ID'):
    if rep=='ID':
        sc.pp.neighbors(a,use_rep='ID',metric='cosine',n_neighbors=15)
    else:
        sc.pp.neighbors(a,n_neighbors=15)
    sc.tl.umap(a)
    print('Sucessfully run Umap!')
    return a
        
        
    
