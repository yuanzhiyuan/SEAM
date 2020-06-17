import scanpy as sc

def Umap(a,rep='ID'):
    if rep=='ID':
        sc.pp.neighbors(a,use_rep='ID',metric='cosine',n_neighbors=15)
    else:
        sc.pp.neighbors(a,n_neighbors=15)
    a = sc.tl.umap(a,copy=True)
    print('Sucessfully run Umap!')
    return a
        
        
    
