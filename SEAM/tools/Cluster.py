import SIMLR
import scanpy as sc
import time
def run_SIMLR(a,c=8,rep='ID'):
    if rep=='ID':
        input_rep = a.obsm['ID']
    elif rep=='Mean':
        input_rep = a.X
    
    start_main = time.time()
#     input_rep = SIMLR.helper.fast_pca(input_rep,100)
    print(input_rep.shape,c)
    simlr = SIMLR.SIMLR_LARGE(c, 10, 0) ###This is how we initialize an object for SIMLR. the first input is number of rank (clusters) and the second input is number of neighbors. The third one is an binary indicator whether to use memory-saving mode. you can turn it on when the number of cells are extremely large to save some memory but with the cost of efficiency.
    
    S, F,val, ind = simlr.fit(input_rep)
    print('Successfully Run SIMLR! SIMLR took %f seconds in total\n' % (time.time() -         start_main))
    pred_y = simlr.fast_minibatch_kmeans(F,c)
    print('done!')
    a.obs['SIMLR'] =pred_y.astype('int').astype('str')
    a.obs['SIMLR'] = a.obs['SIMLR'].astype('category')
    
    return a

def Cluster(a,method,cluster_param,rep='ID'):
    if method=='SIMLR':
        return run_SIMLR(a,c=cluster_param,rep=rep)
    elif method=='Louvain':
        start_main = time.time()
#         sc.pp.neighbors(a,use_rep=rep,metric='cosine',n_neighbors=15)
        sc.tl.louvain(a,resolution=cluster_param)
        print('Successfully Run Louvain! Louvain took %f seconds in total\n' % (time.time() -         start_main))
        
    elif method=='Leiden':
        start_main = time.time()
#         sc.pp.neighbors(a,use_rep=rep,metric='cosine',n_neighbors=15)
        sc.tl.leiden(a,resolution=cluster_param)
        print('Successfully Run Leiden! Leiden took %f seconds in total\n' % (time.time() -         start_main))
        
        
    return a
        








         
