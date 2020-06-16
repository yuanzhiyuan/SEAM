import numpy as np
from scipy.spatial.distance import *
from scipy.stats import wasserstein_distance



def get_m_sc(a_use,pval_thre,cls,method='pval_topk'):
#     a_use = a_m_hepa
#     pval_thre = 0.01
    sc.tl.rank_genes_groups(a_use,n_genes=a_use.shape[1],groupby=cls)
    
    rec2mat_fun = lambda rec: np.vstack([np.array(list(m)) for m in rec])
    name_mat = a_use.uns['rank_genes_groups']['names']
    pval_mat = a_use.uns['rank_genes_groups']['pvals_adj']
    score_mat = a_use.uns['rank_genes_groups']['scores']
    
    
    name_mat = rec2mat_fun(name_mat)
    pval_mat = rec2mat_fun(pval_mat)
    score_mat = rec2mat_fun(score_mat)
    n_cls = name_mat.shape[1]

    rst_list = []
    for i in range(n_cls):
        if method=='pval_thre':
            cur_idx = (pval_mat[:,i]<=pval_thre)
        elif method=='pval_topk':
            kth_p = np.sort(pval_mat[:,i])[pval_thre]
            cur_idx = (pval_mat[:,i]<=kth_p)
        elif method=='score_thre':
            cur_idx = (score_mat[:,i]>=pval_thre)
        elif method=='score_topk':
            kth_score = np.flip(np.sort(score_mat[:,i]))[pval_thre]
            cur_idx = (score_mat[:,i]>=kth_score)
            
        cur_m = name_mat[cur_idx,i]
        rst_list.append(cur_m)
        print(cur_m.shape[0])
    a_use.uns[cls+'_mz'] = rst_list
    return a_use







def get_dist_mat_emd(a,method='emd'):

    train_x = a.uns['train_x']
    num_cells = a.shape[0]
    num_features = a.shape[1]
    cell_idx = a.uns['cell_idx']
    cell_pixel_dict = {}
    pixel_count = []
    for i in range(num_cells):
        cur_pixels = train_x[cell_idx==i+1,:]
        cell_pixel_dict[i] = cur_pixels
        pixel_count.append(cur_pixels.shape[0])

    dist_mat = np.zeros(shape=(num_features,num_cells,num_cells))
    for k in range(num_features):
        if k%10==0:
            print(k)
#         print(k)
        if method=='emd':
            for i in range(num_cells):
                for j in range(num_cells):
                    cur_dist = wasserstein_distance(cell_pixel_dict[i][:,k],cell_pixel_dict[j][:,k])
        #             cur_dist = euclidean(np.mean(cell_pixel_dict[i][:,k]),np.mean(cell_pixel_dict[j][:,k]))
        #             print(k,i,j,cur_dist)
#                     cur_dist = np.mean(cell_pixel_dict[i][:,k])-np.mean(cell_pixel_dict[j][:,k])
                    dist_mat[k,i,j] = cur_dist
        elif method=='euclidean':
            cur_dist_mat = squareform(pdist(a.X[:,k][:,None]))
            dist_mat[k,:,:] = cur_dist_mat
    a.uns['feature_wise_distmat'] = dist_mat
    return a

def get_wbr(dist_mat,pred_list):
    wbr_list = []
    for i in range(dist_mat.shape[0]):
        cur_dist_mat = dist_mat[i,:,:]
        within_sum_1 = np.sum(dist_mat[i,:,:][pred_list==1,:][:,pred_list==1])
        within_sum_0 = np.sum(dist_mat[i,:,:][pred_list==0,:][:,pred_list==0])    
        between_sum_1 = np.sum(dist_mat[i,:,:][pred_list==1,:][:,pred_list==0])
        between_sum_0 = np.sum(dist_mat[i,:,:][pred_list==0,:][:,pred_list==1])  
        wbr = (within_sum_1+within_sum_0)/(between_sum_1+between_sum_0)
        wbr_list.append(wbr)
    return np.array(wbr_list)

def get_wbr_mat(a_use,cls):
#     a_use = a
# #     wbr_thre = 10
#     cls = 'SIMLR'
#     method='topk'
    dist_mat = a_use.uns['feature_wise_distmat']
    unique_labels = np.unique(a_use.obs[cls])
    wbr_mat = np.zeros(shape=(a_use.shape[1],unique_labels.shape[0]))

    for i in range(wbr_mat.shape[1]):
        cur_label = unique_labels[i]
        cur_pred = a_use.obs[cls].copy().astype('str')
        cur_pred[cur_pred!=cur_label] = -1
        cur_pred[cur_pred==cur_label] = 0
        cur_pred = -cur_pred
        cur_wbr_list = get_wbr(dist_mat,cur_pred)
        wbr_mat[:,i] = cur_wbr_list

    a_use.uns['rank_genes_groups_emd'] = {
        'scores':wbr_mat

    }
    return a_use


def get_m_emd(a_use,thre,cls,method='topk',alg='emd'):
    a_use = get_dist_mat_emd(a_use,method=alg)
    a_use = get_wbr_mat(a_use,cls)
    
    score_mat = a_use.uns['rank_genes_groups_emd']['scores']
    n_cls = score_mat.shape[1]
    name_list = a_use.var_names
    rst_list = []
    for i in range(n_cls):
        if method=='thre':
            cur_idx = (score_mat[:,i]<=thre)
        elif method=='topk':
            kth_p = np.sort(score_mat[:,i])[thre]
            cur_idx = (score_mat[:,i]<=kth_p)
        
            
        cur_m = np.array(name_list[cur_idx])
        rst_list.append(cur_m)
        print(cur_m.shape[0])
    a_use.uns[cls+'_mz_emd'] = rst_list
    return a_use

 





