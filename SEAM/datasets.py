import numpy as np
import pandas as pd
import anndata as ad
import scipy.io as sio
import pickle
from .settings import *

def get_train_data(data_mat_filename,mode,norm,batch_num_list=[1]):



    original_data = {}
    cell_related_data = {}
    data_mat = sio.loadmat(data_mat_filename)
    data_mat=data_mat['data_mat']



    num_features = data_mat.shape[1]-3
    batch_dict = {}

    label_dict = {}
    cell_dict = {}
    pos_dict = {}

    for i in range(1):

            cur_data = data_mat[data_mat[:,0]==i+1,3:num_features+3]


            batch_dict[i+1] = cur_data

            cell_dict[i+1] = data_mat[data_mat[:,0]==i+1,1]
            cur_batch_idx = data_mat[data_mat[:,0]==i+1,2]
            label_dict[i+1] = np.ones(shape=cur_batch_idx.shape)


            pos_dict[i+1] = cur_batch_idx
    original_data['batch_dict'] = batch_dict
    original_data['cell_dict'] = cell_dict
    original_data['label_dict'] = label_dict
    original_data['pos_dict'] = pos_dict






    top_n_var = 250
    train_x_all = None
    cell_idx_all = None
    cell_type_all = None
    cell_pos_all = None
    batch_idx_all = None
    num_cells_all = 0


    for batch_num in batch_num_list:
        train_x = batch_dict[batch_num]
        # train_x = eval('batch_dict_{norm_type}[batch_num]'.format(norm_type=norm_type))
        # train_x = batch_dict[batch_num]
        cell_idx = cell_dict[batch_num]
        cell_type = label_dict[batch_num]
        cell_pos = pos_dict[batch_num]
        # batch_FE = FE_dict[batch_num]
        cell_related_ind = (cell_idx!=0)

        num_cells = int(np.max(cell_idx))
        # num_cells = 2
        train_x = train_x[cell_related_ind,:]

        cell_idx = cell_idx[cell_related_ind]
        cell_type = cell_type[cell_related_ind]
        # cell_type = np.ones(shape=cell_idx.shape)
        cell_pos = cell_pos[cell_related_ind]





        var_li = []
        normed_var_li = []
        for i in range(train_x.shape[1]):
            cur_col = train_x[:,i]
        #     cur_col= cur_row/np.sum(cur_col)
        #     cur_entropy = entropy(cur_col)
            cur_var = np.var(cur_col)
            cur_normed_var = cur_var/np.mean(cur_col)
        #     entropy_li.append(cur_entropy)
            var_li.append(cur_var)
            normed_var_li.append(cur_normed_var)
        # entropy_li = np.array(entropy_li)
        var_li = np.array(var_li)
        normed_var_li = np.array(normed_var_li)
        sort_ind = np.flip(np.argsort(normed_var_li),axis=0)
        sort_val = np.flip(np.sort(normed_var_li),axis=0)
        
        if train_x_all is None:
            train_x_all = train_x
        else:
            train_x_all = np.vstack([train_x_all,train_x])
        if cell_idx_all is None:
            cell_idx_all = cell_idx
        else:
            cell_idx_all = np.hstack([cell_idx_all,cell_idx+np.max(cell_idx_all)])
        if cell_type_all is None:
            cell_type_all = cell_type
        else:
            cell_type_all = np.hstack([cell_type_all,cell_type])
        if cell_pos_all is None:
            cell_pos_all = cell_pos
        else:
            cell_pos_all = np.hstack([cell_pos_all,cell_pos])
        if batch_idx_all is None:
            batch_idx_all = batch_num*np.ones(shape=(cell_idx.shape))
        else:
            batch_idx_all = np.hstack([batch_idx_all,batch_num*np.ones(shape=(cell_idx.shape))])


    train_x = train_x_all
    cell_idx = cell_idx_all
    cell_type = cell_type_all
    cell_pos = cell_pos_all
    batch_idx = batch_idx_all
    num_cells = np.max(cell_idx)
    if mode=='none':
        train_x = train_x
    elif mode=='median':
        train_x = train_x/np.percentile(train_x,50,axis=1,keepdims=True)
        train_x = np.log(train_x+1)
    elif mode=='total':
        train_x = train_x/np.sum(train_x,axis=1,keepdims=True)
        train_x = np.log(train_x+1)

    if norm=='standard':
        train_x = StandardScaler().fit_transform(train_x)
    elif norm=='l1':
        train_x = Normalizer(norm='l1').fit_transform(train_x)
    elif norm=='l2':
        train_x = Normalizer(norm='l2').fit_transform(train_x)
    elif norm=='none':
        train_x = train_x

    cell_related_data['train_x'] = train_x
    cell_related_data['cell_idx'] = cell_idx
    cell_related_data['cell_type'] = cell_type
    cell_related_data['cell_pos'] = cell_pos
    cell_related_data['batch_idx'] =batch_idx
    cell_related_data['num_cells'] = num_cells
    return original_data,cell_related_data



def load_raw_SIMS(data):
    data_mat_filename_temp = DATA_PATH_IMS_PROCESSED+'{0}/cut/rst/datamat.mat'
    matter_list_filename_temp=DATA_PATH_IMS_PROCESSED+'{0}/preprocess/matters_candidate.pkl'

#     data = 'P6_neg1_low0_None_auto'
    test_sample_temp=DATA_PATH_IMS_PROCESSED+'{0}/preprocess/test_samples.mat'



    matter_list_filename = matter_list_filename_temp.format(data)
    data_mat_filename = data_mat_filename_temp.format(data)
    test_sample_filename = test_sample_temp.format(data)
    test_sample_all = sio.loadmat(test_sample_filename)['test_samples']
    mode='none'
    norm='none'
    [original_data,cell_related_data]=get_train_data(data_mat_filename,mode,norm,batch_num_list=[1])
    train_x=cell_related_data['train_x']
    cell_idx=cell_related_data['cell_idx']
    cell_pos=cell_related_data['cell_pos']
    num_cells = np.max(cell_idx)
    matter_list = pickle.load(open(matter_list_filename,'rb'))
    matter_list = np.array(matter_list)

    return train_x,cell_idx,cell_pos,matter_list,num_cells,test_sample_all


def get_mean_representation(train_x,cell_idx,num_cells):
    train_x_tmp = train_x.copy()
    train_x_median = (train_x_tmp+1)/(np.percentile(train_x_tmp,50,axis=1,keepdims=True)+1)
    train_x_total = train_x/np.sum(train_x,axis=1,keepdims=True)
    train_x_median = np.log(train_x_median+1)
    train_x_total = np.log(train_x_total+1)
    # train_x_A = (train_x+1)/(train_x[:,matter_list==134.06]+1)
    sum_profile_list_median = []
    sum_profile_list_total = []

    max_profile_list_median = []
    max_profile_list = []
    mean_profile_list_median = []
    max_profile_list_total = []
    mean_profile_list_total = []
    mean_profile_list=[]
    # mean_profile_list_A=[]
    # max_profile_list_A = []
    for i in range(num_cells):
            mean_profile_list_median.append(np.mean(train_x_median[cell_idx==i+1,:],axis=0))
            max_profile_list_median.append(np.max(train_x_median[cell_idx==i+1,:],axis=0))
            max_profile_list.append(np.max(train_x[cell_idx==i+1,:],axis=0))
    #         mean_profile_list_A.append(np.mean(train_x_A[cell_idx==i+1,:],axis=0))
    #         max_profile_list_A.append(np.max(train_x_A[cell_idx==i+1,:],axis=0))

            sum_profile_list_median.append(np.sum(train_x_median[cell_idx==i+1,:],axis=0))
            mean_profile_list.append(np.mean(train_x[cell_idx==i+1,:],axis=0))
            mean_profile_list_total.append(np.mean(train_x_total[cell_idx==i+1,:],axis=0))
            max_profile_list_total.append(np.max(train_x_total[cell_idx==i+1,:],axis=0))
            sum_profile_list_total.append(np.sum(train_x_total[cell_idx==i+1,:],axis=0))

    mean_profile_list_median = np.array(mean_profile_list_median)
    max_profile_list_median = np.array(max_profile_list_median)
    mean_profile_list_total = np.array(mean_profile_list_total)
    max_profile_list_total = np.array(max_profile_list_total)
    sum_profile_list_median = np.array(sum_profile_list_median)
    sum_profile_list_total = np.array(sum_profile_list_total)
    mean_profile_list = np.array(mean_profile_list)
    max_profile_list = np.array(max_profile_list)
    # mean_profile_list_A = np.array(mean_profile_list_A)
    # max_profile_list_A = np.array(max_profile_list_A)
    return mean_profile_list_median


def load_dataset_raw(data):
    train_x,cell_idx,cell_pos,matter_list,num_cells,test_sample_all = load_raw_SIMS(data)
    mean_profile_list_median = get_mean_representation(train_x,cell_idx,num_cells)
    
    in_X = mean_profile_list_median
    # g = map(str,range(in_X.shape[1]))
    g = map(str,matter_list)
    Genes = []
    None_idx = 0
    Genes = g
    # obs_name must be str
    obs_name = list(map(str,range(in_X.shape[0])))
    obs = pd.DataFrame(index=obs_name)

    # var_name must be str
    var = pd.DataFrame(index=Genes)

    #     var['Genes'] = Genes
    a = ad.AnnData(in_X,  obs=obs,var=var, dtype='float32')
    a.uns['cell_idx'] = cell_idx
    a.uns['cell_pos'] = cell_pos
    a.uns['IMS'] = test_sample_all
    a.uns['train_x'] = train_x
#     a.uns['test_sample_all'] = 
#     a.uns['rep_list'] = rep_list
    return a


def load_dataset_processed(data):
    data_dump_path = DATA_PATH_DUMP+'{0}/data.h5ad'.format(data)
    
    a = ad.read_h5ad(data_dump_path)
    return a