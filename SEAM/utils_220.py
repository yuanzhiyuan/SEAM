import scipy.io as sio
import numpy as np
from sklearn.preprocessing import RobustScaler,MinMaxScaler,StandardScaler,Normalizer,QuantileTransformer
from skimage.color import rgb2lab, lab2rgb
from sklearn.metrics.pairwise import *
#import magic
#import phate
from keras.models import Model
from keras.layers import Dense,Input
from sklearn.metrics.pairwise import euclidean_distances
import pickle
import time
from scipy.stats import *
from scipy.spatial.distance import *
import timeit
from scipy.sparse.csgraph import laplacian
from numpy.linalg import eig
from matplotlib.colors import LinearSegmentedColormap
from sklearn.model_selection import *
from sklearn.model_selection import cross_val_score
from sklearn.svm import *
from sklearn.neighbors import *
import pandas as pd
import hdbscan
import keras
from keras.constraints import *
from keras.regularizers import *
from keras.layers import *

from keras.initializers import *
from scipy.spatial.distance import *
from sklearn.decomposition import *
import keras.backend as K
from keras.optimizers import *
from keras.layers import Lambda,Dropout,Embedding
#from keras.utils import multi_gpu_model
from keras import regularizers
from IPython.display import SVG
from sklearn.cluster.bicluster import SpectralCoclustering
from sklearn.metrics import consensus_score
from sklearn import metrics
# from DEC import DEC
import scipy.cluster.hierarchy as hc
from sklearn.metrics import *
from keras.utils.vis_utils import model_to_dot
import os
import seaborn as sns
from sklearn.manifold import TSNE,Isomap,LocallyLinearEmbedding
import umap
from scipy.cluster.hierarchy import *
from sklearn.cluster import *
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
import tensorflow as tf
import sys
# sys.path.insert(0, '/home/yzy/software/DEC')

from scipy.stats import wasserstein_distance


def numpy2mat(numpy_mat):
    list_mat = list(map(lambda x:list(x),numpy_mat))
    return matlab.double(list_mat)


def reset_weights(model):
    session = K.get_session()
#     k=0
    for layer in model.layers: 
#         print(k)
        if hasattr(layer, 'kernel_initializer'):
            layer.kernel.initializer.run(session=session)
#             k+=1

def label2CM(label):
    label_sz = label.shape[0]
    rst = np.zeros(shape=(label_sz,label_sz))
    for i in range(label_sz):
        for j in range(i+1):
            if label[i]<0 or label[j]<0:
                continue
            if label[i]==label[j]:
                rst[i,j] = 1
                rst[j,i] = 1
    return rst
            
def get_matter_img(matter_idx,batch_num):
    data = batch_dict[batch_num]
    pos = pos_dict[batch_num]
    sorted_idx = np.argsort(pos)
    sorted_data = data[sorted_idx,matter_idx]
    return sorted_data.reshape((256,256))

def ind2ij(ind,size,axis):
    i,j=divmod(ind-1,size)
    i+=1
    j+=1
    return np.array([i,j])[axis]

def cal_square_dist(idx1,idx2,size):
    i1,j1=divmod(idx1-1,size)
    i1 +=1
    j1 +=1
    i2,j2=divmod(idx2-1,size)
    i2+=1
    j2+=1
    return (i1-i2)**2+(j1-j2)**2
def get_labeling(label,cell_idx,cell_pos):
#     y是cell-rela的细胞对应的标签
#     print('pred_y',np.unique(label))
    labeling = np.zeros(shape=(65536,1))
    b = cell_idx.copy()
    num_cells = label.shape[0]
    for i in range(num_cells):
        b[b==i+1] = label[i] + 1
#     print(cell_pos)
#     print('b',np.unique(b))
#     cell_pos = cell_pos.astype('int')
    labeling[cell_pos.astype('int')-1,0] = b
    
    return labeling
def numpy2mat(numpy_mat):
    list_mat = list(map(lambda x:list(x),numpy_mat))
    return matlab.double(list_mat)

def LRR(data_x,lmd=0.12):
    eng = matlab.engine.start_matlab()
    data_x = numpy2mat(data_x)
    eng.addpath('/data01/yzy/software/LRR/code2')
    [Z,E] = eng.solve_lrr(data_x,lmd,nargout=2)
    return np.array(Z)
def SIMLR(data_x,k):
    eng = matlab.engine.start_matlab()
    data_x = numpy2mat(data_x)
    [cur_label, S, F, ydata,alphaK,timeOurs,converge,LF] = eng.SIMLR(data_x,k,10.0,nargout=8)
    y_SIMLR = np.array(cur_label)[:,0]
    aff = np.array(S)
    eng.quit()
    return y_SIMLR,aff
def ClustRF(data_x,k):
    data_x = numpy2mat(data_x)
    eng = matlab.engine.start_matlab()
    [y_ClustRF,affinity_mat] = eng.ClustRF(data_x,'adaptive',k,nargout=2)    
    y_ClustRF = np.array(y_ClustRF)[:,0]
    affinity_mat = np.array(affinity_mat)
    eng.quit()
    return y_ClustRF-1,affinity_mat



def SSR(data_x,k):
    data_x = numpy2mat(data_x)
    eng = matlab.engine.start_matlab()
    [y_SSR,A] = eng.simplexSC(data_x,k,10.0,nargout=2)
    y_SSR = np.array(y_SSR)
    y_SSR = np.transpose(y_SSR)[:,0]
    A = np.array(A)
    eng.quit()
    return y_SSR-1,A
def RMKKM(data_x,k):
#     data_x：numpyarray；
#     k：float，e.g，2.0
    kernel_list = []
    gaussian_ts = [0.01,0.05,0.1,1,10,50,100]
    poly_as = [0,1]
    poly_bs = [2,4]

    # 先是gaussian kernel，
    max_dist = np.max(pdist(data_x,'euclidean'))
    for gaussian_t in gaussian_ts:
        delta = gaussian_t*max_dist
        gamma = 1/(2*np.square(delta))
        cur_kernel = rbf_kernel(data_x,gamma=gamma)
        kernel_list.append(cur_kernel)

    # 再是poly kernel
    for i in range(len(poly_as)):
        poly_a = poly_as[i]
        poly_b = poly_bs[i]
        cur_kernel = polynomial_kernel(data_x,coef0=poly_a,degree=poly_b)
        kernel_list.append(cur_kernel)

    #cosine kernel
    kernel_list.append(cosine_similarity(data_x))

    kernel_list = list(map(lambda m:numpy2mat(m),kernel_list))
    eng = matlab.engine.start_matlab()
    rst = eng.RMKKM(kernel_list,k,'gamma',0.5, 'maxiter', 50, 'replicates', 1,nargout=1)
    eng.quit()
    y_RMKKM=np.array(rst)[:,0]
    return y_RMKKM

def SLKE(data_x,k,gamma=0.001,mu=1):

    eng = matlab.engine.start_matlab()
    #K = cosine_similarity(data_x)
    K = polynomial_kernel(data_x,degree=1,coef0=0)
    eng.addpath('/home/yzy/software/SLKE')
    K = numpy2mat(K)
    [rst,L] = eng.SLKE( K,k,gamma,mu,nargout=2)
    rst = np.array(rst)
    L = np.array(L)

    return rst[:,0]-1,L 
        
def SC3(data_x,k,transformations=['PCA']):
    # data_x = rep
    # k=3
    # transformations = ['PCA','Laplacian']
    k = int(k)
    d_range_low = np.max([np.floor(data_x.shape[0]*0.04),1])
    d_range_high = np.ceil(data_x.shape[0]*0.07)
    d_range = np.arange(d_range_low,d_range_high+1)
    distance_metrics = ['euclidean','pearson','spearman']
    kmeans_input_pool = []
    kmeans_label_list = []
    CM = np.zeros(shape=(data_x.shape[0],data_x.shape[0]))
    for distance_metric in distance_metrics:
        if distance_metric=='euclidean':
            dist_mat = squareform(pdist(data_x,'euclidean'))
        elif distance_metric=='pearson':
            dist_mat = squareform(pdist(data_x,'correlation'))
        elif distance_metric=='spearman':
            dist_mat = 1-spearmanr(np.transpose(data_x))[0]
        for transformation in transformations:
            if transformation=='PCA':
                trans_dist_mat = PCA(n_components=int(d_range_high)).fit_transform(dist_mat)
            elif transformation=='Laplacian':
                simi_mat = np.exp(-dist_mat/(np.max(dist_mat)))
                graph_lap = laplacian(simi_mat,normed=True,return_diag=False)
                eig_value,eig_vector = eig(graph_lap)
                trans_dist_mat = eig_vector[:,np.flip(np.argsort(eig_value),axis=0)[0:int(d_range_high)]]
            for d in d_range:
                cur_input = trans_dist_mat[:,0:int(d)]
                kmeans_input_pool.append(cur_input)
                
                
    if k<-1:
#         如果k=0，就估计一个k,kmax为abs(k)
        silhouette_list = []
        kmax = np.abs(k)
        for ak in range(2,kmax+1):
            for kmeans_input in kmeans_input_pool:
                cur_y = KMeans(ak).fit_predict(kmeans_input)
#                 kmeans_label_list.append(cur_y)
                cur_CM = label2CM(cur_y)
                CM = CM + cur_CM
            CM = CM/np.max(CM)
            y_ak=AgglomerativeClustering(n_clusters=ak,affinity='precomputed',linkage='complete').fit_predict(1-CM)
            silhouette_list.append(silhouette_score(1-CM,metric='precomputed',labels = y_ak))
        silhouette_list = np.array(silhouette_list)
        opt_k = np.argmax(silhouette_list)+2
        y_SC3=AgglomerativeClustering(n_clusters=opt_k,affinity='precomputed',linkage='complete').fit_predict(1-CM)
    elif k>1:
        for kmeans_input in kmeans_input_pool:
            cur_y = KMeans(k).fit_predict(kmeans_input)
            kmeans_label_list.append(cur_y)
            cur_CM = label2CM(cur_y)
            CM = CM + cur_CM
        CM = CM/np.max(CM)
        y_SC3=AgglomerativeClustering(n_clusters=k,affinity='precomputed',linkage='complete').fit_predict(1-CM)
    else:
        print('k error')
        return None
    return y_SC3
def SIMLR(data_x,k,n_neighbors=10):
    eng = matlab.engine.start_matlab()
    data_x = numpy2mat(data_x)
    if k<-1:
#         要估计k
        kmax = np.abs(k)
        ks = list(np.arange(2,kmax+1))
        ks = matlab.double(ks)
        [k1,k2]=eng.Estimate_Number_of_Clusters_SIMLR(data_x,ks,nargout=2)
        opt_k = np.argmax(np.array(k2))+2
    elif k>1:
        opt_k = k
    else:
        print('k error')
        return None
    opt_k = float(opt_k)
    [cur_label, S, F, ydata,alphaK,timeOurs,converge,LF] = eng.SIMLR(data_x,opt_k,float(n_neighbors),nargout=8)
    y_SIMLR = np.array(cur_label)[:,0]
    S = np.array(S)
    eng.quit()
    return y_SIMLR,S
def modi_softmax(args):
    t =1
    pixel_embed,nuclei_embed = args
    
    
    
    pixel_embed = pixel_embed[:,None]
#     pixel_embed = K.l2_normalize(pixel_embed,axis=-1)
#     nuclei_embed = K.l2_normalize(nuclei_embed,axis=-1)
    
#     print('pixel_embed',pixel_embed.shape)
#     print('nuclei_embed',nuclei_embed.shape)
    minus_square = (pixel_embed*nuclei_embed/t)
#     minus_square = K.square(pixel_embed-nuclei_embed)
#     print('minus_square',minus_square.shape)
    
#     sum_minus_square = -K.sum(minus_square,axis=-1)/t
    sum_minus_square = K.sum(minus_square,axis=-1)
#     norm_exp = K.softmax(sum_minus_square)
    return sum_minus_square
    
#     print('norm_exp',norm_exp)
    
def modi_softeuc(args):
    t = 1
    p = 1
    pixel_embed,nuclei_embed = args
    
    
    
    pixel_embed = pixel_embed[:,None]
#     pixel_embed = K.l2_normalize(pixel_embed,axis=-1)
#     nuclei_embed = K.l2_normalize(nuclei_embed,axis=-1)
    
#     print('pixel_embed',pixel_embed.shape)
#     print('nuclei_embed',nuclei_embed.shape)
#     minus_square = (pixel_embed*nuclei_embed/t)
    minus_square = K.square(pixel_embed-nuclei_embed)
#     minus_square = K.abs(pixel_embed-nuclei_embed)

#     print('minus_square',minus_square.shape)
    
#     sum_minus_square = -K.sqrt(K.sum(minus_square,axis=-1))/t
    sum_minus_square = -K.pow(K.sum(minus_square,axis=-1)/t,p)

#     sum_minus_square = K.sum(minus_square,axis=-1)
    norm_exp = K.softmax(sum_minus_square)
    return norm_exp
    
    print('norm_exp',norm_exp)
def cal_kld(nuclei_embed):
#     计算nuclei embed的arccos矩阵和均匀分布的kl散度
#     也就是arccos矩阵的熵
    pass


    d1 = Dense(512,activation=activa,kernel_initializer=glorot_normal(),kernel_regularizer=l2(l2_penalty),use_bias=use_bias)(SIMS_input)
    # d1 = Dense(1024,activation=activa,kernel_initializer='random_uniform')(SIMS_input)
    # d1 = Dense(1024,activation=activa,kernel_initializer='random_uniform')(d1)


    d2 = Dense(256,activation=activa,kernel_initializer=glorot_normal(),kernel_regularizer=l2(l2_penalty),use_bias=use_bias)(d1)
    # d2 = Dense(512,activation=activa,kernel_initializer='random_uniform')(d1)
    # d2 = Dense(512,activation=activa,kernel_initializer='random_uniform')(d2)
    # d2 = Dense(256,activation='linear',kernel_initializer='random_uniform',use_bias=use_bias)(d2)


    # d3 = Dense(64,activation=activa,kernel_initializer=glorot_normal(),kernel_regularizer=l2(l2_penalty))(d2)
    # d3 = Dense(256,activation=activa,kernel_initializer='random_uniform')(d2)
    # d3 = Dense(256,activation=activa,kernel_initializer='random_uniform')(d3)


    d4 = Dense(low_dim,activation=activa,kernel_initializer=glorot_normal(),kernel_regularizer=l2(l2_penalty),use_bias=use_bias)(d2)
    # d4 = Dense(num_cells,activation='linear',kernel_initializer=glorot_normal(),kernel_regularizer=l2(l2_penalty),use_bias=use_bias)(d2)
    ####################MLP################################################################################
    centerloss_embed_layer = Embedding(num_cells, low_dim)(target_input)
    centerloss_out = Lambda(lambda x: K.sum(K.square(x[0]-x[1][:,0]), 1, keepdims=True), name='center')([d4,centerloss_embed_layer])


    embed_layer = Embedding(num_cells,low_dim,input_length=num_cells,embeddings_initializer='random_uniform')(dummy_input)
    normalized_logit_layer = Lambda(modi_softmax,name='normalized_logit_layer')([d4,embed_layer])


    softmax_out = Activation('softmax',name='softmax')(normalized_logit_layer)

    softmax_model = Model([SIMS_input,target_input,dummy_input],[softmax_out,centerloss_out])
    softmax_model.compile(optimizer=adam(),loss=['categorical_crossentropy',lambda y_true,y_pred:y_pred],loss_weights=[1,0])
    onehot_label = keras.utils.to_categorical(cell_idx-1,num_cells)
    
    reset_weights(softmax_model)
    dummy_input_data = np.tile(np.arange(num_cells),(train_x.shape[0],1))
    history=softmax_model.fit([train_x,cell_idx-1,dummy_input_data],[onehot_label,np.ones(shape=cell_idx.shape)],epochs=epochs,shuffle=True,batch_size=64,verbose=verbose)

    
    while np.abs(history.history['loss'][-1]-np.max(history.history['loss']))<=2:
        print('error')
        reset_weights(softmax_model)
        history=softmax_model.fit([train_x,cell_idx-1,dummy_input_data],[onehot_label,np.ones(shape=cell_idx.shape)],epochs=epochs,shuffle=True,batch_size=64,verbose=verbose)

    #             history=softmax_model.fit([train_x,cell_idx-1],[onehot_label,np.ones(shape=cell_idx.shape)],epochs=200,shuffle=True,batch_size=64,verbose=False)
    #         pred_logit = logit_model.predict(train_x)
#         pred_logit = logit_model.predict([train_x,dummy_input_data])
    logit_model = Model([SIMS_input,dummy_input],normalized_logit_layer)
    pred_logit = logit_model.predict([train_x,dummy_input_data])
    rep_list = []
    for t in t_list:
        cur_representation = np.exp(pred_logit/t)
        cur_representation = cur_representation/np.sum(cur_representation,axis=1,keepdims=True)
        cur_representation = np.transpose(cur_representation)
        rep_list.append(cur_representation)
    return rep_list
    # history=softmax_model.fit([train_x,cell_idx-1,dummy_input_data],[onehot_label,np.ones(shape=cell_idx.shape)],epochs=50,shuffle=True,batch_size=64)
# def get_distil_rep(logit_model,t):
    
    
def get_distil_rep(train_x,cell_idx,num_cells,t_list,epochs=50,verbose=False,activa = 'relu',dp_rate = 0.5,low_dim = 128,l2_penalty = 0,l1_penalty = 1e-5,use_bias = False,netwidths=[512,256,128],error_threshold=2):
    
    SIMS_input = Input(shape=(train_x.shape[1],))
    target_input = Input(shape=(1,))
    kernel_init_func = glorot_normal()

    d1 = Dense(netwidths[0],activation=activa,kernel_initializer=kernel_init_func,kernel_regularizer=l2(l2_penalty),use_bias=use_bias)(SIMS_input)
    # d1 = Dense(1024,activation=activa,kernel_initializer='random_uniform')(SIMS_input)
    # d1 = Dense(1024,activation=activa,kernel_initializer='random_uniform')(d1)


    d2 = Dense(netwidths[1],activation=activa,kernel_initializer=kernel_init_func,kernel_regularizer=l2(l2_penalty),use_bias=use_bias)(d1)
    # d2 = Dense(512,activation=activa,kernel_initializer='random_uniform')(d1)
    # d2 = Dense(512,activation=activa,kernel_initializer='random_uniform')(d2)
    d2 = Dense(netwidths[2],activation='linear',kernel_initializer=kernel_init_func,use_bias=use_bias)(d2)


    # d3 = Dense(64,activation=activa,kernel_initializer=glorot_normal(),kernel_regularizer=l2(l2_penalty))(d2)
    # d3 = Dense(256,activation=activa,kernel_initializer='random_uniform')(d2)
    # d3 = Dense(256,activation=activa,kernel_initializer='random_uniform')(d3)


    # d4 = Dense(low_dim,activation=activa,kernel_initializer=glorot_normal(),kernel_regularizer=l2(l2_penalty))(d3)
    d4 = Dense(num_cells,activation='linear',kernel_initializer=kernel_init_func,kernel_regularizer=l2(l2_penalty),use_bias=use_bias)(d2)
    ####################MLP################################################################################
    centerloss_embed_layer = Embedding(num_cells, low_dim)(target_input)
    centerloss_out = Lambda(lambda x: K.sum(K.square(x[0]-x[1][:,0]), 1, keepdims=True), name='center')([d2,centerloss_embed_layer])



    softmax_out = Activation('softmax',name='softmax')(d4)

    softmax_model = Model([SIMS_input,target_input],[softmax_out,centerloss_out])
    softmax_model.compile(optimizer=adam(),loss=['categorical_crossentropy',lambda y_true,y_pred:y_pred],loss_weights=[1,0])
    onehot_label = keras.utils.to_categorical(cell_idx-1,num_cells)
    # reset_weights(softmax_model)


    history=softmax_model.fit([train_x,cell_idx-1],[onehot_label,np.ones(shape=cell_idx.shape)],epochs=epochs,shuffle=True,batch_size=64)
    while np.abs(history.history['loss'][-1]-np.max(history.history['loss']))<=error_threshold:
        print('error')
        reset_weights(softmax_model)
#         history=softmax_model.fit([train_x,cell_idx-1,dummy_input_data],[onehot_label,np.ones(shape=cell_idx.shape)],epochs=epochs,shuffle=True,batch_size=64,verbose=verbose)
        history=softmax_model.fit([train_x,cell_idx-1],[onehot_label,np.ones(shape=cell_idx.shape)],epochs=epochs,shuffle=True,batch_size=64)

    #             history=softmax_model.fit([train_x,cell_idx-1],[onehot_label,np.ones(shape=cell_idx.shape)],epochs=200,shuffle=True,batch_size=64,verbose=False)
    #         pred_logit = logit_model.predict(train_x)
#         pred_logit = logit_model.predict([train_x,dummy_input_data])
#     logit_model = Model([SIMS_input,dummy_input],normalized_logit_layer)
    logit_model = Model(SIMS_input,d4)

    pred_logit = logit_model.predict(train_x)
    rep_list = []
    for t in t_list:
        cur_representation = np.exp(pred_logit/t)
        cur_representation = cur_representation/np.sum(cur_representation,axis=1,keepdims=True)
        cur_representation = np.transpose(cur_representation)
        rep_list.append(cur_representation)
    return rep_list

def get_distil_rep2(train_x,cell_idx,num_cells,t_list,epochs=50,verbose=False,activa = 'relu',dp_rate = 0.5,low_dim = 128,l2_penalty = 0,l1_penalty = 1e-5,use_bias = False,netwidths=[512,256,128],error_threshold=2):

    SIMS_input = Input(shape=(train_x.shape[1],))
    target_input = Input(shape=(1,))
    kernel_init_func = glorot_normal()

    d1 = Dense(netwidths[0],activation=activa,kernel_initializer=kernel_init_func,kernel_regularizer=l2(l2_penalty),use_bias=use_bias)(SIMS_input)
    # d1 = Dense(1024,activation=activa,kernel_initializer='random_uniform')(SIMS_input)
    # d1 = Dense(1024,activation=activa,kernel_initializer='random_uniform')(d1)


    d2 = Dense(netwidths[1],activation=activa,kernel_initializer=kernel_init_func,kernel_regularizer=l2(l2_penalty),use_bias=use_bias)(d1)
    # d2 = Dense(512,activation=activa,kernel_initializer='random_uniform')(d1)
    # d2 = Dense(512,activation=activa,kernel_initializer='random_uniform')(d2)
    d2 = Dense(netwidths[2],activation='linear',kernel_initializer=kernel_init_func,use_bias=use_bias)(d2)


    # d3 = Dense(64,activation=activa,kernel_initializer=glorot_normal(),kernel_regularizer=l2(l2_penalty))(d2)
    # d3 = Dense(256,activation=activa,kernel_initializer='random_uniform')(d2)
    # d3 = Dense(256,activation=activa,kernel_initializer='random_uniform')(d3)


    # d4 = Dense(low_dim,activation=activa,kernel_initializer=glorot_normal(),kernel_regularizer=l2(l2_penalty))(d3)
    d4 = Dense(num_cells,activation='linear',kernel_initializer=kernel_init_func,kernel_regularizer=l2(l2_penalty),use_bias=use_bias)(d2)
    ####################MLP################################################################################
    # centerloss_embed_layer = Embedding(num_cells, low_dim)(target_input)
    # centerloss_out = Lambda(lambda x: K.sum(K.square(x[0]-x[1][:,0]), 1, keepdims=True), name='center')([d2,centerloss_embed_layer])



    softmax_out = Activation('softmax',name='softmax')(d4)

    softmax_model = Model([SIMS_input],[softmax_out])
    softmax_model.compile(optimizer=adam(),loss=['categorical_crossentropy'],loss_weights=[1])
    onehot_label = keras.utils.to_categorical(cell_idx-1,num_cells)
    # reset_weights(softmax_model)


    history=softmax_model.fit([train_x],[onehot_label],epochs=epochs,shuffle=True,batch_size=64)
    while np.abs(history.history['loss'][-1]-np.max(history.history['loss']))<=error_threshold:
        print('error')
        reset_weights(softmax_model)
#         history=softmax_model.fit([train_x,cell_idx-1,dummy_input_data],[onehot_label,np.ones(shape=cell_idx.shape)],epochs=epochs,shuffle=True,batch_size=64,verbose=verbose)
        # history=softmax_model.fit([train_x,cell_idx-1],[onehot_label,np.ones(shape=cell_idx.shape)],epochs=epochs,shuffle=True,batch_size=64)
        history=softmax_model.fit([train_x],[onehot_label],epochs=epochs,shuffle=True,batch_size=64)

    #             history=softmax_model.fit([train_x,cell_idx-1],[onehot_label,np.ones(shape=cell_idx.shape)],epochs=200,shuffle=True,batch_size=64,verbose=False)
    #         pred_logit = logit_model.predict(train_x)
#         pred_logit = logit_model.predict([train_x,dummy_input_data])
#     logit_model = Model([SIMS_input,dummy_input],normalized_logit_layer)
    logit_model = Model(SIMS_input,d4)

    pred_logit = logit_model.predict(train_x)
    rep_list = []
    for t in t_list:
        cur_representation = np.exp(pred_logit/t)
        cur_representation = cur_representation/np.sum(cur_representation,axis=1,keepdims=True)
        cur_representation = np.transpose(cur_representation)
        rep_list.append(cur_representation)
    return rep_list




def get_diff_matter_idx(plot_diff_i,plot_diff_j,MAPlot_df,select_mode,arg,matter_list):
# plot_diff_i = 1
# plot_diff_j = 2
    print('matter_list2',len(matter_list))
    n = arg
    wbr_threshold = arg
    ave_threshold = 0
    # select_mode = 0
    vs_str = 'cluster {i} vs cluster {j}'.format(i=plot_diff_i,j=plot_diff_j)
    idx_list = []
    MAPlot_df_ij = MAPlot_df[(MAPlot_df['VS Clusters']==vs_str) & (MAPlot_df['Average Expression']>=ave_threshold)]
    if select_mode ==0:
    #     nlargest
        print('matter_list1',len(matter_list))
        matters_nlargest = MAPlot_df_ij.nlargest(n,'WB_Ratio').matter
        matters_nsmallest = MAPlot_df_ij.nsmallest(n,'WB_Ratio').matter
        print('matters_nlargest',matters_nlargest)
        for matter in matters_nlargest:
            idx_list.append(matter_list.index(matter))
        for matter in matters_nsmallest:
            idx_list.append(matter_list.index(matter))
        print('idx_list',idx_list)

    else:
        #wbr_threshold
        condition1 = (MAPlot_df_ij['WB_Ratio']>=wbr_threshold)
        condition2 = (MAPlot_df_ij['WB_Ratio']<=-wbr_threshold)

        for matter in MAPlot_df_ij[condition1 | condition2].matter:
            matter_idx = matter_list.index(matter)
            idx_list.append(matter_idx)
    unique_idx_list = list(set(idx_list))
    return unique_idx_list
def simulate_data(batch_idx,mode,fold_var,mean_shift,change_dims):
    cell_idx = cell_dict[batch_idx]
    sample_x = batch_dict[batch_idx]
    sample_x_rst = sample_x.copy().astype('float64')
    cell_idx_rst = cell_idx.copy()
    if mode==0:
        
        sample_x_rst = sample_x
    elif mode==1:
#         随机选change_dims个维，把每个细胞的这些维按N(mean_shift,fold_var)重采样
#         fold_var,mean_shift,change_dims
#         selected_dims = np.random.randint(0,sample_x.shape[1],change_dims)
        selected_dims = np.random.permutation(sample_x.shape[1])[0:change_dims]
        for i in range(np.max(cell_idx)):
            cur_cell = sample_x[cell_idx==i+1,:]
#             cur_cell_mean = np.mean(sample_cell[:,selected_dims],axis=0)
#             cur_cell_std = np.std(sample_cell[:,selected_dims],axis=0)
            for j in selected_dims:
#                 print(i,j)
                cur_cell_mean = np.mean(cur_cell[:,j])+mean_shift
                cur_cell_std = np.std(cur_cell[:,j])*fold_var
                
                cur_simu = np.random.normal((cur_cell_mean), (cur_cell_std), cur_cell.shape[0])
                for k in range(10):
                    if np.min(cur_simu)>=0:
                        break
                    cur_simu = np.random.normal((cur_cell_mean), (cur_cell_std), cur_cell.shape[0])
#                 print(np.min(cur_simu<0))
                if np.min(cur_simu)<0:
                    cur_simu = cur_cell[:,j]
                else:
#                     print(cur_simu)
                    pass
#                     print(np.min(cur_cell[j]))
#                 print(cur_simu)
                    
                sample_x_rst[cell_idx==i+1,j] = cur_simu
    elif mode==2:
#         add zero pixels
#         fold_var:添加的数字
#         mean_shift:每个细胞添加的百分比
#     给每个细胞增加mean_shift比例的fold_var像素点
        selected_dims = np.random.permutation(sample_x.shape[1])[0:change_dims]
        
        for i in range(np.max(cell_idx)):
            cur_area = np.sum(cell_idx==i+1)
#             cell_area_list.append(np.sum(cell_idx==i+1))
            cur_add_num = int(cur_area*mean_shift)+1
            sample_x_rst = np.vstack([sample_x_rst,fold_var*np.ones(shape=(cur_add_num,sample_x.shape[1]))])
#         sample_x_rst[:,selected_dims] = np.vstack([sample_x_rst[:,selected_dims],])
            cell_idx_rst = np.hstack([cell_idx_rst,(i+1)*np.ones(shape=(cur_add_num,))])
            cell_idx_rst = cell_idx_rst.astype('int')

    elif mode==3:
        
#         在每个cell中，按比例把像素点的profile乘2
#         fold_var:乘几（2）；mean_shift:百分比
        for i in range(np.max(cell_idx)):
            cur_cell = sample_x[cell_idx==i+1,:]
            num_change = int(mean_shift*cur_cell.shape[0])
            selected_pixels_idx = np.random.permutation(cur_cell.shape[0])[0:num_change]
            changed_cell = cur_cell.copy()
            changed_cell[selected_pixels_idx,:] = changed_cell[selected_pixels_idx,:]*fold_var
            sample_x_rst[cell_idx==i+1,:] = changed_cell
    elif mode==4:
        
#         把每个像素点随机乘0～fold_var的均匀分布
        uniform_sample = np.random.uniform(0,fold_var,size=(sample_x.shape[0],1))
        sample_x_rst = sample_x_rst*uniform_sample
    elif mode==5:
#         随机选change_dim个维度，把每个像素点的这个维度乘fold_var
        selected_dims = np.random.permutation(sample_x.shape[1])[0:change_dims]
        sample_x_rst[:,selected_dims]*=fold_var
    elif mode==6:
#         指定change_dims（一个list）维度，乘fold_var
        for change_dim_idx in range(len(change_dims)):
            cur_change_dim = change_dims[change_dim_idx]
            cur_fold_change = fold_var[change_dim_idx]
            sample_x_rst[:,cur_change_dim]*=cur_fold_change
        
    elif mode==7:
#         multimodel：对每个细胞，选一个维度，一半值0一半乘2；选同样维度，一半值0.5，一半值1.5；原始数据。分三类，需要选的维度很大。
#     选两个维度，分四类，00，01，10，11
# 这个函数输入change_dims;fold_var一个list，比如[2]代表0和2; mean_shift代表几几开选像素点
        for i in range(np.max(cell_idx)):
            cur_cell_area = np.sum(cell_idx==i+1)
            cur_cell_area_1 = int(cur_cell_area/2)
            cur_cell_perm = np.random.permutation(cur_cell_area)
            cur_cell_part1_idx = cur_cell_perm[0:cur_cell_area_1]
            cur_cell_part2_idx = cur_cell_perm[cur_cell_area_1:]
            for change_dim_idx in range(len(change_dims)):
                cur_change_dim = change_dims[change_dim_idx]
                cur_fold_change = fold_var[change_dim_idx]
                cur_fold_change_1 = cur_fold_change
                cur_fold_change_2 = 2-cur_fold_change
#                 print(np.sum(sample_x_rst[cell_idx==i+1,cur_change_dim][cur_cell_part1_idx]))
#                 print(np.sum(sample_x_rst[cell_idx==i+1,cur_change_dim][cur_cell_part2_idx]))
                cur_change = sample_x_rst[cell_idx==i+1,cur_change_dim]
                cur_change[cur_cell_part1_idx]*=cur_fold_change_1
                cur_change[cur_cell_part2_idx]*=cur_fold_change_2
                sample_x_rst[cell_idx==i+1,cur_change_dim] = cur_change
#                 sample_x_rst[cell_idx==i+1,cur_change_dim][cur_cell_part1_idx]*=cur_fold_change[0]
#                 sample_x_rst[cell_idx==i+1,cur_change_dim][cur_cell_part2_idx]*=cur_fold_change[1]
#                 print(np.sum(sample_x_rst[cell_idx==i+1,cur_change_dim][cur_cell_part1_idx]))
#                 print(np.sum(sample_x_rst[cell_idx==i+1,cur_change_dim][cur_cell_part2_idx]))
#                 cur_cell_part1 = sample_x[cell_idx==i+1,cur_change_dim][cur_cell_part1_idx]*cur_fold_change[0]
#                 cur_cell_part2 = sample_x[cell_idx==i+1,cur_change_dim][cur_cell_part2_idx]*cur_fold_change[1]
#                 print(i,'part1',np.sum(cur_cell_part1),cur_fold_change[0])
#                 print(i,'part2',np.sum(cur_cell_part2),cur_fold_change[1])
                
#                 sample_x_rst[cell_idx==i+1,cur_change_dim][cur_cell_part1_idx] = cur_cell_part1
#                 sample_x_rst[cell_idx==i+1,cur_change_dim][cur_cell_part2_idx] = cur_cell_part2
#                 print(np.sum(sample_x_rst[cell_idx==i+1,cur_change_dim][cur_cell_part2_idx]))
            
    return sample_x_rst,cell_idx_rst

# 有label情况的处理

# 1.147 0.025
def get_train_data(data_mat_filename,mode,norm,batch_num_list=[1]):
#     mode分为:'none','median','total'
#     norm分为:'none',standard','l1','l2'
#     batch_num_list = [3,5]
    # 5行
    # 1,2,3,4,5,6
    # batch_num = 4

    
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
#         plt.plot(sort_val)
#         plt.show()


    #     train_x = train_x/np.percentile(train_x,80,axis=1,keepdims=True)
    #     train_x = np.log(train_x+1)


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


