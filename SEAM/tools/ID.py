import keras
from keras.models import Model
from keras.constraints import *
from keras.regularizers import *
from keras.layers import *
from keras.initializers import *
import keras.backend as K
from keras.optimizers import *



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

    logit_model = Model(SIMS_input,d4)

    pred_logit = logit_model.predict(train_x)
    rep_list = []
    for t in t_list:
        cur_representation = np.exp(pred_logit/t)
        cur_representation = cur_representation/np.sum(cur_representation,axis=1,keepdims=True)
        cur_representation = np.transpose(cur_representation)
        rep_list.append(cur_representation)
    return rep_list






def ID(a,t=5,epochs=200):
    matter_list = np.array(a.var_names)
    # SIMS_id_t_list = [5,10,15,20,25,30,35,40,50]
    SIMS_id_t_list = [t]
    train_x_tmp = a.uns['train_x']
    cell_idx = a.uns['cell_idx']
    num_cells = a.shape[0]

    HEG_list = matter_list

    HEG_col_idx = [list(matter_list).index(HEG) for HEG in HEG_list]

    netwidths=[128,128,128]

    error_threshold=0
    train_x_HEG = train_x_tmp[:,HEG_col_idx]
    train_x_preprocess = train_x_HEG


    train_x_preprocess = (train_x_HEG)/np.sum(train_x_HEG,axis=1,keepdims=True)


    rep_list = get_distil_rep(train_x_preprocess,cell_idx,num_cells,SIMS_id_t_list, verbose=False,epochs=epochs,netwidths=netwidths,low_dim=netwidths[2],error_threshold=error_threshold)
    a.obsm['ID'] = rep_list[0]
    return a
