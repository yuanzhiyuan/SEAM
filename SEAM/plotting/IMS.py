import matplotlib.pyplot as plt
import numpy as np
def get_mz_img(a,mz):
    m_list = a.var_names
    img_flatten = a.uns['IMS'][:,m_list==mz]
    img_square = img_flatten.reshape(256,256)
    return img_square

def IMS(a,mz):
    mz_img = get_mz_img(a,mz)
    plt.imshow(mz_img)  
    plt.title(mz+' m/z')
    plt.show()
def IMS(a,mz_list,n_cols=4):
#     mz_list = list(a.var_names[:10])
#     n_cols = 4
    n_rows = int(np.ceil(len(mz_list)/n_cols))
    fig,axes = plt.subplots(n_rows,n_cols)
    if len(mz_list)<=n_cols:
        axes = axes[:,None].transpose()
    for i in range(n_cols*n_rows):
        cur_row = int(i/n_cols)
        cur_col = int(i%n_cols)
        if i<len(mz_list):
            cur_mz = mz_list[i]
            mz_img = get_mz_img(a,cur_mz)
            axes[cur_row][cur_col].imshow(mz_img)
            axes[cur_row][cur_col].axis('off')
            axes[cur_row][cur_col].set_title(cur_mz+' m/z')
        if i>=len(mz_list):
            axes[cur_row][cur_col].axis('off')



