from skimage.color import rgb2lab, lab2rgb
import seaborn as sns
from sklearn.preprocessing import *

import matplotlib.pyplot as plt
def plot_SIMS_view(fg_umap,save=None):
# fg_umap = a_concat[a_concat.obs['batch']=='0'].obsm['X_pca'][:,0:3]

    fg_umap_norm = MinMaxScaler().fit_transform(fg_umap)
    fg_umap_norm[:,0] = MinMaxScaler(feature_range=(0, 100)).fit_transform(fg_umap_norm[:,0][:,None])[:,0]
    fg_umap_norm[:,1] = MinMaxScaler(feature_range=(-128, 127)).fit_transform(fg_umap_norm[:,1][:,None])[:,0]
    fg_umap_norm[:,2] = MinMaxScaler(feature_range=(-128, 127)).fit_transform(fg_umap_norm[:,2][:,None])[:,0]
#     fg_umap_norm[:,2] = MinMaxScaler(feature_range=(-128, 50)).fit_transform(fg_umap_norm[:,2][:,None])[:,0]


    data_rgb = fg_umap_norm

    data_rgb_img = data_rgb.reshape(256,256,3).astype('float64')
    data_rgb_img = lab2rgb(data_rgb_img)
    sns.set(style='white')
    sns.set_color_codes('deep')
    # sns.set
    # cur_save_file = '{0}tsnemap_img_thres134_{1}_{2}.png'.format(save_path,str(bg_threshold),time_str)
    plt.figure(figsize=(10,10))
    plt.imshow(data_rgb_img)
    plt.xticks([])
    plt.yticks([])
    if save is not None:
        plt.savefig(save,transparent=False,format='png',bbox_inches='tight')

def View(a,method='Umap'):
    plot_SIMS_view(a.uns['IMS_'+method])
