from ..utils import *
import matplotlib.pyplot as plt


def Cut(a,method='dot'):
    if method=='dot':
        fig,ax = plt.subplots(1,1,figsize=(4,4))
        sc.pl.embedding(a,basis='spatial',ax=ax)
    elif method=='mask':
        pred_y = np.ones(shape=(a.shape[0],))
        cmp = ['w']
        plot_label_image(a,pred_y,cmp,save=None,mask=None,figsize=(5,5),anno=False,ifshow=True)

