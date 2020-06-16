import palettable
import scanpy as sc
import os

#ROOT_PATH = os.path.abspath(os.path.dirname('.'))
#ROOT_PATH = 'SEAM'
ROOT_PATH=os.path.dirname(os.path.abspath(__file__))
DATA_PATH_IMS_RAW = ROOT_PATH+'/data/raw/'
DATA_PATH_IMS_PROCESSED = ROOT_PATH+'/data/process/'
DATA_PATH_DUMP = ROOT_PATH+'/data/dump/'
sc.set_figure_params(dpi=500, color_map='viridis',dpi_save=500,transparent=True)

sc.settings.verbosity = 2
heatmap_cmp = palettable.cmocean.diverging.Balance_20.mpl_colormap
