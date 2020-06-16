import matplotlib.pyplot as plt
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
