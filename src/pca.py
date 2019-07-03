import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def create_pca(X,n_components=2):

    # ss = StandardScaler
    # X = ss.fit_transform(X)
    
    pca = PCA(n_components)
    principal_components = pca.fit_transform(X)
    return pca, principal_components

def add_pcs_to_df(pca,filepath,num_pc):
    pc = pd.DataFrame(principal_components[:,:num_pc])
    df = pd.read_csv(filepath)
    pc.columns = ['pc1','pc2','pc3','pc4','pc5','pc6','pc7']
    merged = pd.concat([df,pc],axis=1,join_axes=[df.index])
    merged.to_csv(filepath,index=False)


def plot_pca(principal_components,price):
    fig, ax = plt.subplots(1,1,figsize=(16,16))
    price = pd.DataFrame(price)

    ax.set_xlabel("Principal Component 1", fontsize=20)
    ax.set_ylabel("Principal Component 2", fontsize=20)
    ax.set_title('PCA Plot', fontsize=28)

    color_map = {'20-50':'b', '50-100':'y', '<20':'c', '100+':'k'}
    colors = price[0].map(color_map)
    c = np.array(colors)

    plt.scatter(principal_components[:,0],principal_components[:,1],s=.75,c=c)
    # ax.set_xlim(-5000,0)
    # ax.set_ylim(-2000,2000)
    plt.savefig('../figures/pca_price.jpg')


def scree_plot(pca):
    num_components = pca.n_components_
    ind = np.arange(num_components)
    vals = pca.explained_variance_ratio_
    fig,ax = plt.subplots(1,1,figsize=(10,10))
    ax.plot(ind, vals)
    ax.scatter(ind, vals, s=50)
    for i in range(num_components):
        ax.annotate(r"{:2.2f}".format(vals[i]), 
                   (ind[i]+0.2, vals[i]+0.005), 
                   va="bottom", 
                   ha="center", 
                   fontsize=14)
    ax.set_xlabel("Principal Component", fontsize=24)
    ax.set_ylabel("Variance Explained", fontsize=24)
    ax.set_title('Scree Plot for Principal Components', fontsize=28)
    plt.savefig('../figures/pca_scree_plot.jpg')

if __name__ == '__main__':
    X = np.load('../data/50x50/image_array_no_padding.npy')
    pca, principal_components = create_pca(X,10)
    price = np.load('../data/50x50/sorted_price.npy')
    plot_pca(principal_components,price)
    scree_plot(pca)
    add_pcs_to_df(pca,'../data/50x50/sorted_df.csv',7)
