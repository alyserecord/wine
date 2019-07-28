import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def kmeans_fit(X,n_clusters):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(X)
    return kmeans.labels_,kmeans.inertia_

def add_labels_to_df(labels,filepath):
    labels = pd.DataFrame(labels)
    df = pd.read_csv(filepath)
    labels.columns = ['kmeans_label']
    merged = pd.concat([df,labels],axis=1,join_axes=[df.index])
    merged.to_csv(filepath,index=False)

def show_cluster(labels,n_clusters):
    for cluster in range(n_clusters):
        fig,ax = plt.subplots(9,9,figsize=(16,16))
        fig.suptitle('Cluster {}'.format(cluster),fontsize=60)
        # wines = fname[labels == cluster]

        for i,ax in enumerate(ax.flatten()):
            image = Image.open('../images/{}.jpg'.format(wines[i]))
            ax.set_xticks([]) 
            ax.set_yticks([]) 
            ax.grid()
            ax.imshow(image)
        plt.savefig('../figures/no_padding_cluster{}.jpg'.format(cluster))

def elbow_plot(X,num_k):
    fig, ax = plt.subplots(figsize=(8,8))

    rss_arr = []
    for i in range(1,num_k):
        _,rss = kmeans_fit(X,i)
        rss_arr.append(rss)
    ax.plot(range(1,num_k),rss_arr)
    ax.set_xlabel('k Number of Clusters')
    ax.set_ylabel('RSS')
    plt.show()



if __name__ == '__main__':
    X = np.load('../data/64x64/image_array_cnn.npy')
    # fname = np.load('../data/50x50/file_array_2d.npy')
    # X = X[:1000,:]
    # fname = fname[:1000]
    n_clusters = 10
    labels, rss = kmeans_fit(X,n_clusters)
    
    filepath = '../data/64x64/sorted_df.csv'
    add_labels_to_df(labels,filepath)
    # show_cluster(fname,labels,n_clusters)
    # elbow_plot(X,25)