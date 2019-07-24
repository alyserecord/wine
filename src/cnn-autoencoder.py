# import pandas as pd
import numpy as np
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras import backend as K
from theano import function
import matplotlib.pyplot as plt
# import kmeans 

def build_autoencoder_model():
    
    autoencoder = Sequential()
    
    # encoder layers
    autoencoder.add(Conv2D(32,(3,3), activation='relu', padding='same',input_shape=(50,50,3)))
    autoencoder.add(MaxPooling2D((2,2), padding = 'same'))
    autoencoder.add(Conv2D(8,(3,3), activation='relu', padding='same'))
    autoencoder.add(MaxPooling2D((2,2), padding = 'same'))

    autoencoder.add(Flatten())
    autoencoder.add(Reshape((13,13,8)))

    # decoder layers
    autoencoder.add(Conv2D(8,(3,3), activation='relu', padding='same'))
    autoencoder.add(UpSampling2D((2,2)))
    autoencoder.add(Conv2D(32,(3,3),activation='relu', padding='same'))
    autoencoder.add(UpSampling2D((2,2)))
    autoencoder.add(Conv2D(3,(3,3), activation='sigmoid', padding='same'))

    # autoencoder.summary()

    autoencoder.compile(optimizer='adam', loss='mean_squared_error')

    return autoencoder

def fit(model,train,test,batch_size,epochs):
    model.fit(train, train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(test,test))

def plot_before_after(test,test_decoded,n=10):
    plt.figure(figsize=(n*2, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(X[i])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(test_decoded[i])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    # plt.savefig('test3.png')

def get_layer(model,x,layer):

    get_layer_output = K.function([model.layers[0].input],
                                  [model.layers[layer].output])
    layer_output = get_layer_output([x])[0]

    return layer_output


def cluster_kmeans(encoded_layer,n_clusters):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(encoded_layer)
    return kmeans.labels_,kmeans.inertia_

def show_cluster(fname,labels,n_clusters):
    for cluster in range(n_clusters):
        fig,ax = plt.subplots(9,9,figsize=(16,16))
        fig.suptitle('Cluster {}'.format(cluster),fontsize=60)
        wines = fname[labels == cluster]
        for i,ax in enumerate(ax.flatten()):
            image = Image.open('../images/{}.jpg'.format(wines[i]))
            ax.set_xticks([]) 
            ax.set_yticks([]) 
            ax.grid()
            ax.imshow(image)
        plt.savefig('../figures/no_padding_cluster{}.jpg'.format(cluster))



if __name__=='__main__':
    X = np.load('../data/50x50/image_array_cnn.npy')
    # fname = 
    X = X[:500]
    train, test = train_test_split(X, test_size=0.2)


    model = build_autoencoder_model()

    batch_size = 500
    epochs = 1
    
    fit(model,train,test,batch_size,epochs)

    scores = model.evaluate(test, test)
    print(scores)

    test_decoded = model.predict(X)

    plot_before_after(test,test_decoded,10)

    # # instantiate callbacks
    # tensorboard = TensorBoard(log_dir='./autoencoder_logs', histogram_freq=2, batch_size=batch_size, write_graph=True, write_grads=True, write_images=True)
    # earlystopping = EarlyStopping(monitor='val_loss', patience=2)