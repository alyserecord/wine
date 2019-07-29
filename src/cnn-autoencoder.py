import pandas as pd
import numpy as np
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras import backend as K
from theano import function
import matplotlib.pyplot as plt
import kmeans 
import pickle

class Autoencoder():

    def __init__(self,model=None):
        
        if model != None:
            self.model = model

    def build_autoencoder_model(self):
            
        autoencoder = Sequential()
        
        # encoder layers
        autoencoder.add(Conv2D(128,(3,3), activation='relu', padding='same',input_shape=(64,64,3)))
        autoencoder.add(MaxPooling2D((2,2), padding = 'same'))
        autoencoder.add(Conv2D(64,(3,3), activation='relu', padding='same'))
        autoencoder.add(MaxPooling2D((2,2), padding = 'same'))
        autoencoder.add(Conv2D(32,(3,3), activation='relu', padding='same'))
        autoencoder.add(MaxPooling2D((2,2), padding = 'same'))
        autoencoder.add(Conv2D(8,(3,3), activation='relu', padding='same'))
        autoencoder.add(MaxPooling2D((2,2), padding = 'same'))

        autoencoder.add(Flatten())
        autoencoder.add(Reshape((4,4,8)))

        # decoder layers
        autoencoder.add(Conv2D(8,(3,3), activation='relu', padding='same'))
        autoencoder.add(UpSampling2D((2,2)))
        autoencoder.add(Conv2D(32,(3,3), activation='relu', padding='same'))
        autoencoder.add(UpSampling2D((2,2)))
        autoencoder.add(Conv2D(64,(3,3),activation='relu', padding='same'))
        autoencoder.add(UpSampling2D((2,2)))
        autoencoder.add(Conv2D(128,(3,3), activation='relu', padding='same'))
        autoencoder.add(UpSampling2D((2,2)))
        autoencoder.add(Conv2D(3,(3,3), activation='sigmoid', padding='same'))

        # autoencoder.summary()

        autoencoder.compile(optimizer='adam', loss='mean_squared_error')

        self.model = autoencoder


    def fit(self,train,test,batch_size,epochs):
        self.model.fit(train, train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(test,test))
        self.history = self.model.history.history

    def get_rmse(self,test):
        return self.model.evaluate(test,test)

    def predict(self,test):
        return self.model.predict(test)

    def plot_before_after(self,test,test_decoded,n=10):
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

        plt.savefig('gitfiasco.png')

    def get_layers(self,X,layer):

        batches = np.split(X,5)
        for i,batch in enumerate(batches):
            get_layer_output = K.function([self.model.layers[0].input],
                                        [self.model.layers[layer].output])
            layer_output = get_layer_output([batch])[0]

            if i == 0:
                final_layers = layer_output
            else:
                final_layers = np.vstack((final_layers,layer_output))
        self.layers = final_layers
        return self.layers

    def plot_loss(self):

        fig,ax = plt.subplots(figsize=(8,6))
        ax.plot(cnn.model.history.history['loss'])
        ax.plot(cnn.model.history.history['val_loss'])
        ax.set_title('CNN Autoencoder Model Loss')
        ax.set_ylabel('Loss')
        ax.set_xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper right')
        plt.savefig('cnn_loss.png')

    def execute_kmeans(self,n_clusters,df_filepath):
        labels,inertia = kmeans.kmeans_fit(self.layers,n_clusters=n_clusters)
        kmeans.add_labels_to_df(labels,df_filepath)

    def show_cluster(self,n_clusters,df_filepath):
        kmeans.show_cluster(n_clusters,df_filepath)


if __name__=='__main__':
    X = np.load('../data/64x64/image_array_cnn.npy')
    X = X[:500]
    train, test = train_test_split(X, test_size=0.2)

    #if building the model from scratch
    cnn = Autoencoder()
    cnn.build_autoencoder_model()
    batch_size = 100
    epochs = 2
    cnn.fit(train,test,batch_size,epochs)
    scores = cnn.get_rmse(test)
    print(scores)

    #if passing in a model
    # model = 
    # cnn = Autoencoder(model)


    X_decoded = cnn.predict(X)
    cnn.plot_before_after(X,X_decoded,10)

    layers = cnn.get_layers(X,8)

    df_filepath = '../data/64x64/sorted_df.csv'
    n_clusters = 7
    cnn.execute_kmeans(n_clusters,df_filepath)

    cnn.plot_loss()
    