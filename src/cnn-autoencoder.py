import pandas as pd
import numpy as np
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Sequential

def cnn_autoencoder(X_train):
    
    autoencoder = Sequential()
    autoencoder()

if __name__=='__main__':
    X = np.load('../data/50x50/image_array_cnn.npy')
    X = X[:10]

