import pandas as pd
import numpy as np 
# import boto3
from skimage import io
from skimage.transform import resize
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import os
# s3 = boto3.resource('s3')

import warnings
warnings.filterwarnings('ignore') 


def retrieve_images(s3_keys,bucket):
    '''
    Given a list of S3 keys and bucket name, retrieves all of the files
    and saves in an images directory.

    Input: 
    Array of S3 Bucket Keys
    Bucket Name
    '''
    s3_keys = ['{}.jpg'.format(x) for x in s3_keys]
    for i in s3_keys:
        print(i)
        s3.meta.client.download_file(bucket,i,'../images/{}'.format(i))


def resize_images(filepath,pixel_1,pixel_2,colors,saved_filename,flatten=True):
    '''
    Resizes image in a directory and saves a numpy array of the resized
    image and a numpy array of the file names.

    Inputs:
    filepath: directory filepath for the desired images
    pixel_1 x pixel_2 x colors: Desired image size
    saved_filename: name to append to the name and pixel numpy arrays
    flatten: if true the image array will be flattened (default), if false the image array will retain shape
    '''
    if flatten == True:
        images = np.zeros((len(os.listdir(filepath)),(pixel_1 * pixel_2 * colors)))
    else:
        images = np.zeros((len(os.listdir(filepath)), pixel_1, pixel_2,colors))

    file_names = []
    for idx,fname in enumerate(os.listdir(filepath)):
        image = io.imread(os.path.join(filepath,fname))
        image_resized = resize(image,(pixel_1,pixel_2,colors),mode='constant')
        if flatten == True:
            images[idx] = image_resized.ravel()
        else:
            images[idx] = image_resized
        file_names.append(fname[:-4])

        # plt.imshow(image_resized)
        # plt.show()
    np.save('../data/{}x{}/image_array_{}'.format(pixel_1,pixel_2,saved_filename),images)
    file_names = np.array(file_names)
    np.save('../data/{}x{}/file_array_{}'.format(pixel_1,pixel_2,saved_filename),file_names)

def resize_images_with_padding(filepath,pixel_1,pixel_2,colors):
    images = np.zeros((len(os.listdir(filepath)),(pixel_1 * pixel_2 * colors)))
    file_names = []

    for idx,fname in enumerate(os.listdir(filepath)):
        image = Image.open(os.path.join(filepath,fname))
        old_size = image.size
        ratio = pixel_1/max(old_size)
        new_size = ([int(x*ratio) for x in old_size])
        image = image.resize(new_size,Image.ANTIALIAS)
        image_resized = Image.new("RGB", (pixel_1, pixel_2),(255,255,255))
        image_resized.paste(image, ((pixel_1-new_size[0])//2,
                            (pixel_2-new_size[1])//2))
        image_resized = np.array(image_resized)

        images[idx] = image_resized.ravel()
        file_names.append(fname[:-4])


        # plt.imshow(image_resized)
        # plt.show()
    np.save('../data/{}x{}/image_array'.format(pixel_1,pixel_2),images)
    file_names = np.array(file_names)
    np.save('../data/{}x{}/file_array'.format(pixel_1,pixel_2),file_names)



def merge_and_sort(image_array_filepath,filename_array_filepath,df_filepath,pixel_1,pixel_2,colors,cnn=True):
    '''
    Takes in arrays filepaths for images pixels, filenames, and a dataframe. Merges the three datasets so
    so that the image can be acessed by name and the df and images can be used in models together. 
    Resaves the image pixels after sorting, creates a price bins array, and resaves the sorted dataframe.
    '''
    if cnn == True:
        images = pd.DataFrame(row.flatten() for row in np.load(image_array_filepath))
    else:
        images = pd.DataFrame(np.load(image_array_filepath))
    filename = pd.DataFrame(np.load(filename_array_filepath))
    cols = ['wine']
    filename.columns = cols
    df = pd.read_csv(df_filepath)
    image_df = pd.concat([images,filename],axis=1,join_axes=[images.index])
    merged = pd.merge(image_df,df,how='inner',left_on='wine',right_on='name')
    # merged = drop_outliers(merged)
    
    #re-save the image array after merging all files to ensure image pixels are in correct order
    pixel_cols = pixel_1 * pixel_2 * colors
    onlyimages = merged.iloc[:,:pixel_cols]
    if cnn == True:
        onlyimages = onlyimages.values.reshape(onlyimages.shape[0], 64, 64,3)
        np.save(image_array_filepath,onlyimages)
    else:
        np.save(image_array_filepath,onlyimages.values)
    
    # save an array price bins only for PCA analysis
    # np.save('../data/{}x{}/sorted_price'.format(pixel_1,pixel_2),merged['price_bins'].values)

    # save a new copy of the dataframe that is sorted according to the image pixel array
    cols_to_skip = pixel_1*pixel_2*colors + 2
    subset= merged.iloc[:,cols_to_skip:]
    subset.to_csv('../data/{}x{}/sorted_df.csv'.format(pixel_1,pixel_2),index=False)

def drop_outliers(df):
    df = df[df['price']<500]
    # df = df[df['price']>13.99]
    # df = df[df['price']>10]
    return df


if __name__ == '__main__':
    # bucket = 'winelabelimages'
    # s3_keys = pd.read_csv('../data/s3_keys.csv')
    # s3_keys = s3_keys.iloc[:,1]
    # # retrieve_images(s3_keys,bucket)p

    filepath = '../images/'
    pixel_1 = 64
    pixel_2 = 64
    colors = 3
    # resize_images_with_padding(filepath,pixel_1,pixel_2,colors)

    # for regular kmeans without cnn:
    # resize_images(filepath,pixel_1,pixel_2,colors,'2d',flatten=True)
    # merge_and_sort('../data/50x50/image_array_2d.npy','../data/50x50/file_array_2d.npy','../data/cleaned_data.csv',pixel_1,pixel_2,colors,cnn=False)

    # for kmeans with ccn
    resize_images(filepath,pixel_1,pixel_2,colors,'cnn',flatten=False)
    merge_and_sort('../data/64x64/image_array_cnn.npy','../data/64x64/file_array_cnn.npy','../data/cleaned_data.csv',pixel_1,pixel_2,colors,cnn=True)