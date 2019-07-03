import pandas as pd
import numpy as np 
import boto3
from skimage import io
from skimage.transform import resize
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import os
s3 = boto3.resource('s3')

import warnings
warnings.filterwarnings('ignore') 


def retrieve_images(s3_keys,bucket):
    s3_keys = ['{}.jpg'.format(x) for x in s3_keys]
    for i in s3_keys:
        print(i)
        s3.meta.client.download_file(bucket,i,'../images/{}'.format(i))


def resize_images(filepath,pixel_1,pixel_2,colors):
    images = np.zeros((len(os.listdir(filepath)),(pixel_1 * pixel_2 * colors)))
    file_names = []

    for idx,fname in enumerate(os.listdir(filepath)):
        image = io.imread(os.path.join(filepath,fname))
        image_resized = resize(image,(pixel_1,pixel_2,colors),mode='constant')
        images[idx] = image_resized.ravel()
        file_names.append(fname[:-4])

        # plt.imshow(image_resized)
        # plt.show()
    np.save('../data/{}x{}/image_array_no_padding'.format(pixel_1,pixel_2),images)
    file_names = np.array(file_names)
    np.save('../data/{}x{}/file_array_no_padding'.format(pixel_1,pixel_2),file_names)

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



def merge_and_sort(image_array_filepath,filename_array_filepath,df_filepath,pixel_1,pixel_2,colors):
    images = pd.DataFrame(np.load(image_array_filepath))
    filename = pd.DataFrame(np.load(filename_array_filepath))
    cols = ['wine']
    filename.columns = cols
    df = pd.read_csv(df_filepath)
    image_df = pd.concat([images,filename],axis=1,join_axes=[images.index])
    merged = pd.merge(image_df,df,how='inner',left_on='wine',right_on='name')
    merged = drop_outliers(merged)
    
    pixel_cols = pixel_1 * pixel_2 * colors
    onlyimages = merged.iloc[:,:pixel_cols]
    
    np.save(image_array_filepath,onlyimages.values)
    np.save('../data/{}x{}/sorted_price'.format(pixel_1,pixel_2),merged['price_bins'].values)
    
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
    pixel_1 = 50
    pixel_2 = 50
    colors = 3
    # resize_images(filepath,pixel_1,pixel_2,colors)
    # resize_images_with_padding(filepath,pixel_1,pixel_2,colors)
    merge_and_sort('../data/50x50/image_array_no_padding.npy','../data/50x50/file_array_no_padding.npy','../data/cleaned_data.csv',pixel_1,pixel_2,colors)
