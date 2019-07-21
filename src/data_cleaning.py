import pandas as pd
import numpy as np 
import boto3
s3 = boto3.resource('s3')
import ast


class WineCleaner():

    def __init__(self):
        self.labels = []
    
    def get_image_names(self,bucket):
        '''
        Creates a list of the file names on the S3 bucket.
        
        Input:
        S3 Bucket Name
        
        Output:
        None. Appends to the list of labels initialized with the class.
        '''
        s3_bucket = s3.Bucket(bucket)
        for i in s3_bucket.objects.all():
            self.labels.append(i.key[:-4])
        return self.labels 

    def drop_bad_data(self,df):
        '''
        Drops duplicate record, drops bad data without origin or containing '/'

        Input:
        Dataframe of wine data

        Output:
        Dataframe without invalid records

        '''
        df.drop_duplicates(subset='name',keep='first',inplace=True)
        df.drop(df.head(1).index,inplace=True)
        df.drop(df.tail(1).index,inplace=True)
        df = df[~pd.isnull(df['origin'])]
        df = df[~df['name'].str.contains('/')]
        return df
 
    def drop_records_missing_image(self,df):
        '''
        Takes a dataframe of wine data and looks to see if there is an image on 
        S3 for each record, dropping the records that do not have an image.

        Input:
        Dataframe of wine data

        Output:
        Dataframe only records that have an image in S3

        '''
        df['in_s3'] = 0
        for idx,row in enumerate(df['name']):
            if row in self.labels:
                df.loc[idx,'in_s3'] = 1
        df = df[df['in_s3']!=0]
        df.drop(['in_s3'],axis=1,inplace=True)
        return df

    def expand_ratings(self, df):
        df['rating_dict'].replace('NaN','{}',inplace = True)
        df['rating_dict'].replace('','{}',inplace = True)
        df['rating_dict'] = df.rating_dict.fillna('{}')
        df['rating_dict'] = df['rating_dict'].apply(ast.literal_eval)
        ratings = new_df['rating_dict'].apply(pd.Series)
        df = pd.concat([new_df, ratings], axis=1)
        df.drop(['rating_dict'],axis=1,inplace=True)
        return df

    def remove_non_wine(self,df):
        '''
        Removes the non wine data from the wine dataframe.

        Input:
        Dataframe of wine data

        Output:
        Dataframe of wine data with non-wine data removed
        '''
        mask = df['varietal'].isin(['Stemware & Decanters','In Box Glassware', 'Serve & Preserve', 'Mixed Collections', 'Corkscrews', 'In Box Accessory', 'Entertaining', 'Wine Chillers', 'Wine Storage'])
        df = new_df[~mask]
        return df
       
    def merge_dollars_cents(self,df):
        df['price_cents'].replace('\n','00',regex=True,inplace=True)
        df['price_cents'].replace(['nan','00    '],'00',inplace=True)
        df['price_cents'] = df['price_cents'].astype(int)
        df['price_dollars'].replace(',','',regex=True,inplace=True)
        df['price_dollars'] = df['price_dollars'].astype(int)
        df['price'] = df['price_dollars'] + (df['price_cents']/100)
        df.drop(['price_dollars','price_cents'],axis=1,inplace=True)
        return df

    def get_urls(self,df):
        return df['url']

    def get_keys(self,df):
        s3_keys = df['name']
        s3_keys.to_csv('../data/s3_keys.csv',header=False)
        return s3_keys

    def string_replace(self,df,col,str_lst):
        for i in str_lst:
            df[col][df[col].str.contains(i)] = i
        return df

    def drop_unneed_columns(self,df,col_list):
        df.drop([col_list],axis=1,inplace=True)
        return df

    def bin_prices(self,df):
        bins = [0,20,50,100,np.inf]
        names = ['<20','20-50','50-100','100+']
        df['price_bins'] = pd.cut(df['price'],bins,labels=names)
        return df

    def merge_descriptions(self,df,desc_df):
        desc_df = desc_df[['name','description']]
        merged = pd.merge(df,desc_df,how='left',left_on='name',right_on='name',left_index=True)
        merged = merged[merged['description'] != ' View More']
        merged = merged[~merged['description'].isnull()]
        return merged

    def save_df(self,df,filepath):
        df.to_csv(filepath)


if __name__ =='__main__':
    df = pd.read_csv('../data/scraped_wine_data.csv')
    desc_df = pd.read_csv('../data/wine_descriptions.csv')
    
    wine = WineCleaner()
    bucket = 'winelabelimages'
    wine.get_image_names(bucket)
    new_df = wine.drop_records_missing_image(df)
    new_df = wine.drop_bad_data(new_df)
    # new_df = wine.expand_ratings(new_df)
    new_df = wine.remove_non_wine(new_df)
    new_df = wine.merge_dollars_cents(new_df)
    # s3_keys = wine.get_keys(new_df)
    new_df = wine.bin_prices(new_df)
    col = 'origin'
    str_lst = ['California','France','Italy','Oregon','South Africa',
    'Spain','Australia','Washington','Japan','Austria','Greece',
    'Portugal','Chile','Argentina','New Zealand','Uruguay',
    'Other U.S.','Germany','Hungary','Canada','Israel','England',
    'Croatia','Lebanon','Slovenia','Macedonia','China']
    new_df = wine.string_replace(new_df,col,str_lst)
    new_df = wine.merge_descriptions(new_df,desc_df)
    wine.save_df(new_df,'../data/cleaned_data.csv')