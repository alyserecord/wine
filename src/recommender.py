import pandas as pd 
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class CosineSimilarity():

    def __init__(self,df,nmf_topics):
        '''
        Initializes the wine metadata df, NMF topics from description, and an array of the wine names.

        inputs:
        df: dataframe of wine metadata
        nmf_topics: W matrix of the NMF performed on the wine description + varietal

        outputs: none
        '''
        self.df = df
        self.nmf_topics = nmf_topics
        self.names = self.df['name']

    def prep_sorted_data(self):
        '''
        Using the initialized dataframe of wine metadata, the function removes unneed features,
        turns text features into numbers, and weights features.

        inputs: none
        outputs: none
        '''
        self.df = self.df[['origin','price','red','white','sparking','kmeans_label']]
        self.df = pd.concat([self.df,pd.get_dummies(self.df['origin'], prefix='origin').mul(2)],axis=1)
        self.df.drop(['origin'],axis=1,inplace=True)
        self.df = pd.concat([self.df,pd.get_dummies(self.df['kmeans_label'], prefix='image_cluster').mul(2)],axis=1)
        # self.df.drop(df['kmeans_label'],axis=1,inplace=True)
        self.df['price'] = self.df['price']/3
        self.df[['red','white','sparking']] = self.df[['red','white','sparking']] * 4

    def scale_nmf_clusters(self):
        '''
        Using the initialized NMF topics weights the features so that they impact the cosine similiarity
        calcuation more.

        inputs: none
        outputs: none
        '''
        self.nmf_topics = self.nmf_topics * 20


    def merge_files(self):
        '''
        Merges the dataframe of wine metadata and the NMF topics.

        inputs: none
        outputs: none
        '''
        self.merged = pd.concat([self.df,self.nmf_topics],axis=1)

    
    def get_recommendation(self,wine_name,num_rec=5):
        '''
        Given a wine name, the function will return recommendations similar to the provided wine.

        inputs: 
        wine name: name of wine the user selected for recommendations
        num_rec: the number of recommendations the function should return (default = 5)

        outputs: 
        list of wine recommenations
        '''
        wine_index = self.names[self.names == wine_name].index[0]
        similar_indices = cosine_similarity(self.merged.iloc[wine_index,:].values.reshape(1,-1),self.merged)[0,:].argsort()[-2:-num_rec-2:-1]
        items = []
        for i in similar_indices:
            items.append(self.names[i])
        return items


if __name__ == '__main__':
    df = pd.read_csv('../data/64x64/sorted_df.csv')
    nmf_topics = pd.read_csv('../data/64x64/nmf_topics.csv')

    cs = CosineSimilarity(df,nmf_topics)
    cs.prep_sorted_data()
    cs.merge_files()

    wine = 'Daniel & Julien Barraud Pouilly-Fuisse La Roche 2014'
    recommendations = cs.get_recommendation(wine,20)
    print(recommendations)
