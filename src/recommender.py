import pandas as pd 
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class CosineSimilarity():

    def __init__(self,df,nmf_topics):
        self.df = df
        self.nmf_topics = nmf_topics
        self.names = self.df['name']

    def prep_sorted_data(self):
        self.df = self.df[['origin','price','red','white','sparking','kmeans_label']]
        self.df = pd.concat([self.df,pd.get_dummies(self.df['origin'], prefix='origin').mul(2)],axis=1)
        self.df.drop(['origin'],axis=1,inplace=True)
        self.df = pd.concat([self.df,pd.get_dummies(self.df['kmeans_label'], prefix='image_cluster').mul(2)],axis=1)
        # self.df.drop(df['kmeans_label'],axis=1,inplace=True)
        self.df['price'] = self.df['price']/3
        self.df[['red','white','sparking']] = self.df[['red','white','sparking']] * 4

    def scale_nmf_clusters(self):
        self.nmf_topics = self.nmf_topics * 20


    def merge_files(self):
        self.merged = pd.concat([self.df,self.nmf_topics],axis=1)

    def generate_matrix(self):
        self.similarity_matrix = cosine_similarity(self.merged)

    def get_recommendation(self,wine_name,num_rec=5):
        wine_index = self.names[self.names == wine_name].index[0]
        similar_indices = self.similarity_matrix[wine_index].argsort()[-2:-num_rec-2:-1]
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
    cs.generate_matrix()

    wine = 'Daniel & Julien Barraud Pouilly-Fuisse La Roche 2014'
    recommendations = cs.get_recommendation(wine,20)
    print(recommendations)