import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as sklearn_stop_words
from sklearn.decomposition import NMF
import spacy
import matplotlib.pyplot as plt
import matplotlib as mlp
from sklearn.decomposition import LatentDirichletAllocation 
mlp.style.use('ggplot')


class NLP():

    def __init__(self,text):
        self.text = text


    def lemmatizer(self):
        lemma = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
        for idx,row in enumerate(self.text):
            doc = lemma(row)
            self.text[idx] = " ".join([token.lemma_ for token in doc])
        np.save('../data/lemmatized_desc',self.text)

    def count_vectorizer(self,custom_stop_words):
        '''
        '''
        stop_words = list(sklearn_stop_words) + custom_stop_words    
        self.cv = CountVectorizer(ngram_range=(1, 2),stop_words=stop_words,token_pattern='(?ui)\\b\\w*[a-z]+\\w*\\b')
        self.cv_matrix = self.cv.fit_transform(self.text)
        self.feature_names = self.cv.get_feature_names()

    def tfidf_process(self):
        '''
        '''
        self.tfidf = TfidfTransformer(use_idf=True)
        self.tfidf_matrix = self.tfidf.fit_transform(self.cv_matrix)

    def nmf_process(self,n_components=10):
        '''
        '''
        self.nmf = NMF(n_components)
        self.W = self.nmf.fit_transform(self.tfidf_matrix)
        self.H = self.nmf.components_
        self.reconsturction_err = self.nmf.reconstruction_err_
        return self.reconsturction_err

    def lda_process(self,n_components):
        self.lda = LatentDirichletAllocation(n_components=n_components,learning_method='online',max_iter=5)
        self.lda_matrix = self.lda.fit_transform(self.cv_matrix)
        self.lda_components = self.lda.components_

    def display_top_words(self,model_type,df,num_words=10):
        if model_type == 'nmf':
            matrix = self.H
        else:
            matrix = self.lda_components
        for topic_idx, topic in enumerate(matrix):
            print ("Topic %d:" % (topic_idx))
            print (", ".join([self.feature_names[i] for i in topic.argsort()[:-num_words - 1:-1]]))
            # print ("Top Wines:")
            # ex_wines = []
            # for idx in self.W[:,topic_idx].argsort()[-3:][::-1]:
            #     ex_wines.append(df.iloc[idx,1])
            # print (", ".join(ex_wines))
   
    def plot_nmf_reconstruction_err(self,n_latent_topics):
        '''
        '''
        errors = [self.nmf_process(i) for i in range(1,n_latent_topics)]
        fig, ax = plt.subplots(figsize=(8,8))
        ax.plot(range(1,n_latent_topics),errors)
        ax.set_xlabel('Number of Latent Topics')
        ax.set_ylabel('Reconstruction Error')
        plt.show()

    def plot_latent_top_hist(self):
        '''
        '''
        fig,ax = plt.subplots(4,5,figsize=(12,12))
        for idx,ax in enumerate(ax.flatten()):
            ax.hist(self.W[:,idx],bins=20)
            ax.set_title('Latent Topic: {}'.format(idx),fontsize=10)
        plt.tight_layout(pad=1.4)
        plt.show()

    def save_nmf_topics(self):
        '''
        '''
        df = pd.DataFrame(self.W)
        df.to_csv('../data/64x64/nmf_topics.csv',index=False)


if __name__ == '__main__':
    # first time read in dataframe and combine the varietal and description
    df = pd.read_csv('../data/64x64/sorted_df.csv')
    # df['new_desc'] = df['varietal'] + ' ' + df['description']
    # arr = np.array(df['new_desc'].values)

    # read in lemmatized desc+varietal
    arr = np.load('../data/lemmatized_desc.npy')

    custom_stop_words = ['view more','view','color','dish','grape',
    'year','estate','pair','ideal','flavor','variety','wine','palate',
    'vineyard','du','finish','note','great','excellent','flavor','long'
    'aroma','vintage','dense','month','nose','s','pron','blendsblend',
    'produce','make','plant','grow','valley','friend','ken','want','share',
    'town','extract','soil','section','face','bottle','acre','flow',
    'residual','change','appellation','terrace','nacional','hill',
    'believe','product','break','recall','site','real','people','parcel',
    'winemaking','standard','pior','winemaking','d','somewhat','la',
    'saint','marked','di','say','season','pass','enjoyment','little',
    'e','village','wright','fino','ready','yield','heart','region','santa',
    'cover','slowly','bay','leap','1er','original','north','translate',
    'western','definitely','label','jump','able','winery','del','farming',
    'story','couple','unmistakable','bear','trademark','depend','good',
    'blend','lee','use','day','harvest','locate','source','farm','sonoma',
    'willamette','pairs']
    nlp = NLP(arr)
    # nlp.lemmatizer()
    nlp.count_vectorizer(custom_stop_words)
    nlp.tfidf_process()
    nlp.nmf_process(45)
    nlp.display_top_words('nmf',df,20)
    nlp.save_nmf_topics()
    # nlp.plot_latent_top_hist()
    # nlp.lda_process(20)
    # nlp.display_top_words('lda')