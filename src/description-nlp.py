import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as sklearn_stop_words
from sklearn.decomposition import NMF
import spacy

class NLP():

    def __init__(self,text):
        self.text = text


    def lemmatizer(self):
        lemma = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
        for idx,row in enumerate(self.text):
            doc = lemma(row)
            self.text[idx] = " ".join([token.lemma_ for token in doc])

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

    def nmf_process(self):
        '''
        '''
        self.nmf = NMF(n_components=10)
        self.W = self.nmf.fit_transform(self.tfidf_matrix)
        self.H = self.nmf.components_
        self.reconsturction_err = self.nmf.reconstruction_err_

    def display_top_words(self,num_words=20):
        for topic_idx, topic in enumerate(self.H):
            print ("Topic %d:" % (topic_idx))
            print (", ".join([self.feature_names[i] for i in topic.argsort()[:-num_words - 1:-1]]))



if __name__ == '__main__':
    df = pd.read_csv('../data/cleaned_data.csv')
    arr = np.array(df['description'])
    custom_stop_words = ['view more','view','color','dish','grape',
    'year','estate','pair','ideal','flavor','variety','wine','palete',
    'vineyard','du','finish','note','great','excellent','flavor','long'
    'aroma','vintage','dense','month','nose','s','pron']
    nlp = NLP(arr)
    nlp.lemmatizer()
    nlp.count_vectorizer(custom_stop_words)
    nlp.tfidf_process()
    nlp.nmf_process()
    nlp.display_top_words()