import requests 
from bs4 import BeautifulSoup 
import numpy as np
import csv
import time
import boto3
boto3_connection = boto3.resource('s3')
s3 = boto3.client('s3')
import io
import pandas as pd


def get_all_text(soup,i):
    '''
    Input:
    BeautifulSoup for a page of wines.
    
    Output:
    List containing the wine name, size, alcohol percent, description, and reviews.
    '''
    wine=[] 
    wine.append(i)
    # find name
    wine.append(soup.find('h1', attrs = {'class':'pipName'}).text)
    # find size
    wine.append(soup.find('span',attrs = {'class':'prodAlcoholVolume_text'}).text)
    # find alcohol
    wine.append(soup.find('span',attrs = {'class':'prodAlcoholPercent_percent'}).text)
    # find winemaker notes
    wine.append(soup.find('div', attrs = {'itemprop':'description'}).text)
    #  get each review text
    rating_dict_text = {}
    try:
        for rating in soup.find('div',attrs = {'class':'viewMoreModule_text viewMoreModule-reviews'}):
            rating_dict_text[rating.find('span',attrs = {'class':'wineRatings_initials'}).text] = rating.find('div',attrs = {'class':'pipSecContent_copy'}).text
        wine.append(rating_dict_text)
    except:
        pass
    return wine


def scraper(wine_list):
    # f = csv.writer(open('../data/wine_descriptions2.csv', 'w'))
    # f.writerow(['url','name', 'size','alcohol', 'description','reviews'])
    for i in wine_list:
        URL = '{}{}'.format('https://www.wine.com',i)
        try:
            r = requests.get(URL)
            soup = BeautifulSoup(r.content, 'html5lib')
            try:
                wine = get_all_text(soup,i)
                f = csv.writer(open('../data/wine_descriptions3.csv', 'a'))
                f.writerow(wine)
            except:
            	pass
        except:
            pass
        # time.sleep(10)



if __name__ == '__main__':
    # df = pd.read_csv('../data/scraped_wine_data.csv')
    # urls = ['/product/chateau-clerc-milon-15-liter-magnum-2005/434799','/product/royal-tokaji-late-harvest-500ml-2016/414521','/product/vietti-barolo-rocche-5-liter-2015/525412']
    # urls = df['url'][1:].tolist()

    df = pd.read_csv('../data/cleaned_data.csv',index_col=0)
    urls = df[df.description.isnull()]['url'].tolist()

    scraper(urls)
