import requests 
from bs4 import BeautifulSoup 
import numpy as np
import csv
import time
from PIL import Image
import boto3
boto3_connection = boto3.resource('s3')
s3 = boto3.client('s3')
import io

def flags(item):
    '''
    Input:
    Html for one wine.
    
    Output:
    All the booleans for the wine (icons on the page). For example: Red Wine, White Wine, Sparking Wine, Collectible, etc.
    '''
    class_attr = ['icon icon-glass-red prodAttr_icon prodAttr_icon-redWine','icon icon-glass-white prodAttr_icon prodAttr_icon-whiteWine', 'icon icon-champagne prodAttr_icon prodAttr_icon-champagne','icon icon-collectible prodAttr_icon prodAttr_icon-collectible','icon icon-gift prodAttr_icon prodAttr_icon-giftable','icon icon-greenWine prodAttr_icon prodAttr_icon-greenWine','icon icon-screwcap prodAttr_icon prodAttr_icon-screwcap','icon icon-boutique prodAttr_icon prodAttr_icon-boutique']
    attr_list = []
    for attr in class_attr:
        try:
            check = item.find('li',attrs = {'class': attr})['title']
            if check != None:
                attr_list.append(1)
        except:
            attr_list.append(0)
    return attr_list

def get_all_attr(soup,i):
    '''
    Gets all the attributes on the page for each wine. Calls the flags function for all boolean for a particular wine.

    Input:
    BeautifulSoup for a page of wines.
    
    Output:
    List of all the wines and the attributes for each respective wine.
    '''
    wines=[] 

    items = soup.find('ul', attrs = {'class':'prodList'}) 

    for item in items.findAll('li', attrs = {'class':'prodItem'}):
        wine = []
        name = item.find('span', attrs = {'itemprop':'name'})['title']
        if 'Gift Set' in name or 'Gift Box' in name or 'Case' in name or 'Trio' in name or 'Collection' in name: 
            pass
        else:
            wine.append(name)
            wine.append(item.find('a',attrs = {'class':'prodItemInfo_link'},href=True)['href'])
            image = item.find('source',attrs ={'class':'prodItemImage_image-tablet'})
            if image is not None:
                image_url = image['srcset'].split('1x,')[1]
                image_url = image_url[1:-3]

                # to get the full image instead of the tablet thumbnail:
                # image = item.find('img')
                # image_url = image['src']
                
                try:
                    img = Image.open(requests.get('http://www.wine.com/{}'.format(image_url), stream = True).raw)        
                    buffer = io.BytesIO()
                    img.save(buffer, "JPEG")
                    buffer.seek(0) # rewind pointer back to start
                    s3.put_object(Bucket='winelabelimages',Key='{}.jpg'.format(name),Body=buffer,ContentType='image/jpeg')
                except:
                    pass
            wine.append(item.find('span',attrs = {'class':'prodItemInfo_varietal'}).text)
            wine.append(item.find('span',attrs = {'class':'prodItemInfo_originText'}).text)
            rating_dict = {}
            for rating in item.find('ul',attrs = {'class':'wineRatings_list'}):
                rating_dict[rating.find('span',attrs= {'class','wineRatings_initials'}).text] = rating.find('span',attrs= {'class','wineRatings_rating'}).text
            wine.append(rating_dict)
            wine.append(item.find('span',attrs = {'class':'averageRating_average'}).text)
            wine.append(item.find('span',attrs = {'class':'averageRating_number'}).text)
            wine.append(item.find('span',attrs = {'class':'productPrice_price-regWhole'}).next)
            wine.append(item.find('span',attrs = {'class':'productPrice_price-regFractional'}).next)
            wine.append(i)
            wine = wine + flags(item)
            wines.append(wine)
    return wines
        
def save_wines(wines):
    #change the w to a a to not overwrite
    for wine in wines:
        f = csv.writer(open('scraped_wine_data.csv', 'a'))
        f.writerow(wine)


def scraper(num):
    # f = csv.writer(open('scraped_wine_data.csv', 'w'))
    # f.writerow(['name', 'url','varietal', 'origin','rating_dict','avg_rating','rating_count','price_dollars','price_cents','page','red','white','sparking','collectible','giftable','green','screwcap','boutique'])
    for i in range(131,num):
        URL = '{}{}{}'.format('https://www.wine.com/list/wine/7155/',i,'?sortBy=mostInteresting')
        r = requests.get(URL)
        soup = BeautifulSoup(r.content, 'html5lib')
        #call the function that will scape all the infor for each wine on the page
        wines = get_all_attr(soup,i)
        # call the function to save the data 
        save_wines(wines)
        time.sleep(15)



if __name__ == '__main__':
    # URL = "https://www.wine.com/list/wine/7155/2"   
    # r = requests.get(URL) 
    # soup = BeautifulSoup(r.content, 'html5lib')
    # wines = get_all_attr(soup)
    # save_wines(wines)
    scraper(650)
