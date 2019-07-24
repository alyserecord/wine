# Demonstrates Bootstrap version 3.3 Starter Template
# available here: https://getbootstrap.com/docs/3.3/getting-started/#examples

from flask import Flask, render_template,request, send_from_directory
import pandas as pd
import os
app = Flask(__name__)
import sys
sys.path.insert(0, '../src')
from recommender import CosineSimilarity

# home page
@app.route('/', methods=['GET','POST'])
def index():
    df = pd.read_csv('../data/50x50/sorted_df.csv')
    return render_template('index.html', table = df)

@app.route("/search_results", methods=['GET','POST'])
def search_results():
    if request.method=='POST':
        selected_origin = request.form.get('origin')
        selected_varietal = request.form.get('varietal')
        selected_price = request.form.get('price')
        df = pd.read_csv('../data/50x50/sorted_df.csv')
        selected = df[(df['origin']==selected_origin) & (df['varietal']==selected_varietal) & (df['price_bins']==selected_price)]
        try:
            subset = selected.sample(12)
        except:
            subset = selected
        if subset.shape[0] > 0:
            subset['name'] = subset['name'].astype(str) + '.jpg'
            subset = subset['name'].tolist()
        else:
            subset = []
    return render_template("search_results.html",image_list = subset)


MEDIA_FOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'images')
MEDIA_FOLDER = '../images/'
@app.route('/images/<path:filename>')
def download_file(filename):
    return send_from_directory(MEDIA_FOLDER, filename)

def get_wines(selected_wine):
    df = pd.read_csv('../data/50x50/sorted_df.csv')
    nmf_topics = pd.read_csv('../data/50x50/nmf_topics.csv')    

    cs = CosineSimilarity(df,nmf_topics)
    cs.prep_sorted_data()
    cs.scale_nmf_clusters()
    cs.merge_files()
    cs.generate_matrix()

    return cs.get_recommendation(selected_wine,20)  


@app.route('/recomendations', methods=['GET','POST'])
def recommendations():
    df = pd.read_csv('../data/50x50/sorted_df.csv')
    selected_wine = request.args.get('type')
    selected_wine_att = []
    selected_wine_att.append(df[df['name']==selected_wine[:-4]]['varietal'].values[0])
    selected_wine_att.append(df[df['name']==selected_wine[:-4]]['origin'].values[0])
    selected_wine_att.append(df[df['name']==selected_wine[:-4]]['price'].values[0])
    selected_wine_att.append(df[df['name']==selected_wine[:-4]]['kmeans_label'].values[0])
    selected_wine_att.append(df[df['name']==selected_wine[:-4]]['description'].values[0])
    wines = get_wines(selected_wine[:-4])
    # print(wines)
    items = []
    for wine in wines:
        lst = []
        lst.append('{}.jpg'.format(wine))
        lst.append(wine)
        lst.append(df[df['name']==wine]['price'].values[0])
        lst.append(df[df['name']==wine]['varietal'].values[0])
        lst.append(df[df['name']==wine]['origin'].values[0])
        lst.append(df[df['name']==wine]['kmeans_label'].values[0])        
        lst.append(df[df['name']==wine]['description'].values[0])
        items.append(lst)
    return render_template('recommendations.html',img_name = selected_wine, selected_wine_att = selected_wine_att, recs = items)



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, threaded=True, debug=False)
