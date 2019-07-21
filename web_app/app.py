# Demonstrates Bootstrap version 3.3 Starter Template
# available here: https://getbootstrap.com/docs/3.3/getting-started/#examples

from flask import Flask, render_template,request, send_from_directory
import pandas as pd
import os
app = Flask(__name__)

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



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, threaded=True, debug=False)
