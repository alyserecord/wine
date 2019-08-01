# Wine Recommender

## Project Objective
Do you pick your wine based on region? Price? Label? Varietal? 

<img src="/figures/wine-labels.jpg" width="500">

<sup>*Image from The Daily Meal in [Psychology of a Wine Label: Why We Buy the Bottles We Do](https://www.thedailymeal.com/psychology-wine-label-why-we-buy-bottles-we-do)*</sup>

There are so many wine options it seems everyone has their own way of selecting their next bottle of wine. My goal in this project was to take a bottle of wine that a person knows he/she likes, and provide them with recommendations of other wines they might like depending on a number of features.

## The Data
I collected the data by webscaping the search pages of [wine.com](https://www.wine.com/list/wine/7155). First, summary information such as name, label, price, origin and varietal was scraped from the wine.com search pages. Then, a second webscraper collected  the winemaker’s notes in a description field about each wine.

The wine scraping script ran on an EC2 instance on AWS and all of the wine labels were collected and then stored on an S3 bucket. The script collected image labels, descriptions and metadata for about 12,200 wines.

## Architecture
A number of unsupervised models were used to create features that are utilized in the recommender. The raw label images were encoded and clustered, and the varietal and description text went through an NLP process and latent topics were identified. Those engineered features plus the wine price, origin (region), and type (red/white/sparkling) were used in the recommender. The end user would select a wine, and then receive recommendations based on their choice of wine.

<img src="/figures/architecture.jpg" width="900">

## Image Processing

A Convolutional Neural Network Autoencoder was used for dimension reduction on the wine label images. The input to the model was a 64x64x3 color image array, and the model produced a flattened 128 array before reconstructing the image. 
The below figure depicts the architecture of the CNN Autoencoder. The encoded image layer was extracted for use in a clustering model.

<img src="/figures/cnn-architecture.jpg" width="600">

KMeans clustering was performed on the encoded wine label images to produce the following wine label clusters used in the content recommender. By using 7 clusters I found that each cluster had a distinctive color palette and style.

<img src="/figures/cnn_cluster1.jpg" width="300"><img src="/figures/cnn_cluster2.jpg" width="300">
<img src="/figures/cnn_cluster3.jpg" width="300"><img src="/figures/cnn_cluster4.jpg" width="300">
<img src="/figures/cnn_cluster5.jpg" width="300"><img src="/figures/cnn_cluster6.jpg" width="300">

## Text Processing

First, the description and varietal were concatenated together. This was done because there were almost 100 different varietals making it a very large categorical feature, and there was already great deal of overlap between the varietal and the description due to some descriptions listing the grape varietals.
The text was lemmatized, numbers and special characters were removed, and custom stop words were removed. A TF-IDF matrix was procuded from the cleaned text, and a NMF was then used to extract latent topics form the vectorized descriptions and varietals.

<img src="/figures/nlp_process.jpg" width="900">

### Example Latent Topics
A total of 45 different latent topics were used in the final recommender to differential differnt varietals and flavos, but here are some example of the latent topics and the top few words in each topic:

**Buttery & Smooth Topic:** chardonnay, pear, apple vanilla, creamy, butter, hazelnut, pineapple, golden

**Meat Friendly Topic:** meat, red meat, game, pasta, grill, aged cheese, roast, stew, lamb

**Chocolate Lover Topic:** chocolate, dark chocolate, dark fruit, cherry, blackberry dark red, plum

**Refreshing Summer Wine Topic:** lemon, citrus, bright, acidity, green apple, fresh, lime, blossom, crisp

## The Recommender

Finally, all features were combined and weighted in a content recommender that uses the cosine similarity to find wines similar to the user’s selection.
The final features for the recommender include:
Image Cluster
Latent Topic Loadings
Price
Origin
Type (Red, White or Sparkling)

The recommender was deployed on an AWS EC2 instance where users can interact with the recommender searching for their favorite wines and receiving recommendations.
