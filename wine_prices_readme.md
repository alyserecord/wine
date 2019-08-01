# Predicting Wine Price by Labels

## Have you ever picked a wine based on its label?
Picking a wine based on its label seems to be a common occurrence when picking a bottle, and it got me thinking - how does the look of the label impact the price of the wine? Does it?

**“Picking a wine by its label is slightly better than judging a book by its cover.”** *(Alex Burch, as cited in The Takeout in [How to judge a bottle of wine by its label](https://thetakeout.com/how-to-judge-a-bottle-of-wine-by-its-label-1828030852))*

<img src="/figures/wine-labels.jpg" width="800">

<sup>*Image from The Daily Meal in [Psychology of a Wine Label: Why We Buy the Bottles We Do](https://www.thedailymeal.com/psychology-wine-label-why-we-buy-bottles-we-do)*</sup>



## The Data
I collected the data by webscaping the search pages of [wine.com](https://www.wine.com/list/wine/7155). On these pages I was able to get an image of the wine label, the varietal, the origin, various ratings, the type of wine, and the price.

The wine scraping script ran on an EC2 instance on AWS and all of the wine labels were collected and then stored on an S3 bucket. The script collected image labels and metadata for about 14,600 wines.

## Image Clusters using KMeans Clustering
One of the first things I did after collecting the data was an unsupervised model to see if there was a way to cluster to wine labels based on image alone. I used KMeans to cluster the images into four clusters, and found that each cluster had a distinctive color palette and style.

<img src="/figures/no_padding_cluster0.jpg" width="600"><img src="/figures/no_padding_cluster1.jpg" width="600">
<img src="/figures/no_padding_cluster2.jpg" width="600"><img src="/figures/no_padding_cluster3.jpg" width="600">

## The Metadata
Some of the other metadata that I had to explore included reviews, price, origin, varietal, type, and other booleans such as giftable, collectible, green, screwcap, and boutique.

### Origin
The origin data had hundreds of distinct values to start since each wine region can be a small geographic area in a country or state. I first consolidated the data to countries and states, but that still left over 25 distinct values. I took a few of the top producing countries/states and looked at their price distribution in a boxplot to get a sense of the distribution of price in those regions. This was later used when modeling price.

<img src="/figures/price_by_region.jpg" width="600">

### Varietal
There were 98 varietals of wine represented in the dataset, but most of these varietals only had a handful of records. Again, I took a few of the top varietals and looked at their price distribution.

<img src="/figures/price_by_varietal.jpg" width="900">

### Ratings
Wine.com provides a number of different wine review ratings as well as customer ratings. While this data seemed like it would be incredibly useful initially, I found that only ~3,000 of the ~14,000 records had review data making it difficult to use in the price modeling.

## Principal Component Analysis
I was using 50x50 pixel images, so I had 7500 features plus metadate about each wine, so I had to do something to reduce the number of dimensions. I used Principal Component Analysis to reduce the number of features for my image data that I used to feed into my price prediction models. Below is the scree plot which shows that over 50% of the variance can be explained by just the first few principal components.

<img src="/figures/pca_scree_plot.jpg" width="600">

## Predicting Price
For predicting the price I tried both Linear Regression and Random Forest models. Overall, the performance of these models wasn't great, the highest r-squared value being around 0.49 with a Random Forest model. The the root mean squared error was approximately $45, indicating that the predicted values were not very close to the actual. I removed the top and botton 10% of the data to remove the outliers, and while the r-squared value did not improve, the root mean squared error came down to about $17.

In the models I used a combination of principal components and combinations of my metadata. I simplified origin and varietal into the top few categories and one 'other' category to reduce the number of features as seen above in the boxplots. Interestingly, the models performed best when they had all of the origin features by country, not combined into an 'other' category. 

The models also used the KMeans cluster assignment in its prediction, but I found that the KMeans cluster assignment was not very helpful in predicting price. The distributions of each cluster were very similar.

<img src="/figures/price_by_cluster.jpg" width="600">

## Conclusion
Its difficult to predict the price of wine using only the label and an a few features about the wine. So in summary, if you have access to a wine sommolier, continue to use that resource for all your wine purchasing guidance because the label will sometime mislead you! 

## Future Work
I found the analysis on the wine labels themselves (PCA and KMeans) to be very insteresting, so I would like to take this project in a different direction in the future to build a recommender model based on wine labels. I often find that I am drawn to labels that I recognize or look familiar in some way, so I think a recommender system would be interesting on this dataset.

Also, I am in the process of scraping additional information about the 14,600 wines, specifically the Winemarker Notes which 
could be used for NLP.
