# Predicting Wine Price by Labels

## Have you ever picked a wine based on its label?
Picking a wine based on its label seems to be a common occurrence when picking a bottle, and it got me thinking - how does the look of the label impact the price of the wine? Does it?

![blank labels](/figures/wine-labels.jpg)

<sup>*Image from The Daily Meal in [Psychology of a Wine Label: Why We Buy Bottles We Do](https://www.thedailymeal.com/psychology-wine-label-why-we-buy-bottles-we-do)*</sup>


**“Picking a wine by its label is slightly better than judging a book by its cover.”** *(Alex Burch, as cited in The Takeout in [How to judge a bottle of wine by its label](https://thetakeout.com/how-to-judge-a-bottle-of-wine-by-its-label-1828030852))*


## The Data
I collected the data by webscaping the search pages of [wine.com](https://www.wine.com/list/wine/7155). On these pages I was able to get an image of the wine label, the varietal, the origin, various ratings, the type of wine, and the price.

The wine scraping script ran on an EC2 instance on AWS and all of the wine labels were collected and then stored on an S3 bucket. The script collected image labels and metadata for about 14,600 wines.

## Image Clusters using KMeans Clustering
One of the first things I did after collecting the data was an unsupervised model to see if there was a way to cluster to wine labels based on image alone. I used KMeans to cluster the images into four clusters, and found that each cluster had a distinctive color pallete and style.

![cluster0](/figures/no_padding_cluster0.jpg)![cluster1](/figures/no_padding_cluster1.jpg)
![cluster1](/figures/no_padding_cluster2.jpg)![cluster1](/figures/no_padding_cluster3.jpg)

## The Other Metadata
Some of the other metadata that I had to explore included reviews, price, origin, varietal, type, and other booleans such as giftable, collectible, green, screwcap, and boutique.

### Origin
The origin data had hundreds of distinct values to start since each wine region can be a small geographic area in a country or state. I first consolidated the data to countries and states, but that still left over 25 distinct values. I took a few of the top producing countries/states and looked at their price distribution in a boxplot to get a sense of the distribution of price in those regions. This was later used when modeling price.

mg src="/figures/price_by_origin.jpg" width="900">

### Varietal
There were 98 varietals of wine represented in the dataset, but most of these varietals only had a handful of records. Again, I took a few of the top varietals and looked at their price distribution.

<img src="/figures/price_by_varietal.jpg" width="900">

### Ratings
Wine.com provides a number of different wine review ratings as well as customer ratings. While this data seemed like it would be incredibly useful initially, I found that only ~3,000 of the ~14,000 records had review data making it difficult to use in the price modeling.

## Principal Component Analysis
I was using 50x50 pixel images, so I had 7500 features plus metadate about each wine, so I had to do something to reduce the number of dimensions. I used Principal Component Analysis to reduce the number of features for my image data that I used to feed into my price prediction models. Below is the scree plot which shows that over 50% of the variance can be explain by just the first few principal components.

<img src="/figures/pca_scree_plot.jpg" width="900">

## Predicting Price
For predicting the price I tried both Linear Regression and Random Forest models. Overall, the preformance of these models wasn't great, the highest r-squared value being around 0.48 and the root mean squared error approximately $45. Interestingly the Linear Regression model out performed the Random Forest model.

I used a combination of principal components, type, features about the origin and (simplified into the top few and 'other' categories). Interestingly, the model performed best when it had all of the origin features by country, not combined into an 'other' category. The models also used the KMeans cluster assignment in its prediction.

I found that the KMeans cluster assignment was not very helpful in predicting price. The distributions of each cluster were very similiar.

<img src="/figures/price_by_cluster.jpg" width="900">

## Conclusion
Its difficult to predict the price of wine using only the label and an a few features about the wine. So in summary, if you have access to a wine sommolier, continue to use that resource for all your wine purchasing guidance because the label will sometime mislead you! 
