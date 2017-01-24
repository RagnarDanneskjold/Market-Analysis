# Market-Analysis
Creating customer segments using unsupervised learning

Exploring the data using data.describe:



Given the three samples I selected (samples with indices 405, 210, and 13), here are some possibilities of the types of establishments that these three samples might represent:

Mean: Fresh: 12k, milk: 5700, grocery: 8k, frozen: 3k, detergents_paper: 3k, delicatessen: 1500 Max: 112k, Milk: 73k, grocery: 93k, frozen: 61k, detergents_paper: 41k, delicatessen: 48k

The first establishment (index 0) seems to have relatively low costs overall. The costs in each category are much lower overall than the total mean from the sample, which leads me to believe that the establishment might include some retailer which is not as involved in food-related products. However, even though the amount is still lower than the average, the frozen category for the first establishment is still much higher than the minimum value of 25, which implies that this establishment is somehow involved in frozen foods. That means that this establishment might be some smaller fast food restaurant that deals mostly through heating and serving frozen food.

The second establihsment also has relatively low costs across the board, except for fresh and delicatessen. In both of these cases, it surpasses the mean, yet does not come close to approaching the maximum values for those categories. The data point might be representing some establishment which resembles a restaurant which primarily serves foods. The costs for milk, grocery, frozen, and detergents are below average, with fresh and delicatessen well above average.

Finally, the third data point likely represents some food store, as it has values above average across every feature. It's not a particularly large store, as all of the values are still far from the maximum values, but every feature is well-above the overall averages across the dataset which implies a general-purpose retailer, and not a restaurant/cafe.

Next, a label is removed to attempt to predict the value of that label. The label removed is "milk". 

 Interestingly, by changing the random_state parameter in train_test_split, I was able to achieve an R^2 score of 0.4. Although it's not a strong score, it implies that the model was able to somewhat fit to the data. However, by simply changing the random seed, my results were very inconsistent which generally implies that the model fails to fit the data. Using a random_state of 7, I received an R^2 score of -2.16, which implies there was no fit whatsoever. Thus, we cannot draw any conclusions about proportionality of spending patterns in customers.
 
This also shows that the milk feature is independent of the other features -- as in the milk spending does not exist as some linear combination of the other features, which implies that it should be left into our analysis of customer spending.
