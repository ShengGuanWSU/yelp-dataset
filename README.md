# yelp-dataset
This repository is for yelp dataset.

A Yelp reviews data set was publicly released by Yelp for academic research purposes http://www.yelp.com/dataset_challenge. All restaurants and reviews in the U.S. from 2014 to 2015 were considered, which includes 25881 retaurants and 686703 reviews. The frequencies of 1151 keyowrds in the reviews that are non-stop-words and have frequencies above 5000 are considered as attributes. \
We generated a geographic network of restaurants (nodes), in which each restaurant is connected to its 10 nearest restaurants, and there are 244012 edges in total. \
We use the following strategy to generate true subgraph: to generate a connected subgraph of size 100 via random walk, and then removed the reviews which are related with the type "Mexican" out of this subgraph.
