# -*- coding: utf-8 -*-

import pandas as pd
from pymongo import MongoClient
from tweet_gatherer import get_all_tweets, processTweet


#Connect to Mongo DB
IP = "" #Put IP in quotes as a string

client = MongoClient("mongodb://user_name:pass_word@%s/collection_name" % IP) # defaults to port 27017

db = client.collection_name

#Test section
db.collection_name.count()
ok = [{"key":4},{"key":2}]
db.source.insert_many(ok)

#Get Data Sources
data_sources = pd.read_csv("Data Sources.csv")
outlets = data_sources["Outlet"]
data_sources.drop("Outlet", axis = 1, inplace = True)


#Get all the tweets and put them in a dictionary
tweetdict = {}
for colname, col in enumerate(data_sources):
    for source in data_sources[col]:  
        if type(source) == str:
            print(source) #To help keep track of where the gatherer is and if it is working or not
            tweets = get_all_tweets(source)
            dict_list = []
            for tweet in tweets:
                dict_list.append(processTweet(tweet, source))
            db.project_collection.insert_many(dict_list)

