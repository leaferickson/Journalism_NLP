# -*- coding: utf-8 -*-

import pickle
from pymongo import MongoClient

#Connect to Mongo DB
IP = "" #Put this in quotes as a string

client = MongoClient("mongodb://user_name:_pass_word@%s/collection_name" % IP) # defaults to port 27017

db = client.collection_name

cursor = db.collection_name.find()

data = []
for i in range(db.collection_name.count()):
    data.append(cursor[i])

pickle.dump(data, open("journalism_tweets.pkl","wb"))
