# -*- coding: utf-8 -*-

import pandas as pd
import tweepy


###Below funtion based on code from Yanofksy at https://gist.github.com/yanofsky/5436496

def get_all_tweets(screen_name, number_to_grab = 50):
    """Gather a user's last 3240 tweets (the most that twitter will allow).
    To switch to a different number of tweets comment out while lop and switch
    count = 200 to count - number_to_grab."""
    
    consumer_key = ""
    consumer_secret = ""
    access_key = ""
    access_secret = "" 
    
    #authorize twitter, initialize tweepy
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tweepy.API(auth)

    #initialize a list to hold all the tweepy Tweets
    alltweets = []

    #make initial request for most recent tweets (200 is the maximum allowed count)
    new_tweets = api.user_timeline(screen_name = screen_name,count=200) #switch this to number_to_grab if you want < 200

    #save most recent tweets
    alltweets.extend(new_tweets)

    #save the id of the oldest tweet less one
    oldest = alltweets[-1].id - 1

    #keep grabbing tweets until there are no tweets left to grab
    while len(new_tweets) > 0:

        #all subsiquent requests use the max_id param to prevent duplicates
        new_tweets = api.user_timeline(screen_name = screen_name,count=200,max_id=oldest)

        #save most recent tweets
        alltweets.extend(new_tweets)

        #update the id of the oldest tweet less one
        oldest = alltweets[-1].id - 1

#         print ("...%s tweets downloaded so far") % (len(alltweets))

    #transform the tweepy tweets into a 2D array that will populate the csv	
    outtweets = [[tweet.id_str, tweet.created_at, tweet.text.encode("utf-8")] for tweet in alltweets]

    #write the csv
    csv_out = pd.DataFrame(outtweets)
    csv_out.columns = ["id","created_at","text"]
    return outtweets



def processTweet(tweet, source):
    """Process the tweets into a dictionary"""
    tweet_dict = {
        'datetime': tweet[1],
        'tweet': str(tweet[2]),
        'source': str(source)
    }
    return tweet_dict



