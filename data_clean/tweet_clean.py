# -*- coding: utf-8 -*-


###NOTE: Sub the name of your IP, user_id, etc. by searching on "sub"
import pickle
import re
from nlp_pipeline import nlp_preprocessor
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize.casual import TweetTokenizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF, TruncatedSVD
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

all_tweets = pickle.load(open("journalism_tweets.pkl","rb"))


extra_tweets, train_tweets = train_test_split(all_tweets, test_size = .09, random_state = 19, stratify = [tweet["source"] for tweet in all_tweets])
np.sum([tweet["source"] == "AP" for tweet in train_tweets])

stop_words = set(stopwords.words('english'))
stop_words.add("rt")
stop_words.add("repost")
stop_words.add("https")
stop_words.add("http")
stop_words.add("http ")
stop_words.add("https ")
stop_words.add("htt")
stop_words.add("nhttps")


def clean_text(text, tokenizer, stemmer, stopwords = stop_words, combine_tweets = False):
    """
    A naive function to lowercase all works can clean them quickly.
    This is the default behavior if no other cleaning function is specified
    """
    cleaned_text = []
    for post in text:
        cleaned_words = []
        for word in tokenizer(post["tweet"][2:]):
            word = re.sub(r'^https?:\/\/.*[\r\n]*', '', word, flags=re.MULTILINE)
            word = re.sub(r'^nhttps?:\/\/.*[\r\n]*', '', word, flags=re.MULTILINE)
            low_word = word.lower().strip(" ")
            if low_word == "" or len(low_word) <= 3:
                continue
            if word_tokenize(low_word)[0].isalpha():
                if low_word not in stopwords:
                    if "htt" in low_word:
                        continue
#                        print(low_word)
                    cleaned_words.append(low_word)
        if combine_tweets == False:
            cleaned_text.append(' '.join(cleaned_words))
        else:
            cleaned_text = cleaned_words
    return cleaned_text


##Vectorize
#Count Vectorize
#count_vect = nlp_preprocessor(tokenizer=TweetTokenizer().tokenize, cleaning_function=clean_text, stemmer=None)
#count_vect.fit(train_tweets)
#count_vectorized_tweets = count_vect.transform(train_tweets)
#train_words = pd.DataFrame(pd.DataFrame(count_vectorized_tweets.toarray()).sum())
#train_words = train_words.set_index([count_vect.get_words()])


#TD-IDF Vectorize
tfidf_vect = nlp_preprocessor(vectorizer = TfidfVectorizer(),tokenizer=TweetTokenizer().tokenize, cleaning_function=clean_text, stemmer=None)
tfidf_vect.fit(train_tweets)
tfidf_vectorized_tweets = tfidf_vect.transform(train_tweets)
train_words = pd.DataFrame(pd.DataFrame(tfidf_vectorized_tweets.toarray()).sum())
train_words = train_words.set_index([tfidf_vect.get_words()])


#Display Topics
def display_topics(model, feature_names, no_top_words, topic_names=None):
    output = ""
    for ix, topic in enumerate(model.components_):
        if not topic_names or not topic_names[ix]:
            output += ("\nTopic %d \n" % ix)
        else:
            output += ("\nTopic: '%d \n'" % topic_names[ix])
        output += (", ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))
    return output

##Get Topics
def createTopics(n_topics, word_per_topic = 10):

#    lda = LatentDirichletAllocation(n_components = n_topics)
#    lda_tfidf_topics = lda.fit_transform(tfidf_vectorized_tweets)
#    lda_count_topics = lda.fit_transform(count_vectorized_tweets)
    
    nmf = NMF(n_components = n_topics, random_state = 101)
    nmf_tfidf_topics = nmf.fit_transform(tfidf_vectorized_tweets)
#    nmf_count_topics = nmf.fit_transform(count_vectorized_tweets)
    
#    lsa = TruncatedSVD(n_components = n_topics)
#    lsa_tfidf_topics = lsa.fit_transform(tfidf_vectorized_tweets)
#    lsa_count_topics = lsa.fit_transform(count_vectorized_tweets)
    
    
    topics = []
#    topics.append(display_topics(lda, tfidf_vect.get_words(), word_per_topic))
#    topics.append(display_topics(lda, count_vect.get_words(), word_per_topic))
    topics.append(display_topics(nmf, tfidf_vect.get_words(), word_per_topic))
#    topics.append(display_topics(nmf, count_vect.get_words(), word_per_topic))
#    topics.append(display_topics(lsa, tfidf_vect.get_words(), word_per_topic))
#    topics.append(display_topics(lsa, count_vect.get_words(), word_per_topic))
    return topics

topics25 = createTopics(25)
topics32 = createTopics(32)
topics40 = createTopics(40)
topics50 = createTopics(50)
topics75 = createTopics(75)
topics93 = createTopics(93)
topics95 = createTopics(95)
topics97 = createTopics(97)
topics98 = createTopics(98)
topics100 = createTopics(100)
topics101 = createTopics(100)
topics400 = createTopics(400)



pickle.dump(topics97, open("topics.pkl","wb"))



###Transform all tweets to topic space
nmf = NMF(n_components = 98, random_state = 101)
nmf_tfidf_topics = nmf.fit_transform(tfidf_vectorized_tweets)
tfidf_vectorized_all_tweets = tfidf_vect.transform(all_tweets)
tweets_in_topic_space = nmf.transform(tfidf_vectorized_all_tweets)




#Get a numpy array of tweets in topic space with their info
all_tweets = pickle.load(open("journalism_tweets.pkl","rb"))
ids = [tweet["_id"] for tweet in all_tweets]
times = [tweet["datetime"] for tweet in all_tweets]
sources = [tweet["source"] for tweet in all_tweets]
tweets = [tweet["tweet"] for tweet in all_tweets]
cols = [ids, times, sources, tweets]
all_tweets_df = pd.concat([pd.Series(x) for x in cols], axis=1)
tweet_topics = np.array(pd.concat(all_tweets_df, pd.DataFrame(tweets_in_topic_space), axis = 1))
pickle.dump(tweet_topics, open("tweet_topics.pkl","wb"))


