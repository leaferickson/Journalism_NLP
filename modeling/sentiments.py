# -*- coding: utf-8 -*-

import pickle
from sklearn.decomposition import LatentDirichletAllocation, NMF, TruncatedSVD
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

tweet_topics = pickle.load(open("../data_clean/tweet_topics.pkl","rb"))

def replace_tweet_source(old_source, new_source):
    mask = tweet_topics[:,2] == old_source
    tweet_topics[:,2][mask] = new_source


#If a news outlet had more than one twitter feed, condense it down to one label
replace_tweet_source("nytimespolitics", "nytimes")
replace_tweet_source("BuzzFeedPol", "BuzzFeedNews")
replace_tweet_source("foxnewspolitics", "FoxNews")
replace_tweet_source("postpolitics", "washingtonpost")
replace_tweet_source("CNNPolitics", "CNN")
replace_tweet_source("nprpolitics", "NPR")
replace_tweet_source("ABCPolitics", "ABC")
replace_tweet_source("NBCPolitics", "NBCNews")
replace_tweet_source("HuffPostPol", "HuffPost")
replace_tweet_source("pewjournalism", "pewresearch")
replace_tweet_source("ReutersPolitics", "Reuters")
replace_tweet_source("WSJusnews", "WSJ")
replace_tweet_source("latimespolitics", "latimes")
replace_tweet_source("usatodayDC", "USATODAY")
replace_tweet_source("TheAtlPolitics", "TheAtlantic")
replace_tweet_source("GdnPolitics", "guardian")
replace_tweet_source("BBCPolitics", "BBC")
replace_tweet_source("BBCBreaking", "BBC")
replace_tweet_source("BBCWorld", "BBC")
replace_tweet_source("bpolitics", "Bloomberg")
replace_tweet_source("business", "Bloomberg")


from vaderSentiment import vaderSentiment as vs
sentiment_analyzer = vs.SentimentIntensityAnalyzer()
tweet_polarities = []
for tweet in tweet_topics[:,3]:
    sent_output = sentiment_analyzer.polarity_scores(tweet)
    tweet_polarities.append(sent_output["compound"])

topic_sentiments = (tweet_topics[:,-2]).reshape(len(tweet_topics), 1) * tweet_topics[:,4:-2]
tweet_topics = np.append(tweet_topics, topic_sentiments, axis = 1)





pickle.dump(tweet_topics, open("tweet_topics2.pkl","wb"))



X = tweet_topics[:,4:]
y = tweet_topics[:,2]




######First attempt at modeling. Attempting to classify the news outlet or regress on a 1-10 liberal-conservative scale
######Clustering, clustering, can't forget about clustering
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 17, stratify = y)
#scaler = StandardScaler()
#X_train = scaler.fit_transform(X_train)
#X_test = scaler.transform(X_test)
model = SGDClassifier(loss = 'huber', max_iter = 125, penalty = 'l2', random_state = 44, n_jobs = -1)
model.fit(X_train, y_train)
preds = model.predict(X_train)
accuracy_score(y_train, preds)
preds_test = model.predict(X_test)
accuracy_score(y_test, preds_test)
compare = pd.DataFrame(pd.concat((y_train.reset_index(), pd.Series(preds)), axis = 1))




from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, preds_test)
cm = pd.crosstab(y_test, preds_test)
sources, counts = np.unique(np.array(all_tweets_df.loc[:,2:2]), return_counts = True)
a = pd.DataFrame(sources, counts)
cm2 = pd.crosstab(sources, sources)




data_sources = pd.read_csv("Data Sources.csv")


def add_bias():
    tweet_topics2 = np.append(tweet_topics, np.array([0] * len(tweet_topics)).reshape(len(tweet_topics), 1), axis = 1)
    for source in np.unique(tweet_topics2[:,2]):
        print(source)
        mask = tweet_topics2[:,2] == source
        tweet_topics2[:,-1][mask] = data_sources[data_sources["Outlet"] == source]["R_L_scale"]
    return tweet_topics2

tweet_topics = add_bias()






##Regression Section
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

X = tweet_topics[:,4:-1]
y = tweet_topics[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 17, stratify = y)

lr = LinearRegression()
lr.fit(X_train, y_train)
predictions_train = lr.predict(X_train)
mean_squared_error(y_train, predictions_train)
predictions_test = lr.predict(X_test)
mean_squared_error(y_test, predictions_test)

preds = []
for pred in predictions_test:
    preds.append(int(round(pred)))
plt.hist(preds)
y_test = list(y_test)
plt.hist(y_test)
plt.scatter(y_test, preds)












###The final product. Taking out sources not easily labeled as either liberal or conservative, try to predict if a tweet from a news outlet is one or the other..
tweet_topics = pickle.load(open("tweet_topics2.pkl","rb"))
data_sources = pd.read_csv("Data Sources simple.csv")

#Remove unwanted tweets
def delete_tweet_source(source_to_remove):
    return np.delete(tweet_topics, np.where(tweet_topics[:,2] == source_to_remove), axis = 0)

tweet_topics = delete_tweet_source("ajam")
tweet_topics = delete_tweet_source("pewresearch")
tweet_topics = delete_tweet_source("politico")
tweet_topics = delete_tweet_source("Reuters")
tweet_topics = delete_tweet_source("AP")
tweet_topics = delete_tweet_source("TheEconomist")
tweet_topics = delete_tweet_source("WSJ")
tweet_topics = delete_tweet_source("Bloomberg")
tweet_topics = delete_tweet_source("USATODAY")
tweet_topics = delete_tweet_source("TIME")


def add_bias():
    tweet_topics2 = np.append(tweet_topics, np.array([0] * len(tweet_topics)).reshape(len(tweet_topics), 1), axis = 1)
    for source in np.unique(tweet_topics2[:,2]):
        print(source)
        mask = tweet_topics2[:,2] == source
        tweet_topics2[:,-1][mask] = data_sources[data_sources["Outlet"] == source]["R_L_scale"]
    return tweet_topics2

tweet_topics = add_bias()


##Regression Section
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X = tweet_topics[:,4:-1]
y = tweet_topics[:,-1]
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y =le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 17, stratify = y)

lr = LogisticRegression()
lr.fit(X_train, y_train)
predictions_train = lr.predict(X_train)
accuracy_score(y_train, predictions_train)
#mean_squared_error(y_train, predictions_train)
predictions_test = lr.predict(X_test)
accuracy_score(y_test, predictions_test)
#mean_squared_error(y_test, predictions_test)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, predictions_test)

plt.hist(list(y))



pickle.dump(X, open("X.pkl","wb"))
pickle.dump(y, open("y.pkl","wb"))
