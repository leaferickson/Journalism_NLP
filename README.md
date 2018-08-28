# Journalism NLP

## Organization and Methodology
Method for gathering tweets is in [get_tweets](https://github.com/leaferickson/Journalism_NLP/blob/master/data_collection/get_tweets.py) script in the [Data Collection] folder(https://github.com/leaferickson/Journalism_NLP/tree/master/data_collection).

Data was stored in MongoDB and then retrieved using [this](https://github.com/leaferickson/Journalism_NLP/blob/master/data_clean/get_data.py) script.

Data was then cleaned using the tweet_clean.py file in [Data Clean](https://github.com/leaferickson/Journalism_NLP/tree/master/data_clean), which uses nlp_pipeline as a pipeline manager. Credit to [Zach](https://github.com/ZWMiller/nlp_pipe_manager) for the pipeline manager.

After cleaning the tweets and vectorizing them, topic modeling is done futher down in the tweet_clean.py script, and sentiment analysis is done in the [modeling](https://github.com/leaferickson/Journalism_NLP/tree/master/modeling) folder.

Finally, a logistic regression classification was then tuned and trained. I did attempts reression prediction on a 1-10 liberal-conservative scale, but this did not turn out vas well as I hoped. The simple liberal-conservative mix was all that I could model from tweets alone and with a the constantly changing topics of news outlets.
