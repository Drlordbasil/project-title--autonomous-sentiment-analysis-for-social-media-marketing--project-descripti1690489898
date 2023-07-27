import tweepy
import re
import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from textblob import TextBlob
from wordcloud import WordCloud

# Twitter API credentials
consumer_key = 'your_consumer_key'
consumer_secret = 'your_consumer_secret'
access_token = 'your_access_token'
access_token_secret = 'your_access_token_secret'

# Authenticating with Twitter API
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

# Scraping and storing data from Twitter
def scrape_tweets(query, count):
    tweets = []
    try:
        fetched_tweets = api.search(q=query, lang="en", count=count)
        for tweet in fetched_tweets:
            parsed_tweet = {}
            parsed_tweet['text'] = tweet.text
            parsed_tweet['sentiment'] = get_sentiment(tweet.text)
            if tweet.retweet_count > 0:
                if parsed_tweet not in tweets:
                    tweets.append(parsed_tweet)
            else:
                tweets.append(parsed_tweet)
        return tweets
    except tweepy.TweepError as e:
        print("Error : " + str(e))

# Preprocessing data
def preprocess_tweet(tweets):
    stop_words = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()
    preprocessed_tweets = []
    
    for tweet in tweets:
        tweet_lower = tweet['text'].lower()
        tweet_clean = re.sub(r"http\S+|www\S+|https\S+", "", tweet_lower)
        tweet_punct = re.sub(r'[^\w\s]', '', tweet_clean)
        tweet_tokens = word_tokenize(tweet_punct)
        tweet_lem = [lemmatizer.lemmatize(word) for word in tweet_tokens]
        tweet_filtered = [word for word in tweet_lem if word not in stop_words]
        
        tweet['text'] = ' '.join(tweet_filtered)
        preprocessed_tweets.append(tweet)
    
    return preprocessed_tweets

# Sentiment analysis using TextBlob
def get_sentiment(tweet_text):
    blob = TextBlob(tweet_text)
    sentiment = blob.sentiment.polarity
    
    if sentiment > 0:
        return 'positive'
    elif sentiment < 0:
        return 'negative'
    else:
        return 'neutral'

# Real-time sentiment analysis
def real_time_analysis(query, count):
    tweets = scrape_tweets(query, count)
    preprocessed_tweets = preprocess_tweet(tweets)
    sentiment_df = pd.DataFrame(preprocessed_tweets)
    
    # Plotting sentiment distribution
    sentiment_count = sentiment_df['sentiment'].value_counts()
    sentiment_count.plot(kind='bar', title='Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    
    # Word cloud visualization
    positive_tweets = sentiment_df[sentiment_df['sentiment'] == 'positive']['text']
    positive_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(positive_tweets))
    plt.figure(figsize=(10, 5))
    plt.imshow(positive_wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Positive Sentiments Word Cloud')
    
    negative_tweets = sentiment_df[sentiment_df['sentiment'] == 'negative']['text']
    negative_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(negative_tweets))
    plt.figure(figsize=(10, 5))
    plt.imshow(negative_wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Negative Sentiments Word Cloud')
    
    # Sentiment insights summary
    positive_count = sentiment_count['positive']
    negative_count = sentiment_count['negative']
    neutral_count = sentiment_count['neutral']
    total_count = positive_count + negative_count + neutral_count
    
    print("Sentiment Insights Summary:")
    print("Total Tweets:", total_count)
    print("Positive Tweets:", positive_count)
    print("Negative Tweets:", negative_count)
    print("Neutral Tweets:", neutral_count)
    print("Positive %:", round(positive_count / total_count * 100, 2))
    print("Negative %:", round(negative_count / total_count * 100, 2))
    print("Neutral %:", round(neutral_count / total_count * 100, 2))

# Example usage
query = 'social media marketing'
count = 100

real_time_analysis(query, count)