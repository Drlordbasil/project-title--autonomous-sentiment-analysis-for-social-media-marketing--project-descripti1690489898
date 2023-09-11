from wordcloud import WordCloud
from textblob import TextBlob
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import nltk
import re
import tweepy
To optimize the Python script, the following improvements can be made:

1. Move the import statements to the top of the file for better readability.
2. Use list comprehension for preprocessing the tweets to improve performance.
3. Minimize the number of API calls by using the tweepy.Cursor() method instead of fetching all tweets at once.
4. Combine the positive and negative word cloud visualizations into a single function.
5. Remove unnecessary print statements.

Here is the optimized code:

```python

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
        for tweet in tweepy.Cursor(api.search, q=query, lang="en").items(count):
            parsed_tweet = {
                'text': tweet.text,
                'sentiment': get_sentiment(tweet.text)
            }
            if tweet.retweet_count > 0:
                if parsed_tweet not in tweets:
                    tweets.append(parsed_tweet)
            else:
                tweets.append(parsed_tweet)
        return tweets
    except tweepy.TweepError as e:
        print("Error:", str(e))

# Preprocessing data


def preprocess_tweet(tweets):
    stop_words = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()
    preprocessed_tweets = []

    for tweet in tweets:
        tweet_lower = tweet['text'].lower()
        tweet_clean = re.sub(r"http\S+|www\S+|https\S+", "", tweet_lower)
        tweet_filtered = [
            lemmatizer.lemmatize(word) for word in word_tokenize(tweet_clean)
            if word not in stop_words and word.isalpha()
        ]

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
    plt.show()

    # Word cloud visualization
    def generate_word_cloud(sentiment_type):
        sentiment_tweets = sentiment_df[sentiment_df['sentiment']
                                        == sentiment_type]['text']
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(
            ' '.join(sentiment_tweets))
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'{sentiment_type.capitalize()} Sentiments Word Cloud')
        plt.show()

    generate_word_cloud('positive')
    generate_word_cloud('negative')

    # Sentiment insights summary
    positive_count = sentiment_count.get('positive', 0)
    negative_count = sentiment_count.get('negative', 0)
    neutral_count = sentiment_count.get('neutral', 0)
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
```

These optimizations will improve the efficiency and readability of the code.
