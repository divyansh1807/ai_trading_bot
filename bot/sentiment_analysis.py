import pandas as pd
import numpy as np
import requests
import json
import os
from datetime import datetime, timedelta
import time
from textblob import TextBlob
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class SentimentAnalyzer:
    def __init__(self):
        """Initialize the sentiment analyzer"""
        self.news_api_key = os.getenv('NEWS_API_KEY')
        self.twitter_bearer_token = os.getenv('TWITTER_BEARER_TOKEN')
        self.reddit_client_id = os.getenv('REDDIT_CLIENT_ID')
        self.reddit_client_secret = os.getenv('REDDIT_CLIENT_SECRET')
    
    def clean_text(self, text):
        """
        Clean text for sentiment analysis
        
        Args:
            text (str): Text to clean
            
        Returns:
            str: Cleaned text
        """
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions
        text = re.sub(r'@\w+', '', text)
        
        # Remove hashtags
        text = re.sub(r'#\w+', '', text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        
        # Convert to lowercase
        text = text.lower()
        
        return text
    
    def analyze_sentiment(self, text):
        """
        Analyze sentiment of text
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Sentiment analysis results
        """
        # Clean text
        cleaned_text = self.clean_text(text)
        
        # Skip empty text
        if not cleaned_text.strip():
            return {
                'polarity': 0,
                'subjectivity': 0,
                'sentiment': 'neutral'
            }
        
        # Analyze sentiment using TextBlob
        analysis = TextBlob(cleaned_text)
        
        # Determine sentiment label
        if analysis.sentiment.polarity > 0.1:
            sentiment = 'positive'
        elif analysis.sentiment.polarity < -0.1:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        return {
            'polarity': analysis.sentiment.polarity,
            'subjectivity': analysis.sentiment.subjectivity,
            'sentiment': sentiment
        }
    
    def get_news_sentiment(self, symbol, days=7):
        """
        Get sentiment from news articles
        
        Args:
            symbol (str): Cryptocurrency symbol (e.g., 'BTC', 'ETH')
            days (int): Number of days to look back
            
        Returns:
            dict: Aggregated sentiment metrics
        """
        if not self.news_api_key:
            print("❌ NEWS_API_KEY not set in environment variables")
            return None
        
        # Map symbol to search term
        search_terms = {
            'BTC': 'Bitcoin OR BTC',
            'ETH': 'Ethereum OR ETH'
        }
        
        search_term = search_terms.get(symbol.upper(), symbol)
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Format dates for API
        from_date = start_date.strftime('%Y-%m-%d')
        to_date = end_date.strftime('%Y-%m-%d')
        
        # Make API request
        url = f"https://newsapi.org/v2/everything"
        params = {
            'q': search_term,
            'from': from_date,
            'to': to_date,
            'language': 'en',
            'sortBy': 'publishedAt',
            'apiKey': self.news_api_key
        }
        
        try:
            response = requests.get(url, params=params)
            data = response.json()
            
            if data['status'] != 'ok':
                print(f"❌ News API error: {data.get('message', 'Unknown error')}")
                return None
            
            articles = data.get('articles', [])
            
            if not articles:
                print(f"No news articles found for {symbol}")
                return None
            
            # Analyze sentiment for each article
            sentiments = []
            for article in articles:
                # Combine title and description for analysis
                text = f"{article.get('title', '')} {article.get('description', '')}"
                sentiment = self.analyze_sentiment(text)
                sentiments.append(sentiment)
            
            # Calculate aggregate metrics
            polarities = [s['polarity'] for s in sentiments]
            
            # Count sentiment categories
            positive_count = sum(1 for s in sentiments if s['sentiment'] == 'positive')
            negative_count = sum(1 for s in sentiments if s['sentiment'] == 'negative')
            neutral_count = sum(1 for s in sentiments if s['sentiment'] == 'neutral')
            
            # Calculate sentiment score (-1 to 1)
            sentiment_score = sum(polarities) / len(polarities) if polarities else 0
            
            # Determine overall sentiment
            if sentiment_score > 0.1:
                overall_sentiment = 'positive'
            elif sentiment_score < -0.1:
                overall_sentiment = 'negative'
            else:
                overall_sentiment = 'neutral'
            
            return {
                'symbol': symbol,
                'articles_count': len(articles),
                'sentiment_score': sentiment_score,
                'overall_sentiment': overall_sentiment,
                'positive_count': positive_count,
                'negative_count': negative_count,
                'neutral_count': neutral_count,
                'source': 'news'
            }
        
        except Exception as e:
            print(f"❌ Error fetching news sentiment: {str(e)}")
            return None
    
    def get_social_sentiment(self, symbol, platform='twitter', days=7):
        """
        Get sentiment from social media
        
        Args:
            symbol (str): Cryptocurrency symbol (e.g., 'BTC', 'ETH')
            platform (str): Social media platform ('twitter' or 'reddit')
            days (int): Number of days to look back
            
        Returns:
            dict: Aggregated sentiment metrics
        """
        if platform == 'twitter':
            if not self.twitter_bearer_token:
                print("❌ TWITTER_BEARER_TOKEN not set in environment variables")
                return None
            
            # Map symbol to search term
            search_terms = {
                'BTC': 'Bitcoin OR BTC',
                'ETH': 'Ethereum OR ETH'
            }
            
            search_term = search_terms.get(symbol.upper(), symbol)
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Format dates for API
            start_time = start_date.strftime('%Y-%m-%dT%H:%M:%SZ')
            end_time = end_date.strftime('%Y-%m-%dT%H:%M:%SZ')
            
            # Make API request
            url = "https://api.twitter.com/2/tweets/search/recent"
            headers = {
                "Authorization": f"Bearer {self.twitter_bearer_token}"
            }
            params = {
                'query': search_term,
                'max_results': 100,
                'start_time': start_time,
                'end_time': end_time,
                'tweet.fields': 'created_at,public_metrics'
            }
            
            try:
                response = requests.get(url, headers=headers, params=params)
                data = response.json()
                
                tweets = data.get('data', [])
                
                if not tweets:
                    print(f"No tweets found for {symbol}")
                    return None
                
                # Analyze sentiment for each tweet
                sentiments = []
                for tweet in tweets:
                    text = tweet.get('text', '')
                    sentiment = self.analyze_sentiment(text)
                    sentiments.append(sentiment)
                
                # Calculate aggregate metrics
                polarities = [s['polarity'] for s in sentiments]
                
                # Count sentiment categories
                positive_count = sum(1 for s in sentiments if s['sentiment'] == 'positive')
                negative_count = sum(1 for s in sentiments if s['sentiment'] == 'negative')
                neutral_count = sum(1 for s in sentiments if s['sentiment'] == 'neutral')
                
                # Calculate sentiment score (-1 to 1)
                sentiment_score = sum(polarities) / len(polarities) if polarities else 0
                
                # Determine overall sentiment
                if sentiment_score > 0.1:
                    overall_sentiment = 'positive'
                elif sentiment_score < -0.1:
                    overall_sentiment = 'negative'
                else:
                    overall_sentiment = 'neutral'
                
                return {
                    'symbol': symbol,
                    'tweets_count': len(tweets),
                    'sentiment_score': sentiment_score,
                    'overall_sentiment': overall_sentiment,
                    'positive_count': positive_count,
                    'negative_count': negative_count,
                    'neutral_count': neutral_count,
                    'source': 'twitter'
                }
            
            except Exception as e:
                print(f"❌ Error fetching Twitter sentiment: {str(e)}")
                return None
        
        elif platform == 'reddit':
            if not self.reddit_client_id or not self.reddit_client_secret:
                print("❌ Reddit API credentials not set in environment variables")
                return None
            
            # Map symbol to subreddit
            subreddits = {
                'BTC': 'Bitcoin',
                'ETH': 'ethereum'
            }
            
            subreddit = subreddits.get(symbol.upper(), f"Crypto_{symbol}")
            
            # Get Reddit API token
            auth_url = "https://www.reddit.com/api/v1/access_token"
            auth_data = {
                'grant_type': 'client_credentials'
            }
            auth_headers = {
                'User-Agent': 'CryptoSentimentAnalyzer/1.0'
            }
            
            try:
                auth_response = requests.post(
                    auth_url,
                    data=auth_data,
                    headers=auth_headers,
                    auth=(self.reddit_client_id, self.reddit_client_secret)
                )
                
                token_data = auth_response.json()
                access_token = token_data.get('access_token')
                
                if not access_token:
                    print("❌ Failed to get Reddit API token")
                    return None
                
                # Make API request for posts
                posts_url = f"https://oauth.reddit.com/r/{subreddit}/hot"
                posts_headers = {
                    'User-Agent': 'CryptoSentimentAnalyzer/1.0',
                    'Authorization': f"Bearer {access_token}"
                }
                posts_params = {
                    'limit': 100
                }
                
                posts_response = requests.get(posts_url, headers=posts_headers, params=posts_params)
                posts_data = posts_response.json()
                
                posts = posts_data.get('data', {}).get('children', [])
                
                if not posts:
                    print(f"No Reddit posts found for {symbol}")
                    return None
                
                # Analyze sentiment for each post
                sentiments = []
                for post in posts:
                    post_data = post.get('data', {})
                    title = post_data.get('title', '')
                    selftext = post_data.get('selftext', '')
                    
                    # Combine title and text for analysis
                    text = f"{title} {selftext}"
                    sentiment = self.analyze_sentiment(text)
                    sentiments.append(sentiment)
                
                # Calculate aggregate metrics
                polarities = [s['polarity'] for s in sentiments]
                
                # Count sentiment categories
                positive_count = sum(1 for s in sentiments if s['sentiment'] == 'positive')
                negative_count = sum(1 for s in sentiments if s['sentiment'] == 'negative')
                neutral_count = sum(1 for s in sentiments if s['sentiment'] == 'neutral')
                
                # Calculate sentiment score (-1 to 1)
                sentiment_score = sum(polarities) / len(polarities) if polarities else 0
                
                # Determine overall sentiment
                if sentiment_score > 0.1:
                    overall_sentiment = 'positive'
                elif sentiment_score < -0.1:
                    overall_sentiment = 'negative'
                else:
                    overall_sentiment = 'neutral'
                
                return {
                    'symbol': symbol,
                    'posts_count': len(posts),
                    'sentiment_score': sentiment_score,
                    'overall_sentiment': overall_sentiment,
                    'positive_count': positive_count,
                    'negative_count': negative_count,
                    'neutral_count': neutral_count,
                    'source': 'reddit'
                }
            
            except Exception as e:
                print(f"❌ Error fetching Reddit sentiment: {str(e)}")
                return None
        
        else:
            print(f"❌ Unsupported platform: {platform}")
            return None
    
    def get_combined_sentiment(self, symbol, days=7):
        """
        Get combined sentiment from all sources
        
        Args:
            symbol (str): Cryptocurrency symbol (e.g., 'BTC', 'ETH')
            days (int): Number of days to look back
            
        Returns:
            dict: Combined sentiment metrics
        """
        # Get sentiment from different sources
        news_sentiment = self.get_news_sentiment(symbol, days)
        twitter_sentiment = self.get_social_sentiment(symbol, 'twitter', days)
        reddit_sentiment = self.get_social_sentiment(symbol, 'reddit', days)
        
        # Combine sentiment scores
        sources = [s for s in [news_sentiment, twitter_sentiment, reddit_sentiment] if s]
        
        if not sources:
            print(f"❌ No sentiment data available for {symbol}")
            return None
        
        # Calculate weighted sentiment score
        # News: 40%, Twitter: 30%, Reddit: 30%
        weights = {
            'news': 0.4,
            'twitter': 0.3,
            'reddit': 0.3
        }
        
        weighted_score = 0
        total_weight = 0
        
        for source in sources:
            source_type = source['source']
            weight = weights.get(source_type, 0)
            weighted_score += source['sentiment_score'] * weight
            total_weight += weight
        
        combined_score = weighted_score / total_weight if total_weight > 0 else 0
        
        # Determine overall sentiment
        if combined_score > 0.1:
            overall_sentiment = 'positive'
        elif combined_score < -0.1:
            overall_sentiment = 'negative'
        else:
            overall_sentiment = 'neutral'
        
        # Convert sentiment to signal
        if overall_sentiment == 'positive':
            signal = 1  # Buy signal
        elif overall_sentiment == 'negative':
            signal = -1  # Sell signal
        else:
            signal = 0  # Neutral
        
        return {
            'symbol': symbol,
            'sentiment_score': combined_score,
            'overall_sentiment': overall_sentiment,
            'signal': signal,
            'sources': [s['source'] for s in sources],
            'date': datetime.now().strftime('%Y-%m-%d')
        }
