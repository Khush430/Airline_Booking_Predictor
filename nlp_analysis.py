"""
British Airways Review NLP Analysis
===================================

This module implements advanced Natural Language Processing techniques
for analyzing British Airways customer reviews, including sentiment analysis
and topic modeling to extract actionable business insights.

Author: Data Science Portfolio
Project: British Airways End-to-End Analytics
"""

import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

# NLP Libraries
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Machine Learning Libraries
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import GridSearchCV

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
except:
    pass

class ReviewNLPAnalyzer:
    """
    A comprehensive NLP analyzer for British Airways customer reviews.
    
    This class provides sentiment analysis, topic modeling, and text preprocessing
    capabilities for extracting insights from customer feedback.
    """
    
    def __init__(self):
        """Initialize the NLP analyzer with required components."""
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Add custom stop words for airline reviews
        self.custom_stop_words = {
            'british', 'airways', 'ba', 'airline', 'flight', 'plane', 'aircraft',
            'would', 'could', 'one', 'two', 'also', 'get', 'go', 'time'
        }
        self.stop_words.update(self.custom_stop_words)
        
        self.vectorizer = None
        self.lda_model = None
        self.processed_reviews = None
    
    def load_data(self, filepath_or_dataframe, text_column='body'):
        """
        Load review data from CSV file or DataFrame.
        
        Args:
            filepath_or_dataframe: Path to CSV file or pandas DataFrame
            text_column (str): Name of the column containing review text
            
        Returns:
            pd.DataFrame: Loaded review data
        """
        if isinstance(filepath_or_dataframe, str):
            df = pd.read_csv(filepath_or_dataframe)
        else:
            df = filepath_or_dataframe.copy()
        
        # Ensure the text column exists
        if text_column not in df.columns:
            available_cols = [col for col in df.columns if 'text' in col.lower() or 'review' in col.lower()]
            if available_cols:
                text_column = available_cols[0]
                print(f"Using column '{text_column}' for text analysis")
            else:
                raise ValueError(f"Text column '{text_column}' not found in data")
        
        # Remove rows with missing text
        df = df.dropna(subset=[text_column])
        df = df[df[text_column].str.len() > 10]  # Remove very short reviews
        
        print(f"Loaded {len(df)} reviews for analysis")
        return df
    
    def preprocess_text(self, text):
        """
        Preprocess individual text for analysis.
        
        Args:
            text (str): Raw text to preprocess
            
        Returns:
            str: Preprocessed text
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stop words and lemmatize
        tokens = [
            self.lemmatizer.lemmatize(token) 
            for token in tokens 
            if token not in self.stop_words and len(token) > 2
        ]
        
        return ' '.join(tokens)
    
    def analyze_sentiment(self, df, text_column='body'):
        """
        Perform sentiment analysis on reviews using VADER.
        
        Args:
            df (pd.DataFrame): DataFrame containing reviews
            text_column (str): Column name containing review text
            
        Returns:
            pd.DataFrame: DataFrame with sentiment scores added
        """
        print("Performing sentiment analysis...")
        
        # Calculate sentiment scores
        sentiment_scores = []
        for text in df[text_column]:
            scores = self.sentiment_analyzer.polarity_scores(str(text))
            sentiment_scores.append(scores)
        
        # Convert to DataFrame
        sentiment_df = pd.DataFrame(sentiment_scores)
        
        # Add sentiment scores to original DataFrame
        df = df.copy()
        df['sentiment_negative'] = sentiment_df['neg']
        df['sentiment_neutral'] = sentiment_df['neu']
        df['sentiment_positive'] = sentiment_df['pos']
        df['sentiment_compound'] = sentiment_df['compound']
        
        # Classify sentiment based on compound score
        def classify_sentiment(compound_score):
            if compound_score >= 0.05:
                return 'Positive'
            elif compound_score <= -0.05:
                return 'Negative'
            else:
                return 'Neutral'
        
        df['sentiment_category'] = df['sentiment_compound'].apply(classify_sentiment)
        
        # Enhanced sentiment classification
        def enhanced_sentiment_classification(compound_score):
            if compound_score >= 0.5:
                return 'Very Positive'
            elif compound_score >= 0.05:
                return 'Positive'
            elif compound_score <= -0.5:
                return 'Very Negative'
            elif compound_score <= -0.05:
                return 'Negative'
            else:
                return 'Neutral'
        
        df['sentiment_detailed'] = df['sentiment_compound'].apply(enhanced_sentiment_classification)
        
        print("Sentiment analysis completed!")
        return df
    
    def preprocess_for_topic_modeling(self, df, text_column='body'):
        """
        Preprocess text data for topic modeling.
        
        Args:
            df (pd.DataFrame): DataFrame containing reviews
            text_column (str): Column name containing review text
            
        Returns:
            list: List of preprocessed text documents
        """
        print("Preprocessing text for topic modeling...")
        
        processed_texts = []
        for text in df[text_column]:
            processed_text = self.preprocess_text(str(text))
            if processed_text:  # Only add non-empty processed texts
                processed_texts.append(processed_text)
        
        self.processed_reviews = processed_texts
        print(f"Preprocessed {len(processed_texts)} reviews")
        return processed_texts
    
    def perform_topic_modeling(self, processed_texts, n_topics=5, max_features=1000):
        """
        Perform topic modeling using Latent Dirichlet Allocation (LDA).
        
        Args:
            processed_texts (list): List of preprocessed text documents
            n_topics (int): Number of topics to extract
            max_features (int): Maximum number of features for vectorization
            
        Returns:
            tuple: (LDA model, feature names, document-topic matrix)
        """
        print(f"Performing topic modeling with {n_topics} topics...")
        
        # Vectorize the text
        self.vectorizer = CountVectorizer(
            max_features=max_features,
            min_df=2,
            max_df=0.8,
            stop_words='english'
        )
        
        doc_term_matrix = self.vectorizer.fit_transform(processed_texts)
        
        # Fit LDA model
        self.lda_model = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            max_iter=10,
            learning_method='online'
        )
        
        doc_topic_matrix = self.lda_model.fit_transform(doc_term_matrix)
        
        feature_names = self.vectorizer.get_feature_names_out()
        
        print("Topic modeling completed!")
        return self.lda_model, feature_names, doc_topic_matrix
    
    def display_topics(self, n_words=10):
        """
        Display the top words for each topic.
        
        Args:
            n_words (int): Number of top words to display per topic
        """
        if self.lda_model is None or self.vectorizer is None:
            print("Please run topic modeling first!")
            return
        
        feature_names = self.vectorizer.get_feature_names_out()
        
        print(f"\nTop {n_words} words for each topic:")
        print("=" * 50)
        
        for topic_idx, topic in enumerate(self.lda_model.components_):
            top_words_idx = topic.argsort()[-n_words:][::-1]
            top_words = [feature_names[i] for i in top_words_idx]
            
            print(f"Topic {topic_idx + 1}: {', '.join(top_words)}")
    
    def plot_sentiment_distribution(self, df):
        """
        Plot the distribution of sentiment categories.
        
        Args:
            df (pd.DataFrame): DataFrame with sentiment analysis results
        """
        plt.figure(figsize=(12, 5))
        
        # Basic sentiment distribution
        plt.subplot(1, 2, 1)
        sentiment_counts = df['sentiment_category'].value_counts()
        colors = ['#ff6b6b', '#ffd93d', '#6bcf7f']
        plt.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%', colors=colors)
        plt.title('Sentiment Distribution')
        
        # Detailed sentiment distribution
        plt.subplot(1, 2, 2)
        detailed_counts = df['sentiment_detailed'].value_counts()
        plt.bar(detailed_counts.index, detailed_counts.values, color='skyblue')
        plt.title('Detailed Sentiment Distribution')
        plt.xticks(rotation=45)
        plt.ylabel('Number of Reviews')
        
        plt.tight_layout()
        plt.show()
    
    def plot_sentiment_scores(self, df):
        """
        Plot sentiment score distributions.
        
        Args:
            df (pd.DataFrame): DataFrame with sentiment analysis results
        """
        plt.figure(figsize=(15, 5))
        
        # Compound score distribution
        plt.subplot(1, 3, 1)
        plt.hist(df['sentiment_compound'], bins=30, alpha=0.7, color='blue')
        plt.axvline(df['sentiment_compound'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {df["sentiment_compound"].mean():.3f}')
        plt.title('Compound Sentiment Score Distribution')
        plt.xlabel('Compound Score')
        plt.ylabel('Frequency')
        plt.legend()
        
        # Positive vs Negative scores
        plt.subplot(1, 3, 2)
        plt.scatter(df['sentiment_positive'], df['sentiment_negative'], alpha=0.6)
        plt.xlabel('Positive Score')
        plt.ylabel('Negative Score')
        plt.title('Positive vs Negative Sentiment Scores')
        
        # Sentiment over time (if date column exists)
        plt.subplot(1, 3, 3)
        if 'scraped_at' in df.columns:
            df['date'] = pd.to_datetime(df['scraped_at']).dt.date
            daily_sentiment = df.groupby('date')['sentiment_compound'].mean()
            plt.plot(daily_sentiment.index, daily_sentiment.values)
            plt.title('Average Sentiment Over Time')
            plt.xticks(rotation=45)
        else:
            # Box plot of sentiment by category
            sentiment_data = [
                df[df['sentiment_category'] == 'Negative']['sentiment_compound'],
                df[df['sentiment_category'] == 'Neutral']['sentiment_compound'],
                df[df['sentiment_category'] == 'Positive']['sentiment_compound']
            ]
            plt.boxplot(sentiment_data, labels=['Negative', 'Neutral', 'Positive'])
            plt.title('Sentiment Score Distribution by Category')
            plt.ylabel('Compound Score')
        
        plt.tight_layout()
        plt.show()
    
    def generate_wordcloud(self, df, text_column='body', sentiment_filter=None):
        """
        Generate word cloud for reviews.
        
        Args:
            df (pd.DataFrame): DataFrame containing reviews
            text_column (str): Column name containing review text
            sentiment_filter (str): Filter by sentiment category (optional)
        """
        # Filter by sentiment if specified
        if sentiment_filter:
            df_filtered = df[df['sentiment_category'] == sentiment_filter]
            title = f'Word Cloud - {sentiment_filter} Reviews'
        else:
            df_filtered = df
            title = 'Word Cloud - All Reviews'
        
        # Combine all text
        text = ' '.join(df_filtered[text_column].astype(str))
        
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        # Generate word cloud
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            max_words=100,
            colormap='viridis'
        ).generate(processed_text)
        
        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(title, fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def get_sentiment_summary(self, df):
        """
        Generate a comprehensive sentiment analysis summary.
        
        Args:
            df (pd.DataFrame): DataFrame with sentiment analysis results
            
        Returns:
            dict: Summary statistics
        """
        summary = {
            'total_reviews': len(df),
            'average_sentiment': df['sentiment_compound'].mean(),
            'sentiment_std': df['sentiment_compound'].std(),
            'sentiment_distribution': df['sentiment_category'].value_counts().to_dict(),
            'detailed_sentiment_distribution': df['sentiment_detailed'].value_counts().to_dict(),
            'most_positive_review': df.loc[df['sentiment_compound'].idxmax()],
            'most_negative_review': df.loc[df['sentiment_compound'].idxmin()]
        }
        
        return summary

def main():
    """
    Main execution function demonstrating the NLP analysis workflow.
    """
    print("British Airways Review NLP Analysis")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = ReviewNLPAnalyzer()
    
    # Load sample data (replace with actual file path)
    try:
        df = analyzer.load_data("british_airways_reviews.csv")
    except FileNotFoundError:
        print("Sample data file not found. Creating sample data...")
        # Create sample data for demonstration
        sample_reviews = [
            "The flight was delayed for 3 hours. Very disappointing service.",
            "Excellent cabin crew and comfortable seats. Great experience!",
            "Average flight, nothing special but got me to my destination.",
            "Terrible food and rude staff. Will not fly BA again.",
            "Outstanding service from check-in to landing. Highly recommended!"
        ]
        df = pd.DataFrame({'body': sample_reviews})
    
    # Perform sentiment analysis
    df_with_sentiment = analyzer.analyze_sentiment(df)
    
    # Display sentiment summary
    summary = analyzer.get_sentiment_summary(df_with_sentiment)
    print(f"\nSentiment Analysis Summary:")
    print(f"Total reviews analyzed: {summary['total_reviews']}")
    print(f"Average sentiment score: {summary['average_sentiment']:.3f}")
    print(f"Sentiment distribution: {summary['sentiment_distribution']}")
    
    # Preprocess for topic modeling
    processed_texts = analyzer.preprocess_for_topic_modeling(df_with_sentiment)
    
    # Perform topic modeling
    if len(processed_texts) >= 5:  # Need minimum reviews for topic modeling
        lda_model, feature_names, doc_topic_matrix = analyzer.perform_topic_modeling(
            processed_texts, n_topics=3
        )
        
        # Display topics
        analyzer.display_topics(n_words=8)
    
    # Generate visualizations
    if len(df_with_sentiment) > 1:
        analyzer.plot_sentiment_distribution(df_with_sentiment)
        analyzer.plot_sentiment_scores(df_with_sentiment)
        analyzer.generate_wordcloud(df_with_sentiment)

if __name__ == "__main__":
    main()

