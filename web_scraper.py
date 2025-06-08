"""
British Airways Review Web Scraper
==================================

This module implements an ethical web scraper to collect customer reviews
from Skytrax (airlinequality.com) for British Airways sentiment analysis.

The scraper follows best practices including rate limiting, proper headers,
and respectful data collection methods.

Author: Data Science Portfolio
Project: British Airways End-to-End Analytics
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
from urllib.parse import urljoin
import logging
from typing import List, Dict, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BritishAirwaysReviewScraper:
    """
    A web scraper for collecting British Airways customer reviews from Skytrax.
    
    This class implements ethical scraping practices including rate limiting,
    proper user-agent headers, and comprehensive error handling.
    """
    
    def __init__(self, delay_between_requests: float = 1.0):
        """
        Initialize the scraper with configuration.
        
        Args:
            delay_between_requests (float): Delay in seconds between requests
        """
        self.base_url = "https://www.airlinequality.com/airline-reviews/british-airways"
        self.delay = delay_between_requests
        self.session = requests.Session()
        
        # Set respectful headers
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        
        self.reviews_data = []
    
    def get_page_url(self, page_number: int, page_size: int = 100) -> str:
        """
        Construct URL for a specific page of reviews.
        
        Args:
            page_number (int): Page number to scrape
            page_size (int): Number of reviews per page
            
        Returns:
            str: Complete URL for the page
        """
        return f"{self.base_url}/page/{page_number}/?sortby=post_date%3ADesc&pagesize={page_size}"
    
    def extract_review_data(self, soup: BeautifulSoup) -> List[Dict]:
        """
        Extract review data from a BeautifulSoup object.
        
        Args:
            soup (BeautifulSoup): Parsed HTML content
            
        Returns:
            List[Dict]: List of review dictionaries
        """
        reviews = []
        review_containers = soup.find_all("div", {"class": "text_content"})
        
        for container in review_containers:
            try:
                # Extract review text
                review_text = container.get_text(strip=True)
                
                # Clean the review text
                review_text = self.clean_review_text(review_text)
                
                if review_text and len(review_text) > 50:  # Filter out very short reviews
                    reviews.append({
                        'review_text': review_text,
                        'review_length': len(review_text),
                        'scraped_at': pd.Timestamp.now()
                    })
                    
            except Exception as e:
                logger.warning(f"Error extracting review: {e}")
                continue
        
        return reviews
    
    def extract_detailed_review_data(self, soup: BeautifulSoup) -> List[Dict]:
        """
        Extract detailed review data including titles and ratings.
        
        Args:
            soup (BeautifulSoup): Parsed HTML content
            
        Returns:
            List[Dict]: List of detailed review dictionaries
        """
        reviews = []
        
        # Find all review articles
        review_articles = soup.find_all('article', {'itemprop': 'review'})
        
        for article in review_articles:
            try:
                # Extract title
                title_element = article.find('h2')
                title = title_element.get_text(strip=True) if title_element else "No Title"
                
                # Extract review body
                body_element = article.find('div', {'class': 'text_content'})
                body = body_element.get_text(strip=True) if body_element else ""
                
                # Extract rating (if available)
                rating_element = article.find('span', class_='star fill')
                rating = None
                if rating_element:
                    try:
                        rating = int(rating_element.get_text(strip=True))
                    except (ValueError, AttributeError):
                        rating = None
                
                # Clean and validate data
                title = self.clean_review_text(title)
                body = self.clean_review_text(body)
                
                if body and len(body) > 50:
                    reviews.append({
                        'title': title,
                        'body': body,
                        'rating': rating,
                        'review_length': len(body),
                        'scraped_at': pd.Timestamp.now()
                    })
                    
            except Exception as e:
                logger.warning(f"Error extracting detailed review: {e}")
                continue
        
        return reviews
    
    def clean_review_text(self, text: str) -> str:
        """
        Clean and normalize review text.
        
        Args:
            text (str): Raw review text
            
        Returns:
            str: Cleaned review text
        """
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:\-\'"()]', '', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def scrape_reviews(self, num_pages: int = 10, page_size: int = 100, 
                      detailed: bool = True) -> pd.DataFrame:
        """
        Scrape reviews from multiple pages.
        
        Args:
            num_pages (int): Number of pages to scrape
            page_size (int): Reviews per page
            detailed (bool): Whether to extract detailed review data
            
        Returns:
            pd.DataFrame: DataFrame containing all scraped reviews
        """
        logger.info(f"Starting to scrape {num_pages} pages of British Airways reviews")
        
        all_reviews = []
        
        for page in range(1, num_pages + 1):
            try:
                logger.info(f"Scraping page {page}/{num_pages}")
                
                # Construct URL
                url = self.get_page_url(page, page_size)
                
                # Make request
                response = self.session.get(url, timeout=10)
                response.raise_for_status()
                
                # Parse HTML
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Extract reviews
                if detailed:
                    page_reviews = self.extract_detailed_review_data(soup)
                else:
                    page_reviews = self.extract_review_data(soup)
                
                all_reviews.extend(page_reviews)
                
                logger.info(f"Extracted {len(page_reviews)} reviews from page {page}")
                logger.info(f"Total reviews collected: {len(all_reviews)}")
                
                # Respectful delay between requests
                time.sleep(self.delay)
                
            except requests.RequestException as e:
                logger.error(f"Request error on page {page}: {e}")
                continue
            except Exception as e:
                logger.error(f"Unexpected error on page {page}: {e}")
                continue
        
        # Convert to DataFrame
        df = pd.DataFrame(all_reviews)
        
        if not df.empty:
            # Remove duplicates based on review text
            initial_count = len(df)
            if detailed:
                df = df.drop_duplicates(subset=['body'], keep='first')
            else:
                df = df.drop_duplicates(subset=['review_text'], keep='first')
            
            final_count = len(df)
            logger.info(f"Removed {initial_count - final_count} duplicate reviews")
            
            # Add metadata
            df['source'] = 'Skytrax'
            df['airline'] = 'British Airways'
        
        logger.info(f"Scraping completed. Total unique reviews: {len(df)}")
        
        return df
    
    def save_reviews(self, df: pd.DataFrame, filename: str = "ba_reviews.csv") -> None:
        """
        Save reviews to CSV file.
        
        Args:
            df (pd.DataFrame): Reviews DataFrame
            filename (str): Output filename
        """
        try:
            df.to_csv(filename, index=False, encoding='utf-8')
            logger.info(f"Reviews saved to {filename}")
        except Exception as e:
            logger.error(f"Error saving reviews: {e}")
    
    def get_review_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Generate statistics about the scraped reviews.
        
        Args:
            df (pd.DataFrame): Reviews DataFrame
            
        Returns:
            Dict: Statistics dictionary
        """
        if df.empty:
            return {}
        
        stats = {
            'total_reviews': len(df),
            'avg_review_length': df['review_length'].mean() if 'review_length' in df.columns else 0,
            'min_review_length': df['review_length'].min() if 'review_length' in df.columns else 0,
            'max_review_length': df['review_length'].max() if 'review_length' in df.columns else 0,
        }
        
        if 'rating' in df.columns:
            ratings = df['rating'].dropna()
            if not ratings.empty:
                stats.update({
                    'avg_rating': ratings.mean(),
                    'rating_distribution': ratings.value_counts().to_dict()
                })
        
        return stats

def main():
    """
    Main execution function demonstrating the scraper usage.
    """
    print("British Airways Review Scraper")
    print("=" * 40)
    
    # Initialize scraper
    scraper = BritishAirwaysReviewScraper(delay_between_requests=1.5)
    
    # Scrape reviews
    reviews_df = scraper.scrape_reviews(
        num_pages=10,
        page_size=100,
        detailed=True
    )
    
    if not reviews_df.empty:
        # Display statistics
        stats = scraper.get_review_statistics(reviews_df)
        print(f"\nScraping Results:")
        print(f"Total reviews collected: {stats.get('total_reviews', 0)}")
        print(f"Average review length: {stats.get('avg_review_length', 0):.1f} characters")
        
        if 'avg_rating' in stats:
            print(f"Average rating: {stats['avg_rating']:.1f}")
        
        # Save to file
        scraper.save_reviews(reviews_df, "british_airways_reviews.csv")
        
        # Display sample reviews
        print(f"\nSample Reviews:")
        print("-" * 40)
        for idx, row in reviews_df.head(3).iterrows():
            if 'title' in row:
                print(f"Title: {row['title']}")
            if 'body' in row:
                print(f"Review: {row['body'][:200]}...")
            if 'rating' in row and pd.notna(row['rating']):
                print(f"Rating: {row['rating']}")
            print("-" * 40)
    
    else:
        print("No reviews were collected. Please check the scraper configuration.")

if __name__ == "__main__":
    main()

