"""
British Airways Data Science Project - Main Integration Script
=============================================================

This script demonstrates the complete end-to-end data science pipeline
for British Airways analytics, integrating booking prediction, web scraping,
and NLP analysis components.

Author: Data Science Portfolio
Project: British Airways End-to-End Analytics
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
import logging

# Import custom modules
from booking_prediction import BookingPredictor
from web_scraper import BritishAirwaysReviewScraper
from nlp_analysis import ReviewNLPAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ba_analytics.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class BritishAirwaysAnalytics:
    """
    Main class integrating all components of the British Airways analytics project.
    
    This class orchestrates the complete data science pipeline from data collection
    to insight generation, combining predictive modeling, web scraping, and NLP analysis.
    """
    
    def __init__(self, output_dir="results"):
        """
        Initialize the analytics pipeline.
        
        Args:
            output_dir (str): Directory to save results and outputs
        """
        self.output_dir = output_dir
        self.create_output_directory()
        
        # Initialize components
        self.booking_predictor = BookingPredictor()
        self.review_scraper = BritishAirwaysReviewScraper(delay_between_requests=1.5)
        self.nlp_analyzer = ReviewNLPAnalyzer()
        
        # Data storage
        self.booking_data = None
        self.review_data = None
        self.booking_results = None
        self.sentiment_results = None
        
        logger.info("British Airways Analytics pipeline initialized")
    
    def create_output_directory(self):
        """Create output directory if it doesn't exist."""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            logger.info(f"Created output directory: {self.output_dir}")
    
    def run_booking_prediction_analysis(self, booking_data_path="customer_booking.csv"):
        """
        Execute the booking prediction component.
        
        Args:
            booking_data_path (str): Path to booking data CSV file
            
        Returns:
            dict: Booking prediction results
        """
        logger.info("Starting booking prediction analysis...")
        
        try:
            # Load and preprocess booking data
            X, y = self.booking_predictor.load_and_preprocess_data(booking_data_path)
            self.booking_data = pd.concat([X, y], axis=1)
            
            # Train model and get results
            self.booking_results = self.booking_predictor.train_model(X, y)
            
            # Save results
            results_summary = {
                'accuracy': self.booking_results['accuracy'],
                'auc_roc': self.booking_results['auc_roc'],
                'cv_mean': self.booking_results['cv_mean'],
                'cv_std': self.booking_results['cv_std'],
                'top_features': self.booking_results['feature_importance'].head(10).to_dict('records')
            }
            
            # Save to file
            results_df = pd.DataFrame([results_summary])
            results_df.to_csv(f"{self.output_dir}/booking_prediction_results.csv", index=False)
            
            # Save feature importance
            self.booking_results['feature_importance'].to_csv(
                f"{self.output_dir}/feature_importance.csv", index=False
            )
            
            logger.info(f"Booking prediction completed - Accuracy: {results_summary['accuracy']:.3f}")
            return results_summary
            
        except Exception as e:
            logger.error(f"Error in booking prediction analysis: {e}")
            return None
    
    def run_review_collection(self, num_pages=10, force_scrape=False):
        """
        Execute the review collection component.
        
        Args:
            num_pages (int): Number of pages to scrape
            force_scrape (bool): Force new scraping even if data exists
            
        Returns:
            pd.DataFrame: Collected review data
        """
        logger.info("Starting review collection...")
        
        review_file = f"{self.output_dir}/british_airways_reviews.csv"
        
        # Check if reviews already exist
        if os.path.exists(review_file) and not force_scrape:
            logger.info("Loading existing review data...")
            self.review_data = pd.read_csv(review_file)
        else:
            try:
                # Scrape new reviews
                self.review_data = self.review_scraper.scrape_reviews(
                    num_pages=num_pages,
                    page_size=100,
                    detailed=True
                )
                
                if not self.review_data.empty:
                    # Save reviews
                    self.review_scraper.save_reviews(self.review_data, review_file)
                    
                    # Generate and save statistics
                    stats = self.review_scraper.get_review_statistics(self.review_data)
                    stats_df = pd.DataFrame([stats])
                    stats_df.to_csv(f"{self.output_dir}/review_collection_stats.csv", index=False)
                    
                    logger.info(f"Review collection completed - {len(self.review_data)} reviews collected")
                else:
                    logger.warning("No reviews were collected")
                    
            except Exception as e:
                logger.error(f"Error in review collection: {e}")
                return None
        
        return self.review_data
    
    def run_nlp_analysis(self):
        """
        Execute the NLP analysis component.
        
        Returns:
            dict: NLP analysis results
        """
        logger.info("Starting NLP analysis...")
        
        if self.review_data is None or self.review_data.empty:
            logger.error("No review data available for NLP analysis")
            return None
        
        try:
            # Perform sentiment analysis
            self.sentiment_results = self.nlp_analyzer.analyze_sentiment(
                self.review_data, text_column='body'
            )
            
            # Save sentiment results
            self.sentiment_results.to_csv(
                f"{self.output_dir}/sentiment_analysis_results.csv", index=False
            )
            
            # Generate sentiment summary
            sentiment_summary = self.nlp_analyzer.get_sentiment_summary(self.sentiment_results)
            
            # Preprocess for topic modeling
            processed_texts = self.nlp_analyzer.preprocess_for_topic_modeling(
                self.sentiment_results, text_column='body'
            )
            
            # Perform topic modeling if enough data
            topic_results = None
            if len(processed_texts) >= 10:
                lda_model, feature_names, doc_topic_matrix = self.nlp_analyzer.perform_topic_modeling(
                    processed_texts, n_topics=5
                )
                
                # Extract topic information
                topics = []
                for topic_idx, topic in enumerate(lda_model.components_):
                    top_words_idx = topic.argsort()[-10:][::-1]
                    top_words = [feature_names[i] for i in top_words_idx]
                    topics.append({
                        'topic_id': topic_idx + 1,
                        'top_words': ', '.join(top_words[:5]),
                        'all_words': ', '.join(top_words)
                    })
                
                topic_results = pd.DataFrame(topics)
                topic_results.to_csv(f"{self.output_dir}/topic_modeling_results.csv", index=False)
            
            # Save comprehensive summary
            nlp_summary = {
                'total_reviews_analyzed': sentiment_summary['total_reviews'],
                'average_sentiment_score': sentiment_summary['average_sentiment'],
                'negative_reviews_pct': sentiment_summary['sentiment_distribution'].get('Negative', 0) / sentiment_summary['total_reviews'] * 100,
                'positive_reviews_pct': sentiment_summary['sentiment_distribution'].get('Positive', 0) / sentiment_summary['total_reviews'] * 100,
                'neutral_reviews_pct': sentiment_summary['sentiment_distribution'].get('Neutral', 0) / sentiment_summary['total_reviews'] * 100,
                'topics_identified': len(topics) if topic_results is not None else 0
            }
            
            summary_df = pd.DataFrame([nlp_summary])
            summary_df.to_csv(f"{self.output_dir}/nlp_analysis_summary.csv", index=False)
            
            logger.info(f"NLP analysis completed - Average sentiment: {nlp_summary['average_sentiment_score']:.3f}")
            return nlp_summary
            
        except Exception as e:
            logger.error(f"Error in NLP analysis: {e}")
            return None
    
    def generate_comprehensive_report(self):
        """
        Generate a comprehensive report combining all analysis results.
        
        Returns:
            dict: Complete project results
        """
        logger.info("Generating comprehensive report...")
        
        report = {
            'project_overview': {
                'title': 'British Airways End-to-End Data Science Project',
                'execution_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'components': ['Booking Prediction', 'Review Collection', 'NLP Analysis']
            }
        }
        
        # Add booking prediction results
        if self.booking_results:
            report['booking_prediction'] = {
                'model_accuracy': f"{self.booking_results['accuracy']:.3f}",
                'auc_roc_score': f"{self.booking_results['auc_roc']:.3f}",
                'cross_validation_mean': f"{self.booking_results['cv_mean']:.3f}",
                'dataset_size': len(self.booking_data) if self.booking_data is not None else 0,
                'top_feature': self.booking_results['feature_importance'].iloc[0]['feature']
            }
        
        # Add review collection results
        if self.review_data is not None:
            report['review_collection'] = {
                'total_reviews_collected': len(self.review_data),
                'average_review_length': self.review_data['review_length'].mean() if 'review_length' in self.review_data.columns else 0,
                'data_source': 'Skytrax (airlinequality.com)',
                'collection_method': 'Ethical web scraping'
            }
        
        # Add NLP analysis results
        if self.sentiment_results is not None:
            sentiment_dist = self.sentiment_results['sentiment_category'].value_counts()
            report['nlp_analysis'] = {
                'reviews_analyzed': len(self.sentiment_results),
                'average_sentiment': f"{self.sentiment_results['sentiment_compound'].mean():.3f}",
                'negative_sentiment_pct': f"{(sentiment_dist.get('Negative', 0) / len(self.sentiment_results) * 100):.1f}%",
                'positive_sentiment_pct': f"{(sentiment_dist.get('Positive', 0) / len(self.sentiment_results) * 100):.1f}%",
                'analysis_methods': ['VADER Sentiment Analysis', 'LDA Topic Modeling']
            }
        
        # Save comprehensive report
        import json
        with open(f"{self.output_dir}/comprehensive_report.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        # Create summary DataFrame
        summary_data = []
        for component, results in report.items():
            if component != 'project_overview':
                for metric, value in results.items():
                    summary_data.append({
                        'component': component,
                        'metric': metric,
                        'value': value
                    })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(f"{self.output_dir}/project_summary.csv", index=False)
        
        logger.info("Comprehensive report generated successfully")
        return report
    
    def run_complete_pipeline(self, booking_data_path="customer_booking.csv", 
                            scrape_pages=10, force_scrape=False):
        """
        Execute the complete analytics pipeline.
        
        Args:
            booking_data_path (str): Path to booking data
            scrape_pages (int): Number of pages to scrape
            force_scrape (bool): Force new scraping
            
        Returns:
            dict: Complete pipeline results
        """
        logger.info("Starting complete British Airways analytics pipeline...")
        
        try:
            # Component 1: Booking Prediction
            booking_results = self.run_booking_prediction_analysis(booking_data_path)
            
            # Component 2: Review Collection
            review_data = self.run_review_collection(scrape_pages, force_scrape)
            
            # Component 3: NLP Analysis
            nlp_results = self.run_nlp_analysis()
            
            # Generate comprehensive report
            final_report = self.generate_comprehensive_report()
            
            logger.info("Complete pipeline execution finished successfully!")
            
            # Print summary
            print("\n" + "="*60)
            print("BRITISH AIRWAYS ANALYTICS - PIPELINE RESULTS")
            print("="*60)
            
            if booking_results:
                print(f"📊 BOOKING PREDICTION:")
                print(f"   • Model Accuracy: {booking_results['accuracy']:.1%}")
                print(f"   • AUC-ROC Score: {booking_results['auc_roc']:.3f}")
                print(f"   • Top Predictor: {booking_results['top_features'][0]['feature']}")
            
            if review_data is not None:
                print(f"\n🕷️  REVIEW COLLECTION:")
                print(f"   • Reviews Collected: {len(review_data):,}")
                print(f"   • Data Source: Skytrax")
                print(f"   • Collection Method: Ethical Web Scraping")
            
            if nlp_results:
                print(f"\n🧠 NLP ANALYSIS:")
                print(f"   • Reviews Analyzed: {nlp_results['total_reviews_analyzed']:,}")
                print(f"   • Average Sentiment: {nlp_results['average_sentiment_score']:.3f}")
                print(f"   • Negative Reviews: {nlp_results['negative_reviews_pct']:.1f}%")
                print(f"   • Topics Identified: {nlp_results['topics_identified']}")
            
            print(f"\n📁 All results saved to: {self.output_dir}/")
            print("="*60)
            
            return final_report
            
        except Exception as e:
            logger.error(f"Error in complete pipeline execution: {e}")
            return None

def main():
    """
    Main execution function for the British Airways analytics project.
    """
    print("British Airways End-to-End Data Science Project")
    print("=" * 60)
    print("Components: Booking Prediction | Web Scraping | NLP Analysis")
    print("=" * 60)
    
    # Initialize analytics pipeline
    ba_analytics = BritishAirwaysAnalytics(output_dir="ba_analytics_results")
    
    # Run complete pipeline
    results = ba_analytics.run_complete_pipeline(
        booking_data_path="customer_booking.csv",
        scrape_pages=5,  # Reduced for demo
        force_scrape=False
    )
    
    if results:
        print("\n✅ Pipeline completed successfully!")
        print("📊 Check the 'ba_analytics_results' folder for detailed outputs")
    else:
        print("\n❌ Pipeline execution failed. Check logs for details.")

if __name__ == "__main__":
    main()

