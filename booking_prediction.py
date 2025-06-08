"""
British Airways Customer Booking Prediction Model
=================================================

This module implements a Random Forest classifier to predict customer booking completion
for British Airways. The model analyzes 50,000 booking records with 14 features to 
identify customers likely to complete their bookings.

Author: Data Science Portfolio
Project: British Airways End-to-End Analytics
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class BookingPredictor:
    """
    A machine learning model to predict customer booking completion.
    
    This class handles data preprocessing, model training, and evaluation
    for predicting whether customers will complete their flight bookings.
    """
    
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.feature_importance = None
        
    def load_and_preprocess_data(self, filepath):
        """
        Load and preprocess the booking data.
        
        Args:
            filepath (str): Path to the CSV file containing booking data
            
        Returns:
            tuple: Preprocessed features (X) and target variable (y)
        """
        # Load the dataset
        df = pd.read_csv(filepath, encoding="ISO-8859-1")
        
        print(f"Dataset loaded: {df.shape[0]} records, {df.shape[1]} features")
        print(f"Missing values: {df.isnull().sum().sum()}")
        
        # Convert flight_day to numerical values
        day_mapping = {
            "Mon": 1, "Tue": 2, "Wed": 3, "Thu": 4, 
            "Fri": 5, "Sat": 6, "Sun": 7
        }
        df["flight_day"] = df["flight_day"].map(day_mapping)
        
        # Encode categorical variables
        categorical_columns = ['sales_channel', 'trip_type', 'route', 'booking_origin']
        
        for column in categorical_columns:
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column])
            self.label_encoders[column] = le
        
        # Separate features and target
        X = df.drop('booking_complete', axis=1)
        y = df['booking_complete']
        
        print(f"Features: {list(X.columns)}")
        print(f"Target distribution: {y.value_counts().to_dict()}")
        
        return X, y
    
    def train_model(self, X, y, test_size=0.2, random_state=42):
        """
        Train the Random Forest model with cross-validation.
        
        Args:
            X (DataFrame): Feature matrix
            y (Series): Target variable
            test_size (float): Proportion of data for testing
            random_state (int): Random seed for reproducibility
            
        Returns:
            dict: Training results and metrics
        """
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        # Initialize and train the model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1
        )
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring='accuracy')
        
        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = self.model.score(X_test, y_test)
        auc_roc = roc_auc_score(y_test, y_pred_proba)
        
        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Results
        results = {
            'accuracy': accuracy,
            'auc_roc': auc_roc,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'classification_report': classification_report(y_test, y_pred),
            'feature_importance': self.feature_importance,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred_proba': y_pred_proba
        }
        
        return results
    
    def plot_feature_importance(self, top_n=10):
        """
        Plot the top N most important features.
        
        Args:
            top_n (int): Number of top features to display
        """
        plt.figure(figsize=(10, 6))
        top_features = self.feature_importance.head(top_n)
        
        sns.barplot(data=top_features, x='importance', y='feature', palette='viridis')
        plt.title(f'Top {top_n} Feature Importance - Booking Prediction Model')
        plt.xlabel('Importance Score')
        plt.ylabel('Features')
        plt.tight_layout()
        plt.show()
    
    def plot_roc_curve(self, y_test, y_pred_proba):
        """
        Plot the ROC curve for model evaluation.
        
        Args:
            y_test (array): True labels
            y_pred_proba (array): Predicted probabilities
        """
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Booking Prediction Model')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def predict_booking_completion(self, customer_data):
        """
        Predict booking completion probability for new customers.
        
        Args:
            customer_data (DataFrame): Customer features
            
        Returns:
            array: Prediction probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained. Please train the model first.")
        
        return self.model.predict_proba(customer_data)[:, 1]

def main():
    """
    Main execution function demonstrating the complete workflow.
    """
    print("British Airways Booking Prediction Model")
    print("=" * 50)
    
    # Initialize the predictor
    predictor = BookingPredictor()
    
    # Load and preprocess data
    X, y = predictor.load_and_preprocess_data("customer_booking.csv")
    
    # Train the model
    results = predictor.train_model(X, y)
    
    # Display results
    print(f"\nModel Performance:")
    print(f"Accuracy: {results['accuracy']:.3f}")
    print(f"AUC-ROC: {results['auc_roc']:.3f}")
    print(f"Cross-validation: {results['cv_mean']:.3f} (+/- {results['cv_std']:.3f})")
    
    print(f"\nTop 5 Most Important Features:")
    print(results['feature_importance'].head())
    
    # Generate visualizations
    predictor.plot_feature_importance()
    predictor.plot_roc_curve(results['y_test'], results['y_pred_proba'])
    
    print(f"\nClassification Report:")
    print(results['classification_report'])

if __name__ == "__main__":
    main()

