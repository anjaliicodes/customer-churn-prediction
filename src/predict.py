"""
Prediction Module

Handles:
- Loading saved model pipeline
- Making predictions on new customer data
- Returning predictions with probabilities
- Handling input validation
"""

import numpy as np
import pandas as pd
import joblib
from .feature_engineering import engineer_features, select_features_for_modeling
from pathlib import Path
from typing import Tuple, Dict, Any


class ChurnPredictor:
    """Makes churn predictions for individual customers."""
    
    def __init__(self, model_path: str):
        """
        Initialize predictor with saved model.
        
        Args:
            model_path: Path to saved model pickle file
        """
        self.model_path = model_path
        self.model = None
        self.load_model()
    
    def load_model(self) -> None:
        """Load saved model pipeline from pickle file."""
        try:
            self.model = joblib.load(self.model_path)
            print(f"✓ Model loaded from {self.model_path}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")
    
    def predict_single(self, customer_data: Dict[str, Any]) -> Tuple[str, float]:
        """
        Predict churn for a single customer.
        
        Args:
            customer_data: Dictionary with customer features
                          {feature_name: value, ...}
        
        Returns:
            Tuple of (prediction, probability)
            - prediction: "Churn" or "No Churn"
            - probability: Float between 0 and 1
        """
        if self.model is None:
            raise ValueError("Model not loaded")
        
        # Convert to DataFrame (required for preprocessing)
        df = pd.DataFrame([customer_data])
        
        # Make prediction
        prediction_class = self.model.predict(df)[0]
        prediction_proba = self.model.predict_proba(df)[0, 1]
        
        # Convert to readable format
        prediction_label = "Churn" if prediction_class == 1 else "No Churn"
        
        return prediction_label, prediction_proba
    
    def predict_batch(self, customer_data: pd.DataFrame) -> pd.DataFrame:
        """
        Predict churn for multiple customers.
        
        Args:
            customer_data: DataFrame with customer features
        
        Returns:
            DataFrame with predictions and probabilities
        """
        if self.model is None:
            raise ValueError("Model not loaded")

        original_data = customer_data.copy()
        customer_data = engineer_features(customer_data)
        customer_data = select_features_for_modeling(customer_data)

        predictions = self.model.predict(customer_data)
        probabilities = self.model.predict_proba(customer_data)[:, 1]
        
        results = original_data.copy()
        results['prediction'] = ['Churn' if p == 1 else 'No Churn' for p in predictions]
        results['probability'] = probabilities
        
        return results
    
    def predict_with_explanation(self, customer_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict churn with additional details.
        
        Args:
            customer_data: Dictionary with customer features
        
        Returns:
            Dictionary with prediction details
        """
        prediction, probability = self.predict_single(customer_data)
        
        result = {
            'prediction': prediction,
            'churn_probability': probability,
            'retention_probability': 1 - probability,
            'risk_level': self._get_risk_level(probability),
            'recommendation': self._get_recommendation(prediction, probability)
        }
        
        return result
    
    @staticmethod
    def _get_risk_level(probability: float) -> str:
        """
        Classify risk level based on probability.
        
        Args:
            probability: Churn probability
        
        Returns:
            Risk level string
        """
        if probability < 0.3:
            return "Low Risk"
        elif probability < 0.6:
            return "Medium Risk"
        else:
            return "High Risk"
    
    @staticmethod
    def _get_recommendation(prediction: str, probability: float) -> str:
        """
        Provide business recommendation based on prediction.
        
        Args:
            prediction: Churn or No Churn
            probability: Churn probability
        
        Returns:
            Recommendation string
        """
        if prediction == "Churn":
            if probability > 0.8:
                return "URGENT: Immediate retention action recommended"
            else:
                return "Schedule retention discussion with customer"
        else:
            return "No immediate action needed. Monitor for changes"


def predict_from_csv(model_path: str, input_csv: str, output_csv: str = None) -> pd.DataFrame:
    """
    Predict churn for all customers in a CSV file.
    
    Args:
        model_path: Path to saved model
        input_csv: Path to input CSV with customer data
        output_csv: Path to save predictions (optional)
    
    Returns:
        DataFrame with predictions
    """
    # Load data
    df = pd.read_csv(input_csv)
    
    # Initialize predictor
    predictor = ChurnPredictor(model_path)
    
    # Make predictions
    results = predictor.predict_batch(df)
    
    # Save if output path provided
    if output_csv:
        results.to_csv(output_csv, index=False)
        print(f"✓ Predictions saved to {output_csv}")
    
    return results


def main():
    """Example usage of the ChurnPredictor."""
    
    # Initialize predictor
    predictor = ChurnPredictor("models/churn_model.pkl")
    
    # Example 1: Single customer prediction
    print("\n" + "="*60)
    print("EXAMPLE 1: SINGLE CUSTOMER PREDICTION")
    print("="*60)
    
    customer = {
        'gender': 'Male',
        'SeniorCitizen': 0,
        'Partner': 'Yes',
        'Dependents': 'No',
        'tenure': 2,
        'PhoneService': 'No',
        'InternetService': 'Fiber optic',
        'OnlineSecurity': 'No',
        'OnlineBackup': 'No',
        'DeviceProtection': 'No',
        'TechSupport': 'No',
        'StreamingTV': 'No',
        'StreamingMovies': 'No',
        'Contract': 'Month-to-month',
        'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 29.85,
        'TotalCharges': 29.85
    }
    
    # Make prediction
    prediction, probability = predictor.predict_single(customer)
    
    print(f"\nPrediction: {prediction}")
    print(f"Probability: {probability:.4f}")
    
    # Get detailed explanation
    details = predictor.predict_with_explanation(customer)
    print(f"\nDetailed Result:")
    print(f"  Churn Probability:      {details['churn_probability']:.4f}")
    print(f"  Retention Probability:  {details['retention_probability']:.4f}")
    print(f"  Risk Level:             {details['risk_level']}")
    print(f"  Recommendation:         {details['recommendation']}")
    
    # Example 2: Batch prediction from CSV
    print("\n" + "="*60)
    print("EXAMPLE 2: BATCH PREDICTION FROM CSV")
    print("="*60)
    
    try:
        results = predict_from_csv(
            model_path="models/churn_model.pkl",
            input_csv="data/new_customers.csv",
            output_csv="predictions/churn_predictions.csv"
        )
        print(f"\n✓ Predictions completed for {len(results)} customers")
        print(f"\nFirst 5 predictions:")
        print(results[['prediction', 'probability']].head())
        
    except FileNotFoundError as e:
        print(f"Note: {e}")
        print("This example requires 'data/new_customers.csv'")


if __name__ == "__main__":
    main()
