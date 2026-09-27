"""
Customer Churn Prediction - ML Pipeline

This package contains modules for:
- Data preprocessing
- Feature engineering
- Model training and evaluation
- Prediction on new data
"""

__version__ = "1.0.0"
__author__ = "ML Project"

from . import data_preprocessing
from . import feature_engineering
from . import train_model
from . import evaluate_model
from . import predict

__all__ = [
    "data_preprocessing",
    "feature_engineering",
    "train_model",
    "evaluate_model",
    "predict",
]