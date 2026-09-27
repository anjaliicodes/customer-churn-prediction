"""
Data Preprocessing Module

Handles:
- Loading data
- Removing unnecessary columns
- Handling missing values
- Converting data types
- Removing duplicates
- Converting target variable to numerical form
- Detecting data quality issues
"""

import pandas as pd
import numpy as np
from typing import Tuple


class DataPreprocessor:
    """Preprocesses raw customer churn data."""
    
    def __init__(self, filepath: str):
        """
        Initialize preprocessor with data file.
        
        Args:
            filepath: Path to CSV file
        """
        self.filepath = filepath
        self.df = None
        self.preprocessing_info = {}
    
    def load_data(self) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Returns:
            Loaded DataFrame
        """
        self.df = pd.read_csv(self.filepath)
        print(f"✓ Data loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        return self.df
    
    def check_data_quality(self) -> dict:
        """
        Perform initial data quality checks.
        
        Returns:
            Dictionary with quality metrics
        """
        quality_report = {
            'total_rows': self.df.shape[0],
            'total_columns': self.df.shape[1],
            'duplicates': self.df.duplicated().sum(),
            'missing_values': self.df.isnull().sum().sum(),
            'memory_usage_mb': self.df.memory_usage(deep=True).sum() / 1024**2
        }
        
        print("\n📊 DATA QUALITY CHECK:")
        print(f"  Total rows: {quality_report['total_rows']}")
        print(f"  Total columns: {quality_report['total_columns']}")
        print(f"  Duplicate rows: {quality_report['duplicates']}")
        print(f"  Total missing values: {quality_report['missing_values']}")
        print(f"  Memory usage: {quality_report['memory_usage_mb']:.2f} MB")
        
        # Show missing values by column
        missing_by_col = self.df.isnull().sum()
        if missing_by_col.sum() > 0:
            print("\n  Missing values by column:")
            print(missing_by_col[missing_by_col > 0])
        
        return quality_report
    
    def remove_duplicates(self) -> None:
        """Remove exact duplicate rows."""
        before = self.df.shape[0]
        self.df = self.df.drop_duplicates()
        after = self.df.shape[0]
        removed = before - after
        
        if removed > 0:
            print(f"✓ Removed {removed} duplicate rows")
        self.preprocessing_info['duplicates_removed'] = removed
    
    def remove_unnecessary_columns(self) -> None:
        """
        Remove columns that don't contribute to prediction.
        
        Rationale:
        - customerID: Just a unique identifier, no predictive value
        """
        cols_to_drop = ['customerID']
        existing_cols = [col for col in cols_to_drop if col in self.df.columns]
        
        if existing_cols:
            self.df = self.df.drop(columns=existing_cols)
            print(f"✓ Removed unnecessary columns: {existing_cols}")
        
        self.preprocessing_info['columns_removed'] = existing_cols
    
    def handle_missing_values(self) -> None:
        """
        Handle missing values.
        
        Strategy:
        - For numeric: Forward fill or mean imputation
        - For categorical: Mode imputation
        """
        missing_cols = self.df.columns[self.df.isnull().any()].tolist()
        
        if not missing_cols:
            print("✓ No missing values found")
            return
        
        print(f"✓ Handling missing values in: {missing_cols}")
        
        for col in missing_cols:
            if self.df[col].dtype in ['float64', 'int64']:
                self.df[col].fillna(self.df[col].mean(), inplace=True)
            else:
                self.df[col].fillna(self.df[col].mode()[0], inplace=True)
        
        self.preprocessing_info['missing_values_handled'] = missing_cols
    
    def fix_data_types(self) -> None:
        """
        Convert columns to appropriate data types.
        
        Fixes:
        - TotalCharges: Should be numeric (sometimes stored as string)
        - SeniorCitizen: Should be numeric (0/1)
        """
        print("✓ Fixing data types...")
        
        # Fix TotalCharges if it exists and is object type
        if 'TotalCharges' in self.df.columns :
            self.df['TotalCharges'] = pd.to_numeric(self.df['TotalCharges'], errors='coerce')
            # Fill any NaN created by coercion with 0
            self.df['TotalCharges'] = self.df['TotalCharges'].fillna(0)
        
        # Ensure SeniorCitizen is numeric
        if 'SeniorCitizen' in self.df.columns:
            self.df['SeniorCitizen'] = pd.to_numeric(self.df['SeniorCitizen'], errors='coerce')
        
        self.preprocessing_info['data_types_fixed'] = True
    
    def preprocess_target_variable(self) -> None:
        """
        Convert target variable (Churn) from Yes/No to 1/0.
        
        Mapping:
        - Yes → 1 (Churned)
        - No → 0 (Retained)
        """
        if 'Churn' in self.df.columns:
            self.df['Churn'] = (self.df['Churn'] == 'Yes').astype(int)
            churn_dist = self.df['Churn'].value_counts()
            print(f"✓ Target variable converted to numerical (0/1)")
            print(f"  Class distribution:")
            print(f"    No Churn (0): {churn_dist.get(0, 0)} ({churn_dist.get(0, 0)/len(self.df)*100:.1f}%)")
            print(f"    Churn (1):    {churn_dist.get(1, 0)} ({churn_dist.get(1, 0)/len(self.df)*100:.1f}%)")
            
            self.preprocessing_info['class_distribution'] = churn_dist.to_dict()
    
    def get_feature_info(self) -> dict:
        """
        Generate detailed feature information.
        
        Returns:
            Dictionary with feature types and statistics
        """
        feature_info = {
            'numerical_features': self.df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_features': self.df.select_dtypes(include=['object']).columns.tolist(),
            'target_variable': 'Churn' if 'Churn' in self.df.columns else None,
        }
        
        print(f"\n📋 FEATURE INFORMATION:")
        print(f"  Numerical features ({len(feature_info['numerical_features'])}): {feature_info['numerical_features']}")
        print(f"  Categorical features ({len(feature_info['categorical_features'])}): {feature_info['categorical_features']}")
        print(f"  Target variable: {feature_info['target_variable']}")
        
        return feature_info
    
    def run_full_preprocessing(self) -> Tuple[pd.DataFrame, dict, dict]:
        """
        Execute complete preprocessing pipeline.
        
        Returns:
            Tuple of (cleaned_dataframe, preprocessing_info, feature_info)
        """
        print("\n" + "="*60)
        print("🔄 STARTING DATA PREPROCESSING PIPELINE")
        print("="*60)
        
        self.load_data()
        self.check_data_quality()
        self.remove_duplicates()
        self.remove_unnecessary_columns()
        self.handle_missing_values()
        self.fix_data_types()
        self.preprocess_target_variable()
        feature_info = self.get_feature_info()
        
        print(f"\n✅ PREPROCESSING COMPLETE")
        print(f"   Final shape: {self.df.shape[0]} rows × {self.df.shape[1]} columns")
        print("="*60 + "\n")
        
        return self.df, self.preprocessing_info, feature_info
