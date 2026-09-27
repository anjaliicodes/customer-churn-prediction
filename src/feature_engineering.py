"""
Feature Engineering Module

Creates meaningful features to improve model performance:
- Tenure groups (customer segments by loyalty)
- Service count (total number of services)
- Charge ratio (average per month vs total)
- Service adoption rate
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def engineer_features(X):
    """
    Create new meaningful features from existing ones.
    
    Features Created:
    1. tenure_group: Segment customers by tenure duration
    2. total_services: Count of services subscribed
    3. charge_ratio: Monthly vs total charge ratio
    4. has_security_features: Any security feature subscribed
    5. has_streaming_services: Any streaming service subscribed
    
    Args:
        X (pd.DataFrame): Feature matrix
        
    Returns:
        pd.DataFrame: Enhanced feature matrix
    """
    X = X.copy()
    
    print("\nEngineering features...")
    
    # Feature 1: Tenure Groups
    # Segment customers based on how long they've been with the company
    if 'tenure' in X.columns:
                X['tenure_group'] = pd.cut(
                    X['tenure'],
                    bins=[-1, 6, 12, 24, 48, 72],
                    labels=[
                        '0-6_months',
                        '6-12_months',
                        '1-2_years',
                        '2-4_years',
                        '4+_years'
                   ],
                   include_lowest=True
                )
    X['tenure_group'] = X['tenure_group'].astype(str)
    print("  ✓ Created tenure_group feature")
    
    # Feature 2: Total Services Count
    # Count how many services each customer has subscribed to
    service_cols = [
        'PhoneService', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    
    # Filter to only existing columns
    service_cols = [col for col in service_cols if col in X.columns]
    
    if service_cols:
        # Count services where value is 'Yes' or similar
        X['total_services'] = 0
        for col in service_cols:
            if col == 'InternetService':
                # InternetService has 3 values: Fiber optic, DSL, No
                X['total_services'] += (X[col] != 'No').astype(int)
            elif col == 'PhoneService':
                X['total_services'] += (X[col] == 'Yes').astype(int)
            else:
                X['total_services'] += (X[col] == 'Yes').astype(int)
        
        print("  ✓ Created total_services feature")
    
    # Feature 3: Charge Ratio
    # Relationship between monthly and total charges
    if 'MonthlyCharges' in X.columns and 'TotalCharges' in X.columns:
        X['charge_ratio'] = X['MonthlyCharges'] / (X['TotalCharges'] + 1)
        print("  ✓ Created charge_ratio feature")
    
    # Feature 4: Has Security Features
    security_features = ['OnlineSecurity', 'DeviceProtection']
    security_features = [col for col in security_features if col in X.columns]
    
    if security_features:
        X['has_security'] = 0
        for col in security_features:
            X['has_security'] += (X[col] == 'Yes').astype(int)
        X['has_security'] = (X['has_security'] > 0).astype(int)
        print("  ✓ Created has_security feature")
    
    # Feature 5: Has Streaming Services
    streaming_features = ['StreamingTV', 'StreamingMovies']
    streaming_features = [col for col in streaming_features if col in X.columns]
    
    if streaming_features:
        X['has_streaming'] = 0
        for col in streaming_features:
            X['has_streaming'] += (X[col] == 'Yes').astype(int)
        X['has_streaming'] = (X['has_streaming'] > 0).astype(int)
        print("  ✓ Created has_streaming feature")
    
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"New features created")

        # Additional customer behavior features

    X['avg_monthly_charge'] = np.where(
        X['tenure'] > 0,
        X['TotalCharges'] / X['tenure'],
        X['MonthlyCharges']
    )

    X['tenure_monthly_interaction'] = (
        X['tenure'] * X['MonthlyCharges']
    )

    X['services_monthly_interaction'] = (
        X['total_services'] * X['MonthlyCharges']
    )

    X['new_customer'] = (
        X['tenure'] <= 6
    ).astype(int)

    X['long_term_customer'] = (
        X['tenure'] >= 48
    ).astype(int)

    X['month_to_month'] = (
        X['Contract'] == 'Month-to-month'
    ).astype(int)

    X['electronic_payment'] = (
        X['PaymentMethod'] == 'Electronic check'
    ).astype(int)

    X['no_security_support'] = (
        (X['OnlineSecurity'] == 'No') &
        (X['TechSupport'] == 'No')
    ).astype(int)

    X['fiber_without_security'] = (
        (X['InternetService'] == 'Fiber optic') &
        (X['OnlineSecurity'] == 'No')
    ).astype(int)
    
    return X


def select_features_for_modeling(X, y=None):
    """
    Select relevant features for modeling.
    
    Exclusion logic:
    - Drop original service columns (replaced by engineered features)
    - Keep numerical and categorical features needed for prediction
    - Drop highly correlated features (if applicable)
    
    Args:
        X (pd.DataFrame): Feature matrix
        y (pd.Series, optional): Target variable
        
    Returns:
        pd.DataFrame: Selected features
    """
    X = X.copy()
    
    print("\nSelecting features for modeling...")
    
    # List of columns to potentially drop
    drop_cols = []
    
    # Drop tenure_group_original if tenure_group exists (keep engineered feature)
    # In this case, we keep both tenure and tenure_group
    
    # Don't drop individual service indicators that might be useful
    # They'll be handled in preprocessing (OneHotEncoding)
    
    # Check for columns with very low variance or all same values
    for col in X.columns:
        if X[col].nunique() == 1:
            drop_cols.append(col)
            print(f"  ✓ Dropped {col} (no variance)")
    
    if drop_cols:
        X = X.drop(columns=drop_cols)
    
    print(f"\nFinal feature matrix shape: {X.shape}")
    print(f"Columns: {X.columns.tolist()}")
    
    return X


def analyze_feature_distributions(X, y):
    """
    Analyze feature distributions and correlation with target.
    
    Args:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target variable
    """
    print("\n" + "="*60)
    print("FEATURE ANALYSIS")
    print("="*60)
    
    # Numerical features summary
    numerical_cols = X.select_dtypes(include=[np.number]).columns
    print(f"\nNumerical Features ({len(numerical_cols)}):")
    print(X[numerical_cols].describe())
    
    # Categorical features
    categorical_cols = X.select_dtypes(include=['object']).columns
    print(f"\nCategorical Features ({len(categorical_cols)}):")
    for col in categorical_cols:
        print(f"\n{col}:")
        print(X[col].value_counts())


def main():
    """
    Main function to demonstrate feature engineering.
    """
    from data_preprocessing import DataPreprocessor
    
    # Load and preprocess data
    preprocessor = DataPreprocessor('data/dataset.csv')
    df, preprocessing_info, feature_info = preprocessor.run_full_preprocessing()
    
    # Separate features and target
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    
    # Engineer features
    X_engineered = engineer_features(X)
    
    # Select features
    X_selected = select_features_for_modeling(X_engineered, y)
    
    # Analyze
    analyze_feature_distributions(X_selected, y)
    
    print("\nFeature engineering complete!")


if __name__ == '__main__':
    main()