"""
Model Training Module

Trains multiple ML models:
- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

Features:
- Handles class imbalance (SMOTE)
- Hyperparameter tuning with GridSearchCV
- Stratified K-Fold cross-validation
- Model comparison and selection
"""

import numpy as np
import pandas as pd
import joblib
import os
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings

warnings.filterwarnings('ignore')


from .data_preprocessing import DataPreprocessor
from .feature_engineering import (
    engineer_features,
    select_features_for_modeling
)

class ModelTrainer:
    """
    Main class for training and evaluating ML models.
    """
    
    def __init__(self, random_state=42):
        """
        Initialize ModelTrainer.
        
        Args:
            random_state (int): Random seed for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.cv_results = {}
        self.tuned_models = {}
        
    def train_validation_test_split_data(self, X, y, test_size=0.2):
        """
        Split data into train/validation/test sets with stratification.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target
            test_size (float): Test set proportion
            
        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        print("\n" + "="*60)
        print("TRAIN/VALIDATION/TEST SPLIT")
        print("="*60)
        
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=0.20,
            stratify=y,
            random_state=self.random_state
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=0.25,
            stratify=y_temp,
            random_state=self.random_state
        )

        print(f"✓ Training data prepared: {X_train.shape}")
        print(f"✓ Validation data prepared: {X_val.shape}")
        print(f"✓ Test data prepared: {X_test.shape}")
        
        print(f"\nTraining set: {X_train.shape[0]} samples ({(1-0.40)*100:.0f}%)")
        print(f"\nValidation set: {X_val.shape[0]} samples ({0.20*100:.0f}%)")
        print(f"\nTest set: {X_test.shape[0]} samples ({0.20*100:.0f}%)")
        print(f"\nTraining set class distribution:")
        print(y_train.value_counts())
        print(f"\nValidation set class distribution:")
        print(y_val.value_counts())
        print(f"\nTest set class distribution:")
        print(y_test.value_counts())
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def check_class_imbalance(self, y_train, y_test):
        """
        Check if the target classes are imbalanced.
        
        Args:
            y_train (pd.Series): Training target
            y_test (pd.Series): Test target
        """
        print("\n" + "="*60)
        print("CLASS IMBALANCE ANALYSIS")
        print("="*60)
        
        churn_ratio_train = (y_train == 1).sum() / len(y_train)
        churn_ratio_test = (y_test == 1).sum() / len(y_test)
        
        print(f"\nChurn ratio (training): {churn_ratio_train:.2%}")
        print(f"Churn ratio (test): {churn_ratio_test:.2%}")
        
        if churn_ratio_train < 0.4 or churn_ratio_train > 0.6:
            print("\n⚠️  Classes are imbalanced. Will apply SMOTE.")
            return True
        else:
            print("\n✓ Classes are relatively balanced.")
            return False
    
    def train_baseline_models(self, X_train, X_test, y_train, y_test, 
                             numerical_features, categorical_features):
        """
        Train baseline models without hyperparameter tuning.
        
        Models:
        1. Logistic Regression
        2. Decision Tree
        3. Random Forest
        4. XGBoost
        
        Args:
            X_train, X_test: Feature matrices
            y_train, y_test: Target arrays
            numerical_features: List of numerical columns
            categorical_features: List of categorical columns
        """
        print("\n" + "="*60)
        print("TRAINING BASELINE MODELS")
        print("="*60)
        
        # Create preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
            ],
            remainder='passthrough'
        )
        
        # Model 1: Logistic Regression
        print("\n1. Logistic Regression...")
        lr_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(max_iter=1000, random_state=self.random_state))
        ])
        lr_pipeline.fit(X_train, y_train)
        self.models['Logistic Regression'] = lr_pipeline
        print("   ✓ Trained")
        
        # Model 2: Decision Tree
        print("2. Decision Tree...")
        dt_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', DecisionTreeClassifier(random_state=self.random_state))
        ])
        dt_pipeline.fit(X_train, y_train)
        self.models['Decision Tree'] = dt_pipeline
        print("   ✓ Trained")
        
        # Model 3: Random Forest
        print("3. Random Forest...")
        rf_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=self.random_state))
        ])
        rf_pipeline.fit(X_train, y_train)
        self.models['Random Forest'] = rf_pipeline
        print("   ✓ Trained")
        
        # Model 4: XGBoost
        print("4. XGBoost...")
        xgb_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', XGBClassifier(n_estimators=100, random_state=self.random_state, verbosity=0))
        ])
        xgb_pipeline.fit(X_train, y_train)
        self.models['XGBoost'] = xgb_pipeline
        print("   ✓ Trained")
    
    def train_models_with_smote(self, X_train, X_test, y_train, y_test,
                               numerical_features, categorical_features):
        """
        Train models with SMOTE to handle class imbalance.
        
        IMPORTANT: SMOTE is applied only to training data, never to test set.
        
        Args:
            X_train, X_test: Feature matrices
            y_train, y_test: Target arrays
            numerical_features: List of numerical columns
            categorical_features: List of categorical columns
        """
        print("\n" + "="*60)
        print("TRAINING MODELS WITH SMOTE")
        print("="*60)
        print("\nNote: SMOTE applied only to training data (prevents data leakage)")
        
        # Create preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
            ],
            remainder='passthrough'
        )
        
        # Model 1: Logistic Regression with SMOTE
        print("\n1. Logistic Regression + SMOTE...")
        lr_smote_pipeline = ImbPipeline([
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=self.random_state)),
            ('classifier', LogisticRegression(max_iter=1000, random_state=self.random_state))
        ])
        lr_smote_pipeline.fit(X_train, y_train)
        self.models['Logistic Regression + SMOTE'] = lr_smote_pipeline
        print("   ✓ Trained")
        
        # Model 2: Random Forest with SMOTE
        print("2. Random Forest + SMOTE...")
        rf_smote_pipeline = ImbPipeline([
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=self.random_state)),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=self.random_state))
        ])
        rf_smote_pipeline.fit(X_train, y_train)
        self.models['Random Forest + SMOTE'] = rf_smote_pipeline
        print("   ✓ Trained")
        
        # Model 3: XGBoost with SMOTE
        print("3. XGBoost + SMOTE...")
        xgb_smote_pipeline = ImbPipeline([
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=self.random_state)),
            ('classifier', XGBClassifier(n_estimators=100, random_state=self.random_state, verbosity=0))
        ])
        xgb_smote_pipeline.fit(X_train, y_train)
        self.models['XGBoost + SMOTE'] = xgb_smote_pipeline
        print("   ✓ Trained")
    
    def perform_cross_validation(self, X_train, y_train, cv_folds=5):
        """
        Perform stratified k-fold cross-validation.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            cv_folds (int): Number of cross-validation folds
        """
        print("\n" + "="*60)
        print("CROSS-VALIDATION")
        print("="*60)
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        for model_name, model in self.models.items():
            print(f"\n{model_name}:")
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
            self.cv_results[model_name] = scores
            print(f"  CV ROC-AUC: {scores.mean():.4f} (+/- {scores.std():.4f})")
    
    def hyperparameter_tuning(self, X_train, y_train, 
                             numerical_features, categorical_features):
        """
        Perform hyperparameter tuning on promising models.
        
        Tuned models:
        - Random Forest
        - XGBoost
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            numerical_features: List of numerical columns
            categorical_features: List of categorical columns
        """
        print("\n" + "="*60)
        print("HYPERPARAMETER TUNING")
        print("="*60)
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
            ],
            remainder='passthrough'
        )

                # Tune Logistic Regression
        print("\nTuning Logistic Regression...")

        lr_params = {
            'classifier__C': [0.01, 0.1, 0.5, 1, 2, 5, 10],
            'classifier__class_weight': [None, 'balanced']
        }

        lr_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(
                max_iter=2000,
                random_state=self.random_state
            ))
        ])

        lr_grid = GridSearchCV(
            lr_pipeline,
            lr_params,
            cv=5,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=0
        )

        lr_grid.fit(X_train, y_train)

        self.tuned_models['Logistic Regression'] = lr_grid.best_estimator_

        print(f"  Best ROC-AUC: {lr_grid.best_score_:.4f}")
        print(f"  Best params: {lr_grid.best_params_}")
        
        # Tune Random Forest
        print("\nTuning Random Forest...")
        rf_params = {
            'classifier__n_estimators': [50, 100],
            'classifier__max_depth': [10, 15, 20],
            'classifier__min_samples_split': [2, 5]
        }
        
        rf_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(random_state=self.random_state))
        ])
        
        rf_grid = GridSearchCV(
            rf_pipeline, rf_params, cv=5, scoring='roc_auc', n_jobs=-1, verbose=0
        )
        rf_grid.fit(X_train, y_train)
        self.tuned_models['Random Forest'] = rf_grid.best_estimator_
        print(f"  Best ROC-AUC: {rf_grid.best_score_:.4f}")
        print(f"  Best params: {rf_grid.best_params_}")
        
        # Tune XGBoost
        print("\nTuning XGBoost...")
        xgb_params = {
            'classifier__n_estimators': [50, 100],
            'classifier__max_depth': [5, 7],
            'classifier__learning_rate': [0.01, 0.1],
            'classifier__subsample': [0.8, 1.0]
        }
        
        xgb_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', XGBClassifier(random_state=self.random_state, verbosity=0))
        ])
        
        xgb_grid = GridSearchCV(
            xgb_pipeline, xgb_params, cv=5, scoring='roc_auc', n_jobs=-1, verbose=0
        )
        xgb_grid.fit(X_train, y_train)
        self.tuned_models['XGBoost'] = xgb_grid.best_estimator_
        print(f"  Best ROC-AUC: {xgb_grid.best_score_:.4f}")
        print(f"  Best params: {xgb_grid.best_params_}")
    
    def get_all_models(self):
        """
        Get all trained models (baseline + tuned).
        
        Returns:
            dict: Dictionary of all models
        """
        all_models = self.models.copy()
        all_models.update(self.tuned_models)
        return all_models


    def evaluate_on_validation(self, models, X_val, y_val):
        print("\n" + "=" * 60)
        print("VALIDATION SET EVALUATION")
        print("=" * 60)

        validation_results = {}

        for model_name, model in models.items():
            y_val_proba = model.predict_proba(X_val)[:, 1]

            auc = roc_auc_score(y_val, y_val_proba)

            print(f"\n{model_name}:")
            print(f"  Validation ROC-AUC: {auc:.4f}")

            validation_results[model_name] = {
                "model": model,
                "roc_auc": auc,
                "probabilities": y_val_proba
            }

        return validation_results

    def select_best_model(self, validation_results):
        """
        Select the best model based on validation ROC-AUC.

        Args:
            validation_results (dict): Results from validation evaluation

        Returns:
            tuple: (best_model_name, best_model)
        """

        print("\n" + "=" * 60)
        print("MODEL SELECTION")
        print("=" * 60)

        # Display all validation results
        print("\nAll validation results:")

        for name, result in validation_results.items():
            print(f"  {name}: {result['roc_auc']:.4f}")

        best_model_name = max(
            validation_results,
            key=lambda name: validation_results[name]["roc_auc"]
        )

        best_model = validation_results[
            best_model_name
        ]["model"]

        best_auc = validation_results[
            best_model_name
        ]["roc_auc"]

        print(
            f"\n✓ Best model: {best_model_name}"
        )

        print(
            f"✓ Validation ROC-AUC: {best_auc:.4f}"
        )

        ranked_models = sorted(
            validation_results.items(),
            key=lambda item: item[1]["roc_auc"],
            reverse=True
        )

        print("\nModel ranking by validation ROC-AUC: ")

        for rank, (name, result) in enumerate(
            ranked_models,
            start=1
        ):
            print(
                f"{rank}. {name}: "
                f"{result['roc_auc']:.4f}"
            )

        return best_model_name, best_model


    def save_model(self, model, model_name):
        """
        Save the selected trained model.

        Args:
            model: Trained sklearn pipeline
            model_name (str): Name of the model
        """

        os.makedirs(
            "models",
            exist_ok=True
        )

        model_path = "models/churn_model.pkl"

        joblib.dump(
            model,
            model_path
        )

        print("\n" + "=" * 60)
        print("MODEL SAVING")
        print("=" * 60)

        print(
            f"\n✓ Selected model: {model_name}"
        )

        print(
            f"✓ Model saved to: {model_path}"
        )


def main():
    """
    Main function to demonstrate model training.
    """
   

    # Load and preprocess data
    preprocessor = DataPreprocessor('data/dataset.csv')
    df, preprocessing_info, feature_info =( 
        preprocessor.run_full_preprocessing()
    )


    X = df.drop(columns=['Churn'])
    y = df['Churn']


    # Feature engineering
    X = engineer_features(X)
    X = select_features_for_modeling(X, y)
    
    # Identify feature types
    numerical_features = X.select_dtypes(
        include=['int64', 'float64']
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=['object', 'category', 'bool']
    ).columns.tolist()
    
    # Initialize trainer
    trainer = ModelTrainer(random_state=42)
    
    # Train/validation/test split
    X_train,X_val, X_test, y_train, y_val, y_test = trainer.train_validation_test_split_data(X, y)
    
    # Check class imbalance
    is_imbalanced = trainer.check_class_imbalance(y_train, y_test)
    
    # Train baseline models
    trainer.train_baseline_models(X_train, X_test, y_train, y_test,
                                 numerical_features, categorical_features)
    
    # Train models with SMOTE if imbalanced
    if is_imbalanced:
        trainer.train_models_with_smote(X_train, X_test, y_train, y_test,
                                       numerical_features, categorical_features)
    
    # Cross-validation
    trainer.perform_cross_validation(X_train, y_train)
    
    # Hyperparameter tuning
    trainer.hyperparameter_tuning(X_train, y_train, 
                                 numerical_features, categorical_features)


    # Get all baseline + tuned models
    all_models = trainer.get_all_models()

    # Evaluate models on validation set
    validation_results = trainer.evaluate_on_validation(
        all_models,
        X_val,
        y_val
    )

    # ============================================================
    # SAVE BEST MODEL
    # ============================================================

    best_model_name, best_model = trainer.select_best_model(
    validation_results
    )

    trainer.save_model(
        best_model,
        best_model_name
    )
    
     # ============================================================
    # TRAINING COMPLETE
    # ============================================================

    print("\n" + "=" * 60)
    print("✓ MODEL TRAINING COMPLETE!")
    print("=" * 60)

    print(
        f"\nSelected model: {best_model_name}"
    )

    print(
        f"Validation ROC-AUC: "
        f"{validation_results[best_model_name]['roc_auc']:.4f}"
    )

    print(
        "\nModel is ready for evaluation."
    )



if __name__ == '__main__':
    main()
