"""
Model Evaluation Module

Handles:
- Confusion matrix generation and visualization
- Classification reports
- ROC curve plotting
- Feature importance analysis (tree-based models)
- SHAP value analysis for model interpretability
- Generating evaluation reports
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Any

from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    roc_auc_score,accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import shap
import joblib
from sklearn.model_selection import train_test_split

from .data_preprocessing import DataPreprocessor
from .feature_engineering import engineer_features, select_features_for_modeling


class ModelEvaluator:
    """Evaluates and explains trained models."""
    
    def __init__(self, model: Any, X_test: np.ndarray, y_test: np.ndarray,
                 feature_names: list = None, model_name: str = "Model"):
        """
        Initialize evaluator.
        
        Args:
            model: Trained model object
            X_test: Test features
            y_test: Test target
            feature_names: Names of features
            model_name: Name of the model
        """
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.feature_names = feature_names
        self.model_name = model_name
        
        self.y_pred = model.predict(X_test)
        self.y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    def generate_confusion_matrix(self, save_path: str = None) -> np.ndarray:
        """
        Generate and visualize confusion matrix.
        
        Confusion Matrix Interpretation:
        - True Negatives (TN): Correctly predicted no churn
        - False Positives (FP): Predicted churn, but didn't churn (wasted retention $)
        - False Negatives (FN): Predicted no churn, but did churn (lost customers)
        - True Positives (TP): Correctly predicted churn (saved customers)
        
        Args:
            save_path: Optional path to save the figure
        
        Returns:
            Confusion matrix array
        """
        cm = confusion_matrix(self.y_test, self.y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['No Churn', 'Churn'],
                    yticklabels=['No Churn', 'Churn'])
        plt.title(f'Confusion Matrix - {self.model_name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Confusion matrix saved to {save_path}")
        
        plt.close()
        return cm
    
    def generate_classification_report(self) -> str:
        """
        Generate detailed classification report.
        
        Report includes:
        - Precision: TP / (TP + FP) - Of predicted churners, how many actually churned?
        - Recall: TP / (TP + FN) - Of actual churners, how many did we find?
        - F1-Score: 2 * (Precision * Recall) / (Precision + Recall)
        - Support: Number of samples per class
        
        Returns:
            Classification report string
        """
        report = classification_report(self.y_test, self.y_pred,
                                       target_names=['No Churn', 'Churn'])
        return report

    def test_thresholds(self):
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        probabilities = self.y_pred_proba

        thresholds = [0.50, 0.45, 0.40, 0.35, 0.30, 0.25]

        print("\n" + "=" * 60)
        print("THRESHOLD COMPARISON")
        print("=" * 60)

        for threshold in thresholds:
            predictions = (probabilities >= threshold).astype(int)

            accuracy = accuracy_score(self.y_test, predictions)
            precision = precision_score(self.y_test, predictions)
            recall = recall_score(self.y_test, predictions)
            f1 = f1_score(self.y_test, predictions)

            print(f"\nThreshold: {threshold}")
            print(f"Accuracy : {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall   : {recall:.4f}")
            print(f"F1 Score : {f1:.4f}")

    def find_best_threshold(self, X_val, y_val):
        """
        Find the best classification threshold using the validation set.

        The test set is NOT used for threshold selection.
        """

        validation_probabilities = self.model.predict_proba(X_val)[:, 1]

        thresholds = np.arange(0.20, 0.61, 0.05)

        results = []

        print("\n" + "=" * 60)
        print("VALIDATION THRESHOLD TUNING")
        print("=" * 60)

        for threshold in thresholds:

            predictions = (
                validation_probabilities >= threshold
            ).astype(int)

            accuracy = accuracy_score(y_val, predictions)
            precision = precision_score(
                y_val,
                predictions,
                zero_division=0
            )
            recall = recall_score(
                y_val,
                predictions,
                zero_division=0
            )
            f1 = f1_score(
                y_val,
                predictions,
                zero_division=0
            )

            results.append({
                "threshold": threshold,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1
            })

            print(f"\nThreshold: {threshold:.2f}")
            print(f"Accuracy : {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall   : {recall:.4f}")
            print(f"F1 Score : {f1:.4f}")

        results_df = pd.DataFrame(results)

        # Select threshold with highest F1-score
        best_row = results_df.loc[
            results_df["f1"].idxmax()
        ]

        best_threshold = float(best_row["threshold"])

        print("\n" + "-" * 60)
        print("BEST VALIDATION THRESHOLD")
        print("-" * 60)

        print(f"Threshold: {best_threshold:.2f}")
        print(f"Accuracy : {best_row['accuracy']:.4f}")
        print(f"Precision: {best_row['precision']:.4f}")
        print(f"Recall   : {best_row['recall']:.4f}")
        print(f"F1 Score : {best_row['f1']:.4f}")

        return best_threshold, results_df

    def evaluate_with_threshold(self, threshold):
        """
        Evaluate the model on the test set using a threshold
        selected from the validation set.
        """

        test_probabilities = self.model.predict_proba(
            self.X_test
        )[:, 1]

        test_predictions = (
            test_probabilities >= threshold
        ).astype(int)

        accuracy = accuracy_score(
            self.y_test,
            test_predictions
        )

        precision = precision_score(
            self.y_test,
            test_predictions,
            zero_division=0
        )

        recall = recall_score(
            self.y_test,
            test_predictions,
            zero_division=0
        )

        f1 = f1_score(
            self.y_test,
            test_predictions,
            zero_division=0
        )

        print("\n" + "=" * 60)
        print("FINAL TEST EVALUATION")
        print("=" * 60)

        print(f"\nSelected threshold: {threshold:.2f}")
        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1 Score : {f1:.4f}")

        print("\nClassification Report:")
        print(
            classification_report(
                self.y_test,
                test_predictions,
                target_names=["No Churn", "Churn"],
                zero_division=0
            )
        )

        print("Confusion Matrix:")
        print(
            confusion_matrix(
                self.y_test,
                test_predictions
            )
        )

        return {
            "threshold": threshold,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc_score(
                self.y_test,
                test_probabilities
            )
        }
    
    def generate_roc_curve(self, save_path: str = None) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Generate and visualize ROC curve.
        
        ROC Curve Interpretation:
        - X-axis: False Positive Rate (FP / (FP + TN))
        - Y-axis: True Positive Rate (TP / (TP + FN))
        - Diagonal: Random classifier (AUC = 0.5)
        - Curve above diagonal: Good classifier
        - Area Under Curve (AUC): Single metric for model comparison
        
        ROC-AUC Advantages:
        - Threshold-independent
        - Handles class imbalance well
        - Best metric for imbalanced classification
        
        Args:
            save_path: Optional path to save the figure
        
        Returns:
            Tuple of (fpr, tpr, auc_score)
        """
        fpr, tpr, thresholds = roc_curve(self.y_test, self.y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {self.model_name}')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ ROC curve saved to {save_path}")
        
        plt.close()
        return fpr, tpr, roc_auc
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Extract feature importance from tree-based models.
        
        Applicable to:
        - Decision Tree
        - Random Forest
        - XGBoost
        - Not applicable to Logistic Regression
        
        Feature importance interpretation:
        - Higher values = more important for predictions
        - Shows which features the model relies on
        
        Returns:
            DataFrame with features and their importance scores
        """
        if hasattr(self.model, 'named_steps'):
            classifier = self.model.named_steps.get('classifier')
        else:
            classifier = self.model

        if not hasattr(classifier, 'feature_importances_'):
            print(f"⚠️  {self.model_name} does not support feature_importances_")
            return None

        importances = classifier.feature_importances_
        
        # Handle pipeline objects
        if len(importances) != len(self.feature_names):
            print(f"⚠️  Feature importance length mismatch")
            return None
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def plot_feature_importance(self, top_n: int = 15, save_path: str = None) -> None:
        """
        Visualize top N most important features.
        
        Args:
            top_n: Number of top features to show
            save_path: Optional path to save the figure
        """
        importance_df = self.get_feature_importance()
        
        if importance_df is None:
            return
        
        top_features = importance_df.head(top_n)
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance Score')
        plt.title(f'Top {top_n} Feature Importance - {self.model_name}')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Feature importance plot saved to {save_path}")
        
        plt.close()
    
    def shap_analysis(self, max_samples: int = 100, save_path: str = None) -> None:
        """
        Perform SHAP (SHapley Additive exPlanations) analysis.
        
        SHAP Interpretation:
        - Shows how each feature contributes to predictions
        - Red points: Features that increase churn probability
        - Blue points: Features that decrease churn probability
        - Magnitude: Strength of contribution
        
        Advantages over feature importance:
        - Provides per-sample explanations
        - Shows direction (increases/decreases prediction)
        - Theoretically sound (game theory-based)
        
        Args:
            max_samples: Number of samples to use for explanation
            save_path: Optional path to save the figure
        """
        print(f"\n⏳ Generating SHAP explanations (this may take a moment)...")
        
        try:
            # Use subset for faster computation
            X_sample = self.X_test[:min(max_samples, len(self.X_test))]
            
            # Create explainer based on model type
            if hasattr(self.model, 'predict_proba'):
                explainer = shap.TreeExplainer(self.model)
            else:
                explainer = shap.KernelExplainer(
                    self.model.predict_proba if hasattr(self.model, 'predict_proba') 
                    else self.model.predict,
                    X_sample
                )
            
            shap_values = explainer.shap_values(X_sample)
            
            # Handle different output formats
            if isinstance(shap_values, list):
                # For multi-class, take class 1 (churn)
                shap_values = shap_values[1]
            
            # Plot SHAP summary
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_sample, feature_names=self.feature_names,
                            plot_type="bar", show=False)
            plt.title(f'SHAP Feature Importance - {self.model_name}')
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"✓ SHAP analysis saved to {save_path}")
            
            plt.close()
            print("✓ SHAP analysis complete")
            
        except Exception as e:
            print(f"⚠️  Could not generate SHAP explanation: {str(e)}")
    
    def generate_evaluation_summary(self) -> dict:
        """
        Generate complete evaluation summary.
        
        Returns:
            Dictionary with all evaluation metrics and reports
        """
        summary = {
            'model_name': self.model_name,
            'confusion_matrix': confusion_matrix(self.y_test, self.y_pred),
            'classification_report': self.generate_classification_report(),
            'roc_auc': roc_auc_score(self.y_test, self.y_pred_proba),
            'feature_importance': self.get_feature_importance()
        }
        
        return summary


class EvaluationReportGenerator:
    """Generates comprehensive evaluation reports."""
    
    @staticmethod
    def generate_report(model_name: str, results: dict, feature_names: list,
                       preprocessing_steps: list, models_tested: list,
                       cv_results: dict = None) -> str:
        """
        Generate comprehensive evaluation report.
        
        Args:
            model_name: Name of final model
            results: Dictionary with model metrics
            feature_names: List of feature names
            preprocessing_steps: Data preprocessing steps applied
            models_tested: List of models evaluated
            cv_results: Cross-validation results if available
        
        Returns:
            Report as string
        """
        report = f"""
{'='*70}
CUSTOMER CHURN PREDICTION - MODEL EVALUATION REPORT
{'='*70}

1. DATASET INFORMATION
{'-'*70}
Total features: {len(feature_names)}
Feature names: {', '.join(feature_names[:10])}...

2. PREPROCESSING STEPS
{'-'*70}
"""
        for step in preprocessing_steps:
            report += f"  • {step}\n"
        
        report += f"""
3. MODELS TESTED
{'-'*70}
"""
        for model in models_tested:
            report += f"  • {model}\n"
        
        report += f"""
4. TEST SET PERFORMANCE
{'-'*70}
Model: {model_name}

Accuracy:  {results.get('accuracy', 'N/A'):.4f}
Precision: {results.get('precision', 'N/A'):.4f}
Recall:    {results.get('recall', 'N/A'):.4f}
F1-Score:  {results.get('f1', 'N/A'):.4f}
ROC-AUC:   {results.get('roc_auc', 'N/A'):.4f}

5. METRIC INTERPRETATION
{'-'*70}
Accuracy:  Overall correctness (use with caution on imbalanced data)
Precision: Of predicted churners, how many actually churned?
Recall:    Of actual churners, how many did we find?
F1-Score:  Balance between precision and recall
ROC-AUC:   Best metric for imbalanced classification

6. KEY FINDINGS
{'-'*70}
• High precision means fewer false alarms (efficient retention spending)
• High recall means catching most churners (better retention rate)
• ROC-AUC is most reliable metric for this imbalanced problem
• Feature importance shows which factors drive churn

7. BUSINESS RECOMMENDATIONS
{'-'*70}
• Use model predictions to prioritize retention efforts
• Focus on high-value customers identified as at-risk
• Monitor and improve top features affecting churn
• Regularly retrain model with new customer data

{'='*70}
"""
        return report
    
def main():
    print("\n" + "=" * 60)
    print("CUSTOMER CHURN MODEL EVALUATION")
    print("=" * 60)

    # Load the trained model
    model = joblib.load("models/churn_model.pkl")
    print("\n✓ Trained model loaded")

    # Load and preprocess dataset
    preprocessor = DataPreprocessor("data/dataset.csv")
    df, preprocessing_info, feature_info = preprocessor.run_full_preprocessing()

    # Separate features and target
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # Apply the same feature engineering used during training
    X = engineer_features(X)
    X = select_features_for_modeling(X, y)

# ============================================================
# CREATE 60/20/20 TRAIN-VALIDATION-TEST SPLIT
# ============================================================

    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=0.25,
        stratify=y_temp,
        random_state=42
    )

    print(f"✓ Training data prepared: {X_train.shape}")
    print(f"✓ Validation data prepared: {X_val.shape}")
    print(f"✓ Test data prepared: {X_test.shape}")


    # Create evaluator
    evaluator = ModelEvaluator(
        model=model,
        X_test=X_test,
        y_test=y_test,
        feature_names=X_test.columns.tolist(),
        model_name="Logistic Regression + SMOTE"
    )


    # Tune prediction threshold using validation set
    best_threshold, threshold_results = evaluator.find_best_threshold(
        X_val, 
        y_val
    )

    final_results = evaluator.evaluate_with_threshold(
    best_threshold
    )

     # ============================================================
    # FINAL SUMMARY
    # ============================================================

    print("\n" + "=" * 60)
    print("✓ MODEL EVALUATION COMPLETE!")
    print("=" * 60)

    print(
        f"\nFinal threshold: "
        f"{final_results['threshold']:.2f}"
    )

    print(
        f"Test Accuracy: "
        f"{final_results['accuracy']:.4f}"
    )

    print(
        f"Test Precision: "
        f"{final_results['precision']:.4f}"
    )

    print(
        f"Test Recall: "
        f"{final_results['recall']:.4f}"
    )

    print(
        f"Test F1 Score: "
        f"{final_results['f1']:.4f}"
    )

    print(
        f"Test ROC-AUC: "
        f"{final_results['roc_auc']:.4f}"
    )


if __name__ == "__main__":
    main()