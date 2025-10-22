"""
Model Evaluation Metrics for Smart Product Re-Ranking System
Implements ROC-AUC, Precision, Recall calculation functions and visualization generation
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from dataclasses import dataclass
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Container for model evaluation metrics"""
    roc_auc: float
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    specificity: float
    balanced_accuracy: float
    confusion_matrix: np.ndarray
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'roc_auc': float(self.roc_auc),
            'precision': float(self.precision),
            'recall': float(self.recall),
            'f1_score': float(self.f1_score),
            'accuracy': float(self.accuracy),
            'specificity': float(self.specificity),
            'balanced_accuracy': float(self.balanced_accuracy),
            'confusion_matrix': self.confusion_matrix.tolist()
        }


@dataclass
class ModelPerformanceReport:
    """Comprehensive model performance report"""
    model_name: str
    model_type: str
    evaluation_metrics: EvaluationMetrics
    feature_importance: pd.DataFrame
    performance_grade: str
    recommendations: List[str]
    validation_passed: bool
    threshold_analysis: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'model_name': self.model_name,
            'model_type': self.model_type,
            'evaluation_metrics': self.evaluation_metrics.to_dict(),
            'feature_importance': self.feature_importance.to_dict('records'),
            'performance_grade': self.performance_grade,
            'recommendations': self.recommendations,
            'validation_passed': self.validation_passed,
            'threshold_analysis': self.threshold_analysis
        }


class ModelEvaluationMetrics:
    """Class for calculating and visualizing model evaluation metrics"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        np.random.seed(random_state)
    
    def calculate_roc_auc_precision_recall(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                                         y_pred: Optional[np.ndarray] = None,
                                         threshold: float = 0.5) -> EvaluationMetrics:
        """
        Create ROC-AUC, Precision, and Recall calculation functions
        
        Args:
            y_true: True binary labels
            y_pred_proba: Predicted probabilities for positive class
            y_pred: Predicted binary labels (optional, will be calculated from probabilities)
            threshold: Threshold for converting probabilities to binary predictions
            
        Returns:
            EvaluationMetrics object with all calculated metrics
        """
        logger.info("Calculating ROC-AUC, Precision, and Recall metrics")
        
        # Validate inputs
        if len(y_true) != len(y_pred_proba):
            raise ValueError("y_true and y_pred_proba must have the same length")
        
        # Convert probabilities to binary predictions if not provided
        if y_pred is None:
            y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Calculate ROC-AUC
        try:
            roc_auc = roc_auc_score(y_true, y_pred_proba)
        except ValueError as e:
            logger.warning(f"ROC-AUC calculation failed: {str(e)}")
            roc_auc = 0.5  # Random performance
        
        # Calculate Precision and Recall
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Calculate additional metrics
        accuracy = (y_pred == y_true).mean()
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Calculate specificity (True Negative Rate)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            balanced_accuracy = (recall + specificity) / 2
        else:
            specificity = 0
            balanced_accuracy = accuracy
        
        metrics = EvaluationMetrics(
            roc_auc=roc_auc,
            precision=precision,
            recall=recall,
            f1_score=f1,
            accuracy=accuracy,
            specificity=specificity,
            balanced_accuracy=balanced_accuracy,
            confusion_matrix=cm
        )
        
        logger.info(f"Metrics calculated - ROC-AUC: {roc_auc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
        return metrics
    
    def generate_feature_importance_visualization(self, feature_importance: pd.DataFrame, 
                                                model_name: str = "Model",
                                                top_n: int = 15,
                                                figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Implement visualization generation for feature importance
        
        Args:
            feature_importance: DataFrame with 'feature' and 'importance' columns
            model_name: Name of the model for the title
            top_n: Number of top features to display
            figsize: Figure size tuple
            
        Returns:
            Matplotlib figure object
        """
        logger.info(f"Generating feature importance visualization for {model_name}")
        
        # Select top N features
        top_features = feature_importance.head(top_n).copy()
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(f'Feature Importance Analysis - {model_name}', fontsize=16, fontweight='bold')
        
        # Plot 1: Horizontal bar chart of top features
        ax1 = axes[0, 0]
        bars = ax1.barh(range(len(top_features)), top_features['importance'], 
                       color='steelblue', alpha=0.7)
        ax1.set_yticks(range(len(top_features)))
        ax1.set_yticklabels(top_features['feature'])
        ax1.set_xlabel('Importance Score')
        ax1.set_title('Top Feature Importance')
        ax1.grid(axis='x', alpha=0.3)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax1.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                    f'{width:.3f}', ha='left', va='center', fontsize=9)
        
        # Plot 2: Cumulative importance
        ax2 = axes[0, 1]
        cumulative_importance = top_features['importance'].cumsum() / top_features['importance'].sum()
        ax2.plot(range(1, len(top_features) + 1), cumulative_importance, 
                marker='o', linewidth=2, markersize=6, color='darkgreen')
        ax2.set_xlabel('Number of Features')
        ax2.set_ylabel('Cumulative Importance Ratio')
        ax2.set_title('Cumulative Feature Importance')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='80% threshold')
        ax2.legend()
        
        # Plot 3: Feature importance distribution
        ax3 = axes[1, 0]
        ax3.hist(feature_importance['importance'], bins=20, alpha=0.7, 
                color='orange', edgecolor='black')
        ax3.set_xlabel('Importance Score')
        ax3.set_ylabel('Number of Features')
        ax3.set_title('Feature Importance Distribution')
        ax3.grid(axis='y', alpha=0.3)
        
        # Plot 4: Feature categories (if identifiable)
        ax4 = axes[1, 1]
        categories = self._categorize_features_for_viz(top_features)
        if categories:
            category_importance = {}
            for category, features in categories.items():
                category_importance[category] = sum(
                    top_features[top_features['feature'].isin(features)]['importance']
                )
            
            wedges, texts, autotexts = ax4.pie(
                category_importance.values(), 
                labels=category_importance.keys(),
                autopct='%1.1f%%',
                startangle=90,
                colors=plt.cm.Set3(np.linspace(0, 1, len(category_importance)))
            )
            ax4.set_title('Importance by Feature Category')
        else:
            ax4.text(0.5, 0.5, 'Feature categories\nnot identifiable', 
                    ha='center', va='center', transform=ax4.transAxes,
                    fontsize=12, style='italic')
            ax4.set_title('Feature Categories')
        
        plt.tight_layout()
        
        logger.info(f"Feature importance visualization generated with {len(top_features)} features")
        return fig
    
    def create_roc_curve_visualization(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                                     model_name: str = "Model",
                                     figsize: Tuple[int, int] = (12, 5)) -> plt.Figure:
        """
        Create ROC curve and Precision-Recall curve visualizations
        
        Args:
            y_true: True binary labels
            y_pred_proba: Predicted probabilities for positive class
            model_name: Name of the model for the title
            figsize: Figure size tuple
            
        Returns:
            Matplotlib figure object
        """
        logger.info(f"Generating ROC and PR curve visualizations for {model_name}")
        
        # Calculate curves
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_pred_proba)
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_pred_proba)
        
        # Calculate AUC scores
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        
        # Calculate PR AUC using trapezoidal rule
        pr_auc = np.trapz(precision, recall)
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle(f'Model Performance Curves - {model_name}', fontsize=16, fontweight='bold')
        
        # ROC Curve
        ax1 = axes[0]
        ax1.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.3f})', color='darkorange')
        ax1.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curve')
        ax1.legend(loc="lower right")
        ax1.grid(True, alpha=0.3)
        
        # Precision-Recall Curve
        ax2 = axes[1]
        ax2.plot(recall, precision, linewidth=2, label=f'PR Curve (AUC = {pr_auc:.3f})', color='darkblue')
        
        # Baseline (random classifier performance)
        baseline = np.sum(y_true) / len(y_true)
        ax2.axhline(y=baseline, color='k', linestyle='--', linewidth=1, 
                   label=f'Random Classifier ({baseline:.3f})')
        
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curve')
        ax2.legend(loc="lower left")
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        logger.info(f"ROC and PR curves generated - ROC AUC: {roc_auc:.3f}, PR AUC: {pr_auc:.3f}")
        return fig
    
    def create_confusion_matrix_visualization(self, y_true: np.ndarray, y_pred: np.ndarray,
                                            model_name: str = "Model",
                                            class_names: Optional[List[str]] = None,
                                            figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
        """
        Create confusion matrix visualization
        
        Args:
            y_true: True binary labels
            y_pred: Predicted binary labels
            model_name: Name of the model for the title
            class_names: Names for the classes
            figsize: Figure size tuple
            
        Returns:
            Matplotlib figure object
        """
        logger.info(f"Generating confusion matrix visualization for {model_name}")
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        if class_names is None:
            class_names = ['Low Performing', 'High Performing']
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names,
                   ax=ax, cbar_kws={'label': 'Count'})
        
        ax.set_title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        
        # Add performance metrics as text
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            metrics_text = f'Accuracy: {accuracy:.3f}\nPrecision: {precision:.3f}\nRecall: {recall:.3f}'
            ax.text(1.05, 0.5, metrics_text, transform=ax.transAxes, 
                   verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        logger.info("Confusion matrix visualization generated")
        return fig
    
    def add_model_performance_reporting(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                                      feature_importance: pd.DataFrame,
                                      model_name: str = "Model",
                                      model_type: str = "Unknown",
                                      min_roc_auc_threshold: float = 0.7) -> ModelPerformanceReport:
        """
        Add model performance reporting functionality
        
        Args:
            y_true: True binary labels
            y_pred_proba: Predicted probabilities for positive class
            feature_importance: DataFrame with feature importance
            model_name: Name of the model
            model_type: Type of the model
            min_roc_auc_threshold: Minimum ROC-AUC threshold for validation
            
        Returns:
            ModelPerformanceReport object
        """
        logger.info(f"Creating comprehensive performance report for {model_name}")
        
        # Calculate evaluation metrics
        evaluation_metrics = self.calculate_roc_auc_precision_recall(y_true, y_pred_proba)
        
        # Determine performance grade
        performance_grade = self._get_performance_grade(evaluation_metrics.roc_auc)
        
        # Check if validation passed
        validation_passed = evaluation_metrics.roc_auc >= min_roc_auc_threshold
        
        # Generate recommendations
        recommendations = self._generate_performance_recommendations(evaluation_metrics, validation_passed)
        
        # Threshold analysis
        threshold_analysis = self._analyze_optimal_threshold(y_true, y_pred_proba)
        
        report = ModelPerformanceReport(
            model_name=model_name,
            model_type=model_type,
            evaluation_metrics=evaluation_metrics,
            feature_importance=feature_importance,
            performance_grade=performance_grade,
            recommendations=recommendations,
            validation_passed=validation_passed,
            threshold_analysis=threshold_analysis
        )
        
        logger.info(f"Performance report created - Grade: {performance_grade}, Validation: {'PASSED' if validation_passed else 'FAILED'}")
        return report
    
    def generate_comprehensive_evaluation_report(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                                               feature_importance: pd.DataFrame,
                                               model_name: str = "Model",
                                               model_type: str = "Unknown",
                                               save_plots: bool = False,
                                               output_dir: str = "evaluation_plots") -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report with all visualizations
        
        Args:
            y_true: True binary labels
            y_pred_proba: Predicted probabilities for positive class
            feature_importance: DataFrame with feature importance
            model_name: Name of the model
            model_type: Type of the model
            save_plots: Whether to save plots to files
            output_dir: Directory to save plots
            
        Returns:
            Dictionary with complete evaluation results
        """
        logger.info(f"Generating comprehensive evaluation report for {model_name}")
        
        # Create performance report
        performance_report = self.add_model_performance_reporting(
            y_true, y_pred_proba, feature_importance, model_name, model_type
        )
        
        # Generate visualizations
        feature_importance_fig = self.generate_feature_importance_visualization(
            feature_importance, model_name
        )
        
        roc_pr_fig = self.create_roc_curve_visualization(
            y_true, y_pred_proba, model_name
        )
        
        y_pred = (y_pred_proba >= 0.5).astype(int)
        confusion_matrix_fig = self.create_confusion_matrix_visualization(
            y_true, y_pred, model_name
        )
        
        # Save plots if requested
        if save_plots:
            import os
            os.makedirs(output_dir, exist_ok=True)
            
            feature_importance_fig.savefig(f"{output_dir}/{model_name}_feature_importance.png", 
                                         dpi=300, bbox_inches='tight')
            roc_pr_fig.savefig(f"{output_dir}/{model_name}_roc_pr_curves.png", 
                              dpi=300, bbox_inches='tight')
            confusion_matrix_fig.savefig(f"{output_dir}/{model_name}_confusion_matrix.png", 
                                       dpi=300, bbox_inches='tight')
            
            logger.info(f"Plots saved to {output_dir}")
        
        # Compile comprehensive results
        comprehensive_results = {
            'model_name': model_name,
            'model_type': model_type,
            'performance_report': performance_report.to_dict(),
            'visualizations': {
                'feature_importance_available': True,
                'roc_pr_curves_available': True,
                'confusion_matrix_available': True
            },
            'summary': {
                'roc_auc': performance_report.evaluation_metrics.roc_auc,
                'precision': performance_report.evaluation_metrics.precision,
                'recall': performance_report.evaluation_metrics.recall,
                'f1_score': performance_report.evaluation_metrics.f1_score,
                'performance_grade': performance_report.performance_grade,
                'validation_passed': performance_report.validation_passed,
                'top_feature': feature_importance.iloc[0]['feature'] if not feature_importance.empty else 'N/A'
            }
        }
        
        logger.info("Comprehensive evaluation report generated successfully")
        return comprehensive_results
    
    def _categorize_features_for_viz(self, feature_importance: pd.DataFrame) -> Dict[str, List[str]]:
        """Categorize features for visualization purposes"""
        categories = {
            'Behavioral': [],
            'Conversion': [],
            'Engagement': [],
            'Temporal': [],
            'Other': []
        }
        
        for _, row in feature_importance.iterrows():
            feature_name = row['feature'].lower()
            
            if any(term in feature_name for term in ['view', 'addtocart', 'transaction', 'count']):
                categories['Behavioral'].append(row['feature'])
            elif any(term in feature_name for term in ['rate', 'conversion', 'ratio']):
                categories['Conversion'].append(row['feature'])
            elif any(term in feature_name for term in ['engagement', 'intensity', 'loyalty', 'popularity']):
                categories['Engagement'].append(row['feature'])
            elif any(term in feature_name for term in ['time', 'temporal', 'duration', 'avg']):
                categories['Temporal'].append(row['feature'])
            else:
                categories['Other'].append(row['feature'])
        
        # Remove empty categories
        return {k: v for k, v in categories.items() if v}
    
    def _get_performance_grade(self, roc_auc: float) -> str:
        """Get performance grade based on ROC-AUC score"""
        if roc_auc >= 0.9:
            return 'Excellent'
        elif roc_auc >= 0.8:
            return 'Good'
        elif roc_auc >= 0.7:
            return 'Fair'
        elif roc_auc >= 0.6:
            return 'Poor'
        else:
            return 'Very Poor'
    
    def _generate_performance_recommendations(self, metrics: EvaluationMetrics, 
                                            validation_passed: bool) -> List[str]:
        """Generate performance recommendations based on metrics"""
        recommendations = []
        
        if not validation_passed:
            recommendations.append("Model performance below threshold - consider feature engineering or more data")
        
        if metrics.precision < 0.7:
            recommendations.append("Low precision - model has many false positives, consider adjusting threshold")
        
        if metrics.recall < 0.7:
            recommendations.append("Low recall - model misses many positive cases, consider class balancing")
        
        if metrics.f1_score < 0.7:
            recommendations.append("Low F1-score - poor balance between precision and recall")
        
        if metrics.roc_auc < 0.6:
            recommendations.append("Poor discriminative ability - review feature selection and data quality")
        
        if abs(metrics.precision - metrics.recall) > 0.2:
            recommendations.append("Imbalanced precision/recall - consider threshold optimization")
        
        return recommendations if recommendations else ["Model performance is satisfactory"]
    
    def _analyze_optimal_threshold(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """Analyze optimal threshold for classification"""
        thresholds = np.arange(0.1, 1.0, 0.05)
        threshold_metrics = []
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            threshold_metrics.append({
                'threshold': threshold,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            })
        
        # Find optimal threshold (maximize F1-score)
        best_threshold_idx = np.argmax([m['f1_score'] for m in threshold_metrics])
        optimal_threshold = threshold_metrics[best_threshold_idx]
        
        return {
            'optimal_threshold': optimal_threshold['threshold'],
            'optimal_f1_score': optimal_threshold['f1_score'],
            'optimal_precision': optimal_threshold['precision'],
            'optimal_recall': optimal_threshold['recall'],
            'threshold_analysis': threshold_metrics
        }


# Convenience functions
def evaluate_model_performance(y_true: np.ndarray, y_pred_proba: np.ndarray,
                             feature_importance: pd.DataFrame,
                             model_name: str = "Model") -> Dict[str, Any]:
    """
    Convenience function to evaluate model performance
    
    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities for positive class
        feature_importance: DataFrame with feature importance
        model_name: Name of the model
        
    Returns:
        Dictionary with evaluation results
    """
    evaluator = ModelEvaluationMetrics()
    return evaluator.generate_comprehensive_evaluation_report(
        y_true, y_pred_proba, feature_importance, model_name
    )


def create_evaluation_visualizations(y_true: np.ndarray, y_pred_proba: np.ndarray,
                                   feature_importance: pd.DataFrame,
                                   model_name: str = "Model") -> Tuple[plt.Figure, plt.Figure, plt.Figure]:
    """
    Convenience function to create all evaluation visualizations
    
    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities for positive class
        feature_importance: DataFrame with feature importance
        model_name: Name of the model
        
    Returns:
        Tuple of (feature_importance_fig, roc_pr_fig, confusion_matrix_fig)
    """
    evaluator = ModelEvaluationMetrics()
    
    feature_importance_fig = evaluator.generate_feature_importance_visualization(
        feature_importance, model_name
    )
    
    roc_pr_fig = evaluator.create_roc_curve_visualization(
        y_true, y_pred_proba, model_name
    )
    
    y_pred = (y_pred_proba >= 0.5).astype(int)
    confusion_matrix_fig = evaluator.create_confusion_matrix_visualization(
        y_true, y_pred, model_name
    )
    
    return feature_importance_fig, roc_pr_fig, confusion_matrix_fig


if __name__ == "__main__":
    print("Model Evaluation Metrics module loaded successfully!")
    print("Use ModelEvaluationMetrics class to evaluate model performance")
    print("Available functions:")
    print("- calculate_roc_auc_precision_recall()")
    print("- generate_feature_importance_visualization()")
    print("- create_roc_curve_visualization()")
    print("- add_model_performance_reporting()")