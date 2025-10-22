# Evaluation and Simulation Framework

This module provides comprehensive model evaluation metrics and A/B testing simulation capabilities for the Smart Product Re-Ranking System.

## Components

### 1. Model Evaluation Metrics (`model_evaluation_metrics.py`)

Implements ROC-AUC, Precision, Recall calculation functions and comprehensive model performance reporting.

#### Key Features:
- **ROC-AUC, Precision, and Recall Calculations**: Core evaluation metrics with proper handling of edge cases
- **Feature Importance Visualization**: Multi-panel visualizations showing feature rankings, cumulative importance, and distributions
- **ROC and PR Curve Visualizations**: Performance curves with AUC calculations
- **Confusion Matrix Visualization**: Detailed confusion matrix with performance metrics
- **Comprehensive Performance Reporting**: Complete model assessment with grades and recommendations

#### Classes:
- `ModelEvaluationMetrics`: Main evaluation class
- `EvaluationMetrics`: Data container for metrics
- `ModelPerformanceReport`: Comprehensive performance report

#### Usage Example:
```python
from evaluation.model_evaluation_metrics import ModelEvaluationMetrics

evaluator = ModelEvaluationMetrics()

# Calculate basic metrics
metrics = evaluator.calculate_roc_auc_precision_recall(y_true, y_pred_proba)

# Generate comprehensive report
report = evaluator.add_model_performance_reporting(
    y_true, y_pred_proba, feature_importance, 
    model_name="Product_Ranking_Model"
)

# Create visualizations
fig = evaluator.generate_feature_importance_visualization(feature_importance)
```

### 2. A/B Testing System (`ab_testing.py`)

Implements baseline ranking system, ML ranking system, and comprehensive A/B simulation with statistical significance testing.

#### Key Features:
- **Baseline Ranking System**: Ranking by view count for comparison
- **ML Ranking System**: Ranking by predicted relevance scores
- **A/B Simulation**: Complete user behavior simulation with position bias
- **Statistical Significance Testing**: T-tests with confidence intervals
- **Business Impact Analysis**: Revenue impact and improvement quantification
- **Comprehensive Visualizations**: Multi-panel A/B test result displays

#### Classes:
- `BaselineRankingSystem`: Implements view count-based ranking
- `MLRankingSystem`: Implements ML prediction-based ranking
- `RankingABTest`: Complete A/B testing framework
- `ABTestResults`: Results container with business impact analysis

#### Usage Example:
```python
from evaluation.ab_testing import run_ab_test_simulation

# Run complete A/B test
ab_results = run_ab_test_simulation(
    products_df, 
    num_users_per_group=1000,
    items_per_user=10
)

# Access results
print(f"Purchase rate improvement: {ab_results.business_impact['metric_improvements']['purchase_rate']['relative_improvement']:.2f}%")
print(f"Recommendation: {ab_results.business_impact['overall_assessment']['recommendation']}")
```

## Key Metrics Implemented

### Model Evaluation Metrics:
- **ROC-AUC**: Area under the ROC curve (≥0.7 threshold for validation)
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **Accuracy**: Overall classification accuracy
- **Specificity**: True negative rate
- **Balanced Accuracy**: Average of sensitivity and specificity

### A/B Testing Metrics:
- **Click-Through Rate (CTR)**: Clicks / Views
- **Conversion Rate (CVR)**: Purchases / Clicks
- **Purchase Rate**: Purchases / Views
- **Revenue per User**: Average revenue generated per user
- **Statistical Significance**: P-values with confidence intervals
- **Effect Size**: Magnitude of improvement

## Visualization Capabilities

### Model Evaluation Visualizations:
1. **Feature Importance Analysis**:
   - Horizontal bar chart of top features
   - Cumulative importance curve
   - Feature importance distribution
   - Feature category breakdown

2. **Performance Curves**:
   - ROC curve with AUC score
   - Precision-Recall curve with AUC
   - Comparison with random classifier baseline

3. **Confusion Matrix**:
   - Heatmap with counts
   - Performance metrics overlay
   - Class-specific insights

### A/B Testing Visualizations:
1. **Metrics Comparison**:
   - Side-by-side bar charts
   - Statistical significance indicators
   - Improvement percentages

2. **Position Analysis**:
   - Performance by ranking position
   - Position bias effects
   - Method comparison across positions

3. **Business Impact**:
   - Revenue impact visualization
   - Test summary statistics
   - Recommendation display

## Statistical Testing

### Significance Testing:
- **T-tests**: Independent samples t-tests for metric comparisons
- **Confidence Levels**: Configurable (default 95%)
- **Effect Size**: Cohen's d calculation
- **Multiple Testing**: Individual tests for each key metric

### Business Impact Assessment:
- **Minimum Effect Size**: Configurable threshold (default 5%)
- **Revenue Impact**: Estimated revenue lift per user
- **Recommendation Engine**: Automated business recommendations based on results

## Configuration Options

### Model Evaluation Config:
```python
@dataclass
class EvaluationConfig:
    min_roc_auc_threshold: float = 0.7
    confidence_level: float = 0.95
    top_features_display: int = 15
    save_plots: bool = False
    output_directory: str = "evaluation_plots"
```

### A/B Testing Config:
```python
@dataclass
class ABTestConfig:
    num_users_per_group: int = 1000
    items_per_user: int = 10
    simulation_days: int = 30
    confidence_level: float = 0.95
    minimum_effect_size: float = 0.05
    random_state: int = 42
```

## Output Files

### Model Evaluation Outputs:
- `{model_name}_feature_importance.png`: Feature importance visualization
- `{model_name}_roc_pr_curves.png`: ROC and PR curves
- `{model_name}_confusion_matrix.png`: Confusion matrix
- Performance report JSON with detailed metrics

### A/B Testing Outputs:
- `ab_test_results.json`: Complete test results and metrics
- `simulation_data.csv`: Raw simulation data
- `position_analysis.png`: Position-based performance analysis
- Comprehensive A/B test visualization plots

## Requirements Satisfied

This implementation satisfies the following requirements from the specification:

### Requirement 4.1 & 4.2 (Model Evaluation):
- ✅ ROC-AUC, Precision, and Recall calculation functions
- ✅ Visualization generation for feature importance
- ✅ Model performance reporting functionality

### Requirement 4.3, 4.4 & 4.5 (A/B Testing):
- ✅ Baseline ranking system (by view count) for comparison
- ✅ New ranking system (by predicted relevance score)
- ✅ Conversion rate comparison and statistical significance testing
- ✅ Quantified improvement metrics and business impact reports

## Example Usage

See `python/examples/evaluation_ab_testing_example.py` for a complete demonstration of all functionality.

## Dependencies

- pandas: Data manipulation
- numpy: Numerical computations
- scikit-learn: Evaluation metrics
- matplotlib: Plotting
- seaborn: Statistical visualizations
- scipy: Statistical testing