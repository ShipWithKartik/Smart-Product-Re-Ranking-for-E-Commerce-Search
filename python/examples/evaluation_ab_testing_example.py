"""
Example usage of Model Evaluation Metrics and A/B Testing System
Demonstrates comprehensive evaluation and A/B simulation functionality
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.model_evaluation_metrics import (
    ModelEvaluationMetrics, evaluate_model_performance, create_evaluation_visualizations
)
from evaluation.ab_testing import (
    RankingABTest, BaselineRankingSystem, MLRankingSystem, 
    run_ab_test_simulation, create_ranking_comparison_report
)


def create_sample_data():
    """Create sample data for demonstration"""
    print("Creating sample data for evaluation and A/B testing...")
    
    np.random.seed(42)
    n_products = 500
    
    # Create sample product data
    products_data = {
        'itemid': range(1, n_products + 1),
        'view_count': np.random.poisson(100, n_products),
        'addtocart_count': np.random.poisson(10, n_products),
        'transaction_count': np.random.poisson(2, n_products),
        'unique_visitors': np.random.poisson(80, n_products),
    }
    
    products_df = pd.DataFrame(products_data)
    
    # Calculate behavioral metrics
    products_df['addtocart_rate'] = products_df['addtocart_count'] / products_df['view_count']
    products_df['conversion_rate'] = products_df['transaction_count'] / products_df['view_count']
    products_df['cart_conversion_rate'] = products_df['transaction_count'] / products_df['addtocart_count']
    
    # Fill NaN values
    products_df = products_df.fillna(0)
    
    # Create performance labels (High performing = top 30% by transaction count)
    threshold = products_df['transaction_count'].quantile(0.7)
    products_df['High_Performing_Product'] = (products_df['transaction_count'] >= threshold).astype(int)
    
    # Simulate ML predictions (with some correlation to actual performance)
    base_score = products_df['transaction_count'] / products_df['transaction_count'].max()
    noise = np.random.normal(0, 0.2, len(products_df))
    products_df['relevance_score'] = np.clip(base_score + noise, 0, 1)
    
    print(f"Created sample data with {len(products_df)} products")
    print(f"High performing products: {products_df['High_Performing_Product'].sum()}")
    
    return products_df


def demonstrate_model_evaluation():
    """Demonstrate model evaluation metrics functionality"""
    print("\n" + "="*60)
    print("DEMONSTRATING MODEL EVALUATION METRICS")
    print("="*60)
    
    # Create sample data
    products_df = create_sample_data()
    
    # Prepare data for evaluation
    y_true = products_df['High_Performing_Product'].values
    y_pred_proba = products_df['relevance_score'].values
    
    # Create feature importance data
    feature_names = ['view_count', 'addtocart_rate', 'conversion_rate', 'cart_conversion_rate', 'unique_visitors']
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': np.random.dirichlet(np.ones(len(feature_names))) * 10  # Random importance scores
    }).sort_values('importance', ascending=False)
    
    print(f"\nEvaluating model performance...")
    print(f"True labels: {len(y_true)} samples")
    print(f"Predictions: {len(y_pred_proba)} samples")
    
    # Initialize evaluator
    evaluator = ModelEvaluationMetrics()
    
    # Calculate basic metrics
    print("\n1. Calculating ROC-AUC, Precision, and Recall...")
    evaluation_metrics = evaluator.calculate_roc_auc_precision_recall(y_true, y_pred_proba)
    
    print(f"   ROC-AUC: {evaluation_metrics.roc_auc:.4f}")
    print(f"   Precision: {evaluation_metrics.precision:.4f}")
    print(f"   Recall: {evaluation_metrics.recall:.4f}")
    print(f"   F1-Score: {evaluation_metrics.f1_score:.4f}")
    print(f"   Accuracy: {evaluation_metrics.accuracy:.4f}")
    
    # Generate comprehensive report
    print("\n2. Generating comprehensive performance report...")
    performance_report = evaluator.add_model_performance_reporting(
        y_true, y_pred_proba, feature_importance, 
        model_name="Demo_Product_Ranking_Model", 
        model_type="Logistic_Regression"
    )
    
    print(f"   Performance Grade: {performance_report.performance_grade}")
    print(f"   Validation Passed: {performance_report.validation_passed}")
    print(f"   Top Feature: {performance_report.feature_importance.iloc[0]['feature']}")
    print(f"   Recommendations: {len(performance_report.recommendations)} items")
    
    # Create visualizations
    print("\n3. Creating evaluation visualizations...")
    
    # Feature importance visualization
    feature_fig = evaluator.generate_feature_importance_visualization(
        feature_importance, "Demo Product Ranking Model"
    )
    
    # ROC and PR curves
    roc_pr_fig = evaluator.create_roc_curve_visualization(
        y_true, y_pred_proba, "Demo Product Ranking Model"
    )
    
    # Confusion matrix
    y_pred = (y_pred_proba >= 0.5).astype(int)
    confusion_fig = evaluator.create_confusion_matrix_visualization(
        y_true, y_pred, "Demo Product Ranking Model"
    )
    
    print("   ✓ Feature importance visualization created")
    print("   ✓ ROC and PR curves created")
    print("   ✓ Confusion matrix created")
    
    # Generate comprehensive evaluation report
    print("\n4. Generating comprehensive evaluation report...")
    comprehensive_results = evaluator.generate_comprehensive_evaluation_report(
        y_true, y_pred_proba, feature_importance,
        model_name="Demo_Product_Ranking_Model",
        model_type="Logistic_Regression",
        save_plots=True,
        output_dir="evaluation_plots"
    )
    
    print(f"   ✓ Comprehensive report generated")
    print(f"   ✓ Plots saved to evaluation_plots directory")
    
    return products_df, comprehensive_results


def demonstrate_ab_testing(products_df):
    """Demonstrate A/B testing functionality"""
    print("\n" + "="*60)
    print("DEMONSTRATING A/B TESTING SYSTEM")
    print("="*60)
    
    # Initialize ranking systems
    print("\n1. Initializing ranking systems...")
    baseline_system = BaselineRankingSystem()
    ml_system = MLRankingSystem()
    
    # Test baseline ranking
    print("\n2. Testing baseline ranking system...")
    baseline_ranked = baseline_system.rank_by_view_count(products_df)
    print(f"   ✓ Ranked {len(baseline_ranked)} products by view count")
    print(f"   Top 3 products by baseline: {baseline_ranked.head(3)['itemid'].tolist()}")
    
    # Test ML ranking
    print("\n3. Testing ML ranking system...")
    ml_ranked = ml_system.rank_by_ml_predictions(products_df)
    print(f"   ✓ Ranked {len(ml_ranked)} products by ML predictions")
    print(f"   Top 3 products by ML: {ml_ranked.head(3)['itemid'].tolist()}")
    
    # Run A/B test simulation
    print("\n4. Running A/B test simulation...")
    ab_results = run_ab_test_simulation(
        products_df, 
        num_users_per_group=500,  # Smaller for demo
        items_per_user=8,
        confidence_level=0.95
    )
    
    print(f"   ✓ A/B test completed")
    print(f"   Total users simulated: {ab_results.test_config.num_users_per_group * 2}")
    print(f"   Total interactions: {len(ab_results.simulation_data)}")
    
    # Display key results
    print("\n5. A/B Test Results Summary:")
    baseline_metrics = ab_results.baseline_metrics
    ml_metrics = ab_results.ml_ranking_metrics
    
    print(f"\n   Baseline Performance:")
    print(f"   - Click-through rate: {baseline_metrics['click_through_rate']:.4f}")
    print(f"   - Conversion rate: {baseline_metrics['conversion_rate']:.4f}")
    print(f"   - Purchase rate: {baseline_metrics['purchase_rate']:.4f}")
    print(f"   - Avg purchases per user: {baseline_metrics['avg_purchases_per_user']:.4f}")
    
    print(f"\n   ML Ranking Performance:")
    print(f"   - Click-through rate: {ml_metrics['click_through_rate']:.4f}")
    print(f"   - Conversion rate: {ml_metrics['conversion_rate']:.4f}")
    print(f"   - Purchase rate: {ml_metrics['purchase_rate']:.4f}")
    print(f"   - Avg purchases per user: {ml_metrics['avg_purchases_per_user']:.4f}")
    
    # Statistical significance
    significance = ab_results.statistical_significance
    print(f"\n   Statistical Significance:")
    print(f"   - Purchase count test: p-value = {significance['purchase_count']['p_value']:.4f}, "
          f"significant = {significance['purchase_count']['significant']}")
    print(f"   - CTR test: p-value = {significance['click_through_rate']['p_value']:.4f}, "
          f"significant = {significance['click_through_rate']['significant']}")
    
    # Business impact
    business_impact = ab_results.business_impact
    print(f"\n   Business Impact:")
    print(f"   - Purchase rate improvement: {business_impact['metric_improvements']['purchase_rate']['relative_improvement']:.2f}%")
    print(f"   - Revenue lift per user: ${business_impact['revenue_impact']['revenue_lift_per_user']:.2f}")
    print(f"   - Recommendation: {business_impact['overall_assessment']['recommendation']}")
    
    # Create visualizations
    print("\n6. Creating A/B test visualizations...")
    ab_test = RankingABTest()
    ab_test.plot_ab_results(ab_results)
    
    position_fig = ab_test.plot_position_analysis(ab_results.simulation_data)
    
    print("   ✓ A/B test result plots created")
    print("   ✓ Position analysis plot created")
    
    return ab_results


def demonstrate_comprehensive_report():
    """Demonstrate comprehensive reporting functionality"""
    print("\n" + "="*60)
    print("DEMONSTRATING COMPREHENSIVE REPORTING")
    print("="*60)
    
    # Create sample data
    products_df = create_sample_data()
    
    print("\n1. Creating comprehensive ranking comparison report...")
    
    # Generate comprehensive report
    comprehensive_report = create_ranking_comparison_report(
        products_df,
        save_plots=True,
        output_dir="comprehensive_ab_results"
    )
    
    print(f"   ✓ Comprehensive report generated")
    print(f"   ✓ Results saved to comprehensive_ab_results directory")
    
    # Display summary
    print(f"\n2. Report Summary:")
    print(f"   - Test duration: {comprehensive_report['test_config']['simulation_days']} days")
    print(f"   - Users tested: {comprehensive_report['test_config']['num_users_per_group'] * 2}")
    print(f"   - Primary metric significant: {comprehensive_report['statistical_significance']['summary']['primary_metric_significant']}")
    print(f"   - Business recommendation: {comprehensive_report['business_impact']['overall_assessment']['recommendation']}")
    
    return comprehensive_report


def main():
    """Main demonstration function"""
    print("Smart Product Re-Ranking System - Evaluation & A/B Testing Demo")
    print("="*70)
    
    try:
        # Demonstrate model evaluation
        products_df, evaluation_results = demonstrate_model_evaluation()
        
        # Demonstrate A/B testing
        ab_results = demonstrate_ab_testing(products_df)
        
        # Demonstrate comprehensive reporting
        comprehensive_report = demonstrate_comprehensive_report()
        
        print("\n" + "="*70)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("="*70)
        print("\nGenerated outputs:")
        print("- evaluation_plots/: Model evaluation visualizations")
        print("- comprehensive_ab_results/: A/B test results and plots")
        print("\nKey findings:")
        print(f"- Model ROC-AUC: {evaluation_results['summary']['roc_auc']:.4f}")
        print(f"- Model validation: {'PASSED' if evaluation_results['summary']['validation_passed'] else 'FAILED'}")
        print(f"- A/B test recommendation: {ab_results.business_impact['overall_assessment']['recommendation']}")
        
    except Exception as e:
        print(f"\nError during demonstration: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()