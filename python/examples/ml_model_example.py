"""
Example usage of the ML Model Component for Smart Product Re-Ranking System
Demonstrates performance labeling, model training, and prediction pipeline
"""

import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.behavioral_metrics import calculate_behavioral_metrics
from data.csv_loader import load_events_csv
from models.performance_labeling import create_performance_labels, analyze_label_distribution
from models.model_training_pipeline import compare_all_models, train_model_pipeline
from models.prediction_scoring_system import (
    save_trained_model, load_model_and_predict, 
    PredictionPipeline, list_saved_models
)


def demonstrate_ml_pipeline():
    """Demonstrate the complete ML pipeline"""
    
    print("=" * 60)
    print("Smart Product Re-Ranking ML Pipeline Demo")
    print("=" * 60)
    
    # Step 1: Load and process data
    print("\n1. Loading and processing event data...")
    try:
        events_df = load_events_csv("../../events.csv")
        print(f"   ✓ Loaded {len(events_df)} events")
        
        # Calculate behavioral metrics
        behavioral_metrics = calculate_behavioral_metrics(events_df)
        print(f"   ✓ Calculated behavioral metrics for {len(behavioral_metrics)} items")
        
    except FileNotFoundError:
        print("   ⚠ events.csv not found. Creating synthetic data for demo...")
        behavioral_metrics = create_synthetic_behavioral_data()
        print(f"   ✓ Created synthetic behavioral metrics for {len(behavioral_metrics)} items")
    
    # Step 2: Performance labeling
    print("\n2. Creating performance labels...")
    labeled_data = create_performance_labels(
        behavioral_metrics, 
        quantile_threshold=0.75,
        min_views_threshold=5
    )
    print(f"   ✓ Created performance labels for {len(labeled_data)} items")
    
    # Analyze label distribution
    distribution = analyze_label_distribution(labeled_data)
    print(f"   ✓ High performing items: {distribution['label_distribution']['high_performing']}")
    print(f"   ✓ Low performing items: {distribution['label_distribution']['low_performing']}")
    
    # Step 3: Model training and comparison
    print("\n3. Training and comparing models...")
    
    # Prepare features and target
    feature_columns = [
        'view_count', 'addtocart_count', 'transaction_count', 
        'unique_visitors', 'addtocart_rate', 'conversion_rate', 'cart_conversion_rate'
    ]
    
    X = labeled_data[feature_columns]
    y = labeled_data['High_Performing_Product']
    
    print(f"   ✓ Prepared {X.shape[1]} features for {len(X)} items")
    
    # Compare models
    comparison_results = compare_all_models(X, y, min_roc_auc=0.7)
    print(f"   ✓ Model comparison completed")
    
    print("\n   Model Comparison Results:")
    for _, row in comparison_results.iterrows():
        status = "✓ PASSED" if row['validation_passed'] else "✗ FAILED"
        print(f"     {row['model_type']}: AUC={row['test_auc']:.4f} {status}")
    
    # Select best model
    best_model_row = comparison_results.loc[comparison_results['test_auc'].idxmax()]
    best_model_type = best_model_row['model_type']
    print(f"   ✓ Best model: {best_model_type} (AUC: {best_model_row['test_auc']:.4f})")
    
    # Step 4: Detailed training of best model
    print(f"\n4. Training detailed pipeline for {best_model_type}...")
    
    detailed_results = train_model_pipeline(X, y, best_model_type, min_roc_auc=0.7)
    
    print(f"   ✓ Training completed")
    print(f"   ✓ Test AUC: {detailed_results['training_results'].test_auc:.4f}")
    print(f"   ✓ Validation: {detailed_results['validation_results']['overall_validation']}")
    
    # Show top features
    top_features = detailed_results['feature_importance_analysis']['top_features'][:3]
    print(f"   ✓ Top 3 features:")
    for i, feature in enumerate(top_features, 1):
        print(f"     {i}. {feature['feature']}: {feature['importance']:.4f}")
    
    # Step 5: Model persistence
    print("\n5. Saving trained model...")
    
    model_path = save_trained_model(
        model=detailed_results['training_results'].model_object,
        model_name="demo_product_ranking_model",
        model_type=best_model_type,
        feature_names=feature_columns,
        scaler=detailed_results['training_results'].scaler_object,
        performance_metrics={
            'test_auc': detailed_results['training_results'].test_auc,
            'validation_passed': detailed_results['training_results'].validation_passed,
            'cv_auc_mean': detailed_results['training_results'].cv_auc_mean
        }
    )
    print(f"   ✓ Model saved to: {os.path.basename(model_path)}")
    
    # Step 6: Prediction pipeline
    print("\n6. Testing prediction pipeline...")
    
    # Create test data (subset of original data)
    test_data = labeled_data[feature_columns + ['itemid']].head(20).copy()
    
    # Load model and make predictions
    pipeline = PredictionPipeline()
    success = pipeline.load_model_for_prediction("demo_product_ranking_model")
    
    if success:
        print(f"   ✓ Model loaded successfully")
        
        # Batch prediction
        batch_results = pipeline.implement_batch_prediction(test_data)
        
        print(f"   ✓ Batch prediction completed:")
        print(f"     - Total items: {batch_results.total_items}")
        print(f"     - Successful predictions: {batch_results.successful_predictions}")
        print(f"     - Average score: {batch_results.average_score:.4f}")
        print(f"     - Processing time: {batch_results.processing_time:.3f}s")
        
        # Show sample predictions
        print(f"\n   Sample predictions:")
        for i, pred in enumerate(batch_results.predictions[:5]):
            print(f"     Item {pred.itemid}: Score {pred.relevance_score:.4f} (confidence: {pred.confidence:.3f})")
        
        # Create ranking
        ranking_df = pipeline.create_ranking_from_predictions(batch_results.predictions)
        print(f"\n   Top 5 ranked items:")
        for _, row in ranking_df.head(5).iterrows():
            print(f"     Rank {row['rank']}: Item {row['itemid']} (score: {row['relevance_score']:.4f})")
    
    else:
        print(f"   ✗ Failed to load model for prediction")
    
    # Step 7: List saved models
    print("\n7. Available saved models:")
    saved_models = list_saved_models()
    
    if saved_models:
        for model in saved_models[:3]:  # Show first 3
            print(f"   ✓ {model['model_name']} ({model['model_type']}) - {model['save_timestamp']}")
    else:
        print("   ⚠ No saved models found")
    
    print("\n" + "=" * 60)
    print("ML Pipeline Demo Completed Successfully!")
    print("=" * 60)


def create_synthetic_behavioral_data(n_items: int = 100) -> pd.DataFrame:
    """Create synthetic behavioral data for demo purposes"""
    
    np.random.seed(42)
    
    # Generate synthetic data
    data = []
    for i in range(n_items):
        # Generate correlated behavioral metrics
        base_popularity = np.random.exponential(50)
        view_count = int(base_popularity * np.random.uniform(0.5, 2.0))
        
        # Addtocart rate varies by item quality
        item_quality = np.random.beta(2, 5)  # Skewed towards lower values
        addtocart_count = int(view_count * item_quality * np.random.uniform(0.05, 0.3))
        
        # Transaction rate depends on addtocart
        conversion_quality = np.random.beta(2, 8)
        transaction_count = int(addtocart_count * conversion_quality * np.random.uniform(0.1, 0.8))
        
        # Unique visitors (slightly less than views)
        unique_visitors = int(view_count * np.random.uniform(0.7, 0.95))
        
        # Calculate rates
        addtocart_rate = addtocart_count / view_count if view_count > 0 else 0
        conversion_rate = transaction_count / view_count if view_count > 0 else 0
        cart_conversion_rate = transaction_count / addtocart_count if addtocart_count > 0 else 0
        
        data.append({
            'itemid': i + 1,
            'view_count': view_count,
            'addtocart_count': addtocart_count,
            'transaction_count': transaction_count,
            'unique_visitors': unique_visitors,
            'addtocart_rate': addtocart_rate,
            'conversion_rate': conversion_rate,
            'cart_conversion_rate': cart_conversion_rate
        })
    
    return pd.DataFrame(data)


if __name__ == "__main__":
    try:
        demonstrate_ml_pipeline()
    except Exception as e:
        print(f"\nError during demo: {str(e)}")
        import traceback
        traceback.print_exc()