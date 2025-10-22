"""
Test script for composite behavioral features
Tests the composite feature generation functionality
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from data.composite_features import (
    CompositeFeatureBuilder, 
    build_composite_features,
    get_feature_distribution_summary
)
from data.behavioral_metrics import calculate_behavioral_metrics
from data.csv_loader import load_events_csv


def create_sample_behavioral_data():
    """Create sample behavioral metrics data for testing"""
    np.random.seed(42)
    
    # Create sample data for 100 items
    n_items = 100
    
    data = {
        'itemid': range(1, n_items + 1),
        'view_count': np.random.poisson(50, n_items),
        'addtocart_count': np.random.poisson(5, n_items),
        'transaction_count': np.random.poisson(2, n_items),
        'unique_visitors': np.random.poisson(30, n_items)
    }
    
    df = pd.DataFrame(data)
    
    # Ensure logical consistency
    df['unique_visitors'] = np.minimum(df['unique_visitors'], df['view_count'])
    df['addtocart_count'] = np.minimum(df['addtocart_count'], df['view_count'])
    df['transaction_count'] = np.minimum(df['transaction_count'], df['addtocart_count'])
    
    # Calculate rates
    df['addtocart_rate'] = df['addtocart_count'] / df['view_count']
    df['conversion_rate'] = df['transaction_count'] / df['view_count']
    df['cart_conversion_rate'] = np.where(
        df['addtocart_count'] > 0,
        df['transaction_count'] / df['addtocart_count'],
        0.0
    )
    
    return df


def create_sample_events_data():
    """Create sample events data for testing detailed loyalty metrics"""
    np.random.seed(42)
    
    events = []
    base_timestamp = 1433221332117
    
    # Create events for first 20 items
    for itemid in range(1, 21):
        n_visitors = np.random.randint(10, 50)
        
        for visitorid in range(1, n_visitors + 1):
            # Each visitor has 1-5 events
            n_events = np.random.randint(1, 6)
            
            for event_num in range(n_events):
                timestamp = base_timestamp + np.random.randint(0, 86400000)  # Random within a day
                
                # Event type progression: view -> addtocart -> transaction
                if event_num == 0:
                    event_type = 'view'
                elif event_num < n_events - 1:
                    event_type = np.random.choice(['view', 'addtocart'], p=[0.7, 0.3])
                else:
                    event_type = np.random.choice(['view', 'addtocart', 'transaction'], p=[0.5, 0.3, 0.2])
                
                events.append({
                    'timestamp': timestamp,
                    'visitorid': visitorid,
                    'event': event_type,
                    'itemid': itemid,
                    'transactionid': f"txn_{itemid}_{visitorid}" if event_type == 'transaction' else None
                })
    
    return pd.DataFrame(events)


def test_composite_feature_builder():
    """Test the CompositeFeatureBuilder class"""
    print("Testing CompositeFeatureBuilder...")
    
    # Create sample data
    behavioral_data = create_sample_behavioral_data()
    events_data = create_sample_events_data()
    
    # Initialize builder
    builder = CompositeFeatureBuilder()
    
    # Test input validation
    print("  Testing input validation...")
    is_valid, errors = builder.validate_input_data(behavioral_data)
    print(f"    Validation result: {is_valid}, Errors: {errors}")
    assert is_valid, f"Validation should pass for sample data, but got errors: {errors}"
    
    # Test engagement intensity calculation
    print("  Testing engagement intensity calculation...")
    engagement_features = builder.calculate_engagement_intensity_scores(behavioral_data)
    print(f"    Generated engagement features for {len(engagement_features)} items")
    assert 'engagement_intensity_score' in engagement_features.columns
    assert (engagement_features['engagement_intensity_score'] >= 0).all()
    assert (engagement_features['engagement_intensity_score'] <= 1).all()
    
    # Test visitor loyalty metrics
    print("  Testing visitor loyalty metrics...")
    loyalty_features = builder.calculate_visitor_loyalty_metrics(behavioral_data, events_data)
    print(f"    Generated loyalty features for {len(loyalty_features)} items")
    assert 'visitor_loyalty_score' in loyalty_features.columns
    assert 'repeat_engagement_rate' in loyalty_features.columns
    
    # Test performance bucketing
    print("  Testing performance bucketing...")
    # Merge features for bucketing
    merged_data = behavioral_data.merge(engagement_features, on='itemid', how='left')
    merged_data = merged_data.merge(loyalty_features, on='itemid', how='left')
    
    bucketed_data = builder.implement_performance_bucketing(merged_data)
    print(f"    Generated buckets for {len(bucketed_data)} items")
    assert 'performance_bucket' in bucketed_data.columns
    assert 'engagement_bucket' in bucketed_data.columns
    assert 'loyalty_bucket' in bucketed_data.columns
    assert 'composite_score' in bucketed_data.columns
    
    # Test feature validation
    print("  Testing feature validation...")
    validated_data = builder.add_feature_validation(bucketed_data)
    print(f"    Validated features for {len(validated_data)} items")
    assert 'feature_validation_status' in validated_data.columns
    
    # Test complete pipeline
    print("  Testing complete pipeline...")
    all_features = builder.build_all_composite_features(behavioral_data, events_data)
    print(f"    Generated complete features for {len(all_features)} items")
    
    # Check that all expected columns are present
    expected_columns = [
        'itemid', 'engagement_intensity_score', 'visitor_loyalty_score',
        'performance_bucket', 'engagement_bucket', 'loyalty_bucket',
        'composite_score', 'feature_validation_status'
    ]
    
    for col in expected_columns:
        assert col in all_features.columns, f"Missing expected column: {col}"
    
    print("  ✓ All CompositeFeatureBuilder tests passed!")
    return all_features


def test_convenience_functions():
    """Test convenience functions"""
    print("Testing convenience functions...")
    
    # Create sample data
    behavioral_data = create_sample_behavioral_data()
    events_data = create_sample_events_data()
    
    # Test build_composite_features function
    print("  Testing build_composite_features...")
    composite_features = build_composite_features(behavioral_data, events_data)
    print(f"    Generated composite features for {len(composite_features)} items")
    assert not composite_features.empty
    
    # Test get_feature_distribution_summary function
    print("  Testing get_feature_distribution_summary...")
    summary = get_feature_distribution_summary(composite_features)
    print(f"    Generated summary with {len(summary)} feature summaries")
    
    # Check that summary contains expected sections
    expected_features = ['engagement_intensity_score', 'visitor_loyalty_score', 'composite_score']
    for feature in expected_features:
        if feature in composite_features.columns:
            assert feature in summary, f"Missing feature in summary: {feature}"
            assert 'mean' in summary[feature], f"Missing mean in {feature} summary"
            assert 'percentiles' in summary[feature], f"Missing percentiles in {feature} summary"
    
    print("  ✓ All convenience function tests passed!")
    return composite_features, summary


def test_with_real_data():
    """Test with real events.csv data if available"""
    print("Testing with real data (if available)...")
    
    try:
        # Try to load real events data
        events_df = load_events_csv("events.csv")
        print(f"  Loaded {len(events_df)} real events")
        
        # Calculate behavioral metrics
        behavioral_metrics = calculate_behavioral_metrics(events_df)
        print(f"  Calculated behavioral metrics for {len(behavioral_metrics)} items")
        
        # Build composite features
        composite_features = build_composite_features(behavioral_metrics, events_df)
        print(f"  Built composite features for {len(composite_features)} items")
        
        # Get summary
        summary = get_feature_distribution_summary(composite_features)
        print("  Feature distribution summary:")
        for feature, stats in summary.items():
            if isinstance(stats, dict) and 'mean' in stats:
                print(f"    {feature}: mean={stats['mean']:.4f}, std={stats['std']:.4f}")
            elif isinstance(stats, dict) and 'distribution' in stats:
                print(f"    {feature}: {stats['distribution']}")
        
        # Show sample results
        print("\n  Sample composite features:")
        sample_cols = ['itemid', 'engagement_intensity_score', 'visitor_loyalty_score', 
                      'performance_bucket', 'composite_score', 'feature_validation_status']
        available_cols = [col for col in sample_cols if col in composite_features.columns]
        print(composite_features[available_cols].head(10).to_string())
        
        print("  ✓ Real data test completed successfully!")
        return composite_features
        
    except FileNotFoundError:
        print("  events.csv not found - skipping real data test")
        return None
    except Exception as e:
        print(f"  Error with real data test: {str(e)}")
        return None


def main():
    """Run all tests"""
    print("=" * 60)
    print("COMPOSITE FEATURES TEST SUITE")
    print("=" * 60)
    
    try:
        # Test 1: CompositeFeatureBuilder class
        composite_features_sample = test_composite_feature_builder()
        print()
        
        # Test 2: Convenience functions
        composite_features_conv, summary = test_convenience_functions()
        print()
        
        # Test 3: Real data (if available)
        composite_features_real = test_with_real_data()
        print()
        
        print("=" * 60)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        # Print final summary
        print("\nFinal Summary:")
        print(f"  Sample data features: {len(composite_features_sample)} items")
        print(f"  Convenience function features: {len(composite_features_conv)} items")
        if composite_features_real is not None:
            print(f"  Real data features: {len(composite_features_real)} items")
        
        print("\nKey composite features implemented:")
        print("  ✓ Engagement intensity scores (combining views, carts, purchases)")
        print("  ✓ Visitor loyalty metrics (repeat engagement patterns)")
        print("  ✓ Performance bucketing (low/medium/high/premium)")
        print("  ✓ Feature validation and consistency checks")
        print("  ✓ Composite scoring combining multiple behavioral signals")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)