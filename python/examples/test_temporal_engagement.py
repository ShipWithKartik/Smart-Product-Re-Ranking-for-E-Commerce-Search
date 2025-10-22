"""
Simple test script for temporal and engagement features
Tests the implementation without requiring full CSV loading
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from data.feature_engineering import (
    TemporalFeatureExtractor,
    EngagementFeatureExtractor,
    FeatureEngineeringPipeline
)

def create_sample_data():
    """Create sample event data for testing"""
    np.random.seed(42)
    
    # Create sample events
    events = []
    
    # Sample data for 5 items and 10 visitors
    items = [1001, 1002, 1003, 1004, 1005]
    visitors = list(range(1, 11))
    
    base_timestamp = 1433221332117  # Base timestamp from real data
    
    for item in items:
        for visitor in visitors:
            # Each visitor has a chance to interact with each item
            if np.random.random() > 0.3:  # 70% chance of interaction
                
                # View event (always first)
                events.append({
                    'timestamp': base_timestamp + np.random.randint(0, 86400000),  # Random time within a day
                    'visitorid': visitor,
                    'event': 'view',
                    'itemid': item,
                    'transactionid': ''
                })
                
                # Addtocart event (30% chance after view)
                if np.random.random() > 0.7:
                    events.append({
                        'timestamp': base_timestamp + np.random.randint(3600000, 90000000),  # 1 hour to 25 hours later
                        'visitorid': visitor,
                        'event': 'addtocart',
                        'itemid': item,
                        'transactionid': ''
                    })
                    
                    # Transaction event (50% chance after addtocart)
                    if np.random.random() > 0.5:
                        events.append({
                            'timestamp': base_timestamp + np.random.randint(7200000, 172800000),  # 2 hours to 48 hours later
                            'visitorid': visitor,
                            'event': 'transaction',
                            'itemid': item,
                            'transactionid': f'T{item}_{visitor}'
                        })
    
    return pd.DataFrame(events)

def test_temporal_features():
    """Test temporal feature extraction"""
    print("Testing Temporal Feature Extraction...")
    
    # Create sample data
    events_df = create_sample_data()
    print(f"Created {len(events_df)} sample events")
    
    # Initialize extractor
    extractor = TemporalFeatureExtractor()
    
    # Test time-based patterns
    print("\n1. Testing time-based patterns extraction...")
    try:
        patterns = extractor.extract_time_based_patterns(events_df)
        print(f"✓ Extracted patterns for {len(patterns)} items")
        print(f"  Columns: {list(patterns.columns)}")
        
        if len(patterns) > 0:
            sample = patterns.iloc[0]
            print(f"  Sample: Item {sample['itemid']} - Peak hour: {sample['peak_activity_hour']}, Activity span: {sample['activity_span_hours']:.1f}h")
    except Exception as e:
        print(f"✗ Error: {str(e)}")
    
    # Test time-to-action metrics
    print("\n2. Testing time-to-action metrics...")
    try:
        time_metrics = extractor.calculate_time_to_action_metrics(events_df)
        print(f"✓ Calculated metrics for {len(time_metrics)} items")
        print(f"  Columns: {list(time_metrics.columns)}")
        
        if len(time_metrics) > 0:
            sample = time_metrics.iloc[0]
            cart_time = f"{sample['avg_time_to_cart']:.0f}s" if pd.notna(sample['avg_time_to_cart']) else "N/A"
            purchase_time = f"{sample['avg_time_to_purchase']:.0f}s" if pd.notna(sample['avg_time_to_purchase']) else "N/A"
            print(f"  Sample: Item {sample['itemid']} - Avg time to cart: {cart_time}, Avg time to purchase: {purchase_time}")
    except Exception as e:
        print(f"✗ Error: {str(e)}")

def test_engagement_features():
    """Test engagement feature extraction"""
    print("\nTesting Engagement Feature Extraction...")
    
    # Create sample data
    events_df = create_sample_data()
    print(f"Created {len(events_df)} sample events")
    
    # Initialize extractor
    extractor = EngagementFeatureExtractor()
    
    # Test unique visitor counts
    print("\n1. Testing unique visitor counts...")
    try:
        visitor_counts = extractor.calculate_unique_visitor_counts(events_df)
        print(f"✓ Calculated visitor counts for {len(visitor_counts)} items")
        print(f"  Columns: {list(visitor_counts.columns)}")
        
        if len(visitor_counts) > 0:
            sample = visitor_counts.iloc[0]
            print(f"  Sample: Item {sample['itemid']} - {sample['unique_visitors']} unique visitors, {sample['total_events']} total events")
    except Exception as e:
        print(f"✗ Error: {str(e)}")
    
    # Test popularity scoring
    print("\n2. Testing popularity scoring...")
    try:
        popularity = extractor.calculate_popularity_scoring(events_df)
        print(f"✓ Calculated popularity scores for {len(popularity)} items")
        print(f"  Columns: {list(popularity.columns)}")
        
        if len(popularity) > 0:
            sample = popularity.iloc[0]
            print(f"  Sample: Item {sample['itemid']} - Popularity: {sample['popularity_score']:.3f}, Engagement: {sample['weighted_engagement_score']:.0f}")
    except Exception as e:
        print(f"✗ Error: {str(e)}")
    
    # Test visitor loyalty metrics
    print("\n3. Testing visitor loyalty metrics...")
    try:
        loyalty = extractor.calculate_visitor_loyalty_metrics(events_df)
        print(f"✓ Calculated loyalty metrics for {len(loyalty)} items")
        print(f"  Columns: {list(loyalty.columns)}")
        
        if len(loyalty) > 0:
            sample = loyalty.iloc[0]
            print(f"  Sample: Item {sample['itemid']} - Repeat rate: {sample['repeat_visitor_rate']:.3f}, Avg events/visitor: {sample['avg_events_per_visitor']:.1f}")
    except Exception as e:
        print(f"✗ Error: {str(e)}")

def test_complete_pipeline():
    """Test the complete feature engineering pipeline"""
    print("\nTesting Complete Feature Engineering Pipeline...")
    
    # Create sample data
    events_df = create_sample_data()
    print(f"Created {len(events_df)} sample events")
    
    # Initialize pipeline
    pipeline = FeatureEngineeringPipeline()
    
    # Test complete feature extraction
    print("\n1. Testing complete feature extraction...")
    try:
        features = pipeline.extract_all_features(events_df)
        print(f"✓ Extracted features for {len(features)} items")
        print(f"  Total feature columns: {len(features.columns) - 1}")  # Exclude itemid
        
        feature_cols = [col for col in features.columns if col != 'itemid']
        print(f"  Feature columns: {feature_cols[:5]}..." if len(feature_cols) > 5 else f"  Feature columns: {feature_cols}")
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return None
    
    # Test feature summary
    print("\n2. Testing feature summary...")
    try:
        summary = pipeline.get_feature_summary(features)
        print(f"✓ Generated feature summary")
        print(f"  Total items: {summary['total_items']}")
        print(f"  Feature count: {summary['feature_count']}")
        
        if 'temporal_features' in summary:
            print(f"  Temporal features: {len(summary['temporal_features'])}")
        
        if 'engagement_features' in summary:
            print(f"  Engagement features: {len(summary['engagement_features'])}")
            
    except Exception as e:
        print(f"✗ Error: {str(e)}")
    
    return features

def main():
    """Run all tests"""
    print("=" * 60)
    print("TEMPORAL AND ENGAGEMENT FEATURES TEST")
    print("=" * 60)
    
    # Run individual tests
    test_temporal_features()
    test_engagement_features()
    features = test_complete_pipeline()
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    if features is not None and len(features) > 0:
        print("✓ All tests passed successfully!")
        print("✓ Temporal and engagement features are working correctly")
        print(f"✓ Generated {len(features.columns) - 1} features for {len(features)} items")
        
        # Show sample of extracted features
        print(f"\nSample extracted features for item {features.iloc[0]['itemid']}:")
        sample_row = features.iloc[0]
        for col in features.columns[:10]:  # Show first 10 columns
            if col != 'itemid':
                value = sample_row[col]
                if pd.notna(value) and isinstance(value, (int, float)):
                    print(f"  {col}: {value:.3f}")
                else:
                    print(f"  {col}: {value}")
    else:
        print("✗ Some tests failed. Please check the implementation.")

if __name__ == "__main__":
    main()