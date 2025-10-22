"""
Composite Features Example for Smart Product Re-Ranking System
Demonstrates how to use the composite behavioral features module
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from data.composite_features import (
    CompositeFeatureBuilder, 
    build_composite_features,
    get_feature_distribution_summary,
    PerformanceBucket,
    EngagementBucket,
    LoyaltyBucket
)
from data.behavioral_metrics import calculate_behavioral_metrics
from data.csv_loader import load_events_csv


def demonstrate_composite_features():
    """Demonstrate composite feature generation with sample data"""
    print("=" * 60)
    print("COMPOSITE BEHAVIORAL FEATURES DEMONSTRATION")
    print("=" * 60)
    
    # Try to load real data first, fall back to sample data
    try:
        print("Loading events data...")
        events_df = load_events_csv("events.csv")
        print(f"✓ Loaded {len(events_df)} events from events.csv")
        
        # Calculate behavioral metrics
        print("\nCalculating behavioral metrics...")
        behavioral_metrics = calculate_behavioral_metrics(events_df)
        print(f"✓ Calculated behavioral metrics for {len(behavioral_metrics)} items")
        
        use_real_data = True
        
    except FileNotFoundError:
        print("events.csv not found, using sample data...")
        events_df, behavioral_metrics = create_sample_data()
        use_real_data = False
    
    # Display basic behavioral metrics summary
    print("\n" + "=" * 40)
    print("BASIC BEHAVIORAL METRICS SUMMARY")
    print("=" * 40)
    
    print(f"Total items: {len(behavioral_metrics)}")
    print(f"Total views: {behavioral_metrics['view_count'].sum():,}")
    print(f"Total add-to-carts: {behavioral_metrics['addtocart_count'].sum():,}")
    print(f"Total transactions: {behavioral_metrics['transaction_count'].sum():,}")
    print(f"Average conversion rate: {behavioral_metrics['conversion_rate'].mean():.4f}")
    print(f"Average add-to-cart rate: {behavioral_metrics['addtocart_rate'].mean():.4f}")
    
    # Build composite features
    print("\n" + "=" * 40)
    print("BUILDING COMPOSITE FEATURES")
    print("=" * 40)
    
    print("Building composite behavioral features...")
    composite_features = build_composite_features(
        behavioral_metrics, 
        events_df if use_real_data else None
    )
    print(f"✓ Built composite features for {len(composite_features)} items")
    
    # Display composite features summary
    print("\n" + "=" * 40)
    print("COMPOSITE FEATURES SUMMARY")
    print("=" * 40)
    
    summary = get_feature_distribution_summary(composite_features)
    
    # Display numeric feature statistics
    numeric_features = ['engagement_intensity_score', 'visitor_loyalty_score', 'composite_score']
    for feature in numeric_features:
        if feature in summary:
            stats = summary[feature]
            print(f"\n{feature.upper()}:")
            print(f"  Mean: {stats['mean']:.4f}")
            print(f"  Median: {stats['median']:.4f}")
            print(f"  Std Dev: {stats['std']:.4f}")
            print(f"  Range: {stats['min']:.4f} - {stats['max']:.4f}")
            print(f"  25th-75th percentile: {stats['percentiles']['25th']:.4f} - {stats['percentiles']['75th']:.4f}")
    
    # Display categorical feature distributions
    categorical_features = ['performance_bucket', 'engagement_bucket', 'loyalty_bucket']
    for feature in categorical_features:
        if feature in summary:
            dist = summary[feature]['distribution']
            print(f"\n{feature.upper()} DISTRIBUTION:")
            for bucket, count in dist.items():
                percentage = (count / len(composite_features)) * 100
                print(f"  {bucket}: {count} items ({percentage:.1f}%)")
    
    # Feature validation summary
    if 'feature_validation_status' in summary:
        validation_dist = summary['feature_validation_status']['distribution']
        print(f"\nFEATURE VALIDATION STATUS:")
        for status, count in validation_dist.items():
            percentage = (count / len(composite_features)) * 100
            print(f"  {status}: {count} items ({percentage:.1f}%)")
    
    # Show top performers in different categories
    print("\n" + "=" * 40)
    print("TOP PERFORMERS BY CATEGORY")
    print("=" * 40)
    
    # Top by engagement intensity
    top_engagement = composite_features.nlargest(5, 'engagement_intensity_score')
    print("\nTOP 5 BY ENGAGEMENT INTENSITY:")
    display_cols = ['itemid', 'engagement_intensity_score', 'view_count', 'addtocart_count', 'transaction_count']
    available_cols = [col for col in display_cols if col in top_engagement.columns]
    print(top_engagement[available_cols].to_string(index=False))
    
    # Top by visitor loyalty
    if 'visitor_loyalty_score' in composite_features.columns:
        top_loyalty = composite_features.nlargest(5, 'visitor_loyalty_score')
        print("\nTOP 5 BY VISITOR LOYALTY:")
        display_cols = ['itemid', 'visitor_loyalty_score', 'unique_visitors', 'conversion_rate', 'loyalty_bucket']
        available_cols = [col for col in display_cols if col in top_loyalty.columns]
        print(top_loyalty[available_cols].to_string(index=False))
    
    # Top by composite score
    if 'composite_score' in composite_features.columns:
        top_composite = composite_features.nlargest(5, 'composite_score')
        print("\nTOP 5 BY COMPOSITE SCORE:")
        display_cols = ['itemid', 'composite_score', 'performance_bucket', 'engagement_bucket', 'loyalty_bucket']
        available_cols = [col for col in display_cols if col in top_composite.columns]
        print(top_composite[available_cols].to_string(index=False))
    
    # Demonstrate custom feature builder configuration
    print("\n" + "=" * 40)
    print("CUSTOM CONFIGURATION EXAMPLE")
    print("=" * 40)
    
    # Create custom engagement weights (emphasize transactions more)
    custom_weights = {
        'view': 1.0,
        'addtocart': 5.0,  # Increased from default 3.0
        'transaction': 15.0  # Increased from default 10.0
    }
    
    print("Building features with custom engagement weights...")
    print(f"Custom weights: {custom_weights}")
    
    custom_features = build_composite_features(
        behavioral_metrics, 
        events_df if use_real_data else None,
        engagement_weights=custom_weights
    )
    
    # Compare engagement scores
    comparison = pd.DataFrame({
        'itemid': composite_features['itemid'],
        'default_engagement': composite_features['engagement_intensity_score'],
        'custom_engagement': custom_features['engagement_intensity_score'],
    })
    comparison['difference'] = comparison['custom_engagement'] - comparison['default_engagement']
    
    print("\nCOMPARISON OF ENGAGEMENT SCORES (Top 10 differences):")
    top_differences = comparison.nlargest(10, 'difference')
    print(top_differences.to_string(index=False))
    
    # Business insights
    print("\n" + "=" * 40)
    print("BUSINESS INSIGHTS")
    print("=" * 40)
    
    # Performance bucket insights
    if 'performance_bucket' in composite_features.columns:
        premium_items = composite_features[composite_features['performance_bucket'] == PerformanceBucket.PREMIUM.value]
        low_items = composite_features[composite_features['performance_bucket'] == PerformanceBucket.LOW.value]
        
        print(f"\nPERFORMANCE INSIGHTS:")
        print(f"  Premium items ({len(premium_items)}): Average conversion rate = {premium_items['conversion_rate'].mean():.4f}")
        print(f"  Low items ({len(low_items)}): Average conversion rate = {low_items['conversion_rate'].mean():.4f}")
        
        if len(premium_items) > 0 and len(low_items) > 0:
            improvement_factor = premium_items['conversion_rate'].mean() / low_items['conversion_rate'].mean()
            print(f"  Premium items convert {improvement_factor:.1f}x better than low performers")
    
    # Engagement insights
    if 'engagement_bucket' in composite_features.columns:
        exceptional_items = composite_features[composite_features['engagement_bucket'] == EngagementBucket.EXCEPTIONAL.value]
        minimal_items = composite_features[composite_features['engagement_bucket'] == EngagementBucket.MINIMAL.value]
        
        print(f"\nENGAGEMENT INSIGHTS:")
        if len(exceptional_items) > 0:
            print(f"  Exceptional engagement items ({len(exceptional_items)}): Avg events per visitor = {exceptional_items['events_per_visitor'].mean():.2f}")
        if len(minimal_items) > 0:
            print(f"  Minimal engagement items ({len(minimal_items)}): Avg events per visitor = {minimal_items['events_per_visitor'].mean():.2f}")
    
    # Loyalty insights
    if 'loyalty_bucket' in composite_features.columns:
        devoted_items = composite_features[composite_features['loyalty_bucket'] == LoyaltyBucket.DEVOTED.value]
        transient_items = composite_features[composite_features['loyalty_bucket'] == LoyaltyBucket.TRANSIENT.value]
        
        print(f"\nLOYALTY INSIGHTS:")
        if len(devoted_items) > 0:
            print(f"  Devoted loyalty items ({len(devoted_items)}): Avg repeat engagement = {devoted_items['repeat_engagement_rate'].mean():.4f}")
        if len(transient_items) > 0:
            print(f"  Transient loyalty items ({len(transient_items)}): Avg repeat engagement = {transient_items['repeat_engagement_rate'].mean():.4f}")
    
    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    return composite_features


def create_sample_data():
    """Create sample data for demonstration when real data is not available"""
    print("Creating sample data for demonstration...")
    
    # Create sample events
    np.random.seed(42)
    events = []
    base_timestamp = 1433221332117
    
    for itemid in range(1, 101):  # 100 items
        n_visitors = np.random.randint(20, 100)
        
        for visitorid in range(1, n_visitors + 1):
            n_events = np.random.randint(1, 8)
            
            for event_num in range(n_events):
                timestamp = base_timestamp + np.random.randint(0, 86400000 * 7)  # Random within a week
                
                # Realistic event progression
                if event_num == 0:
                    event_type = 'view'
                elif event_num < n_events - 1:
                    event_type = np.random.choice(['view', 'addtocart'], p=[0.8, 0.2])
                else:
                    event_type = np.random.choice(['view', 'addtocart', 'transaction'], p=[0.6, 0.25, 0.15])
                
                events.append({
                    'timestamp': timestamp,
                    'visitorid': visitorid + (itemid - 1) * 1000,  # Unique visitor IDs
                    'event': event_type,
                    'itemid': itemid,
                    'transactionid': f"txn_{itemid}_{visitorid}" if event_type == 'transaction' else None
                })
    
    events_df = pd.DataFrame(events)
    
    # Calculate behavioral metrics
    behavioral_metrics = calculate_behavioral_metrics(events_df)
    
    print(f"✓ Created sample data: {len(events_df)} events, {len(behavioral_metrics)} items")
    
    return events_df, behavioral_metrics


def demonstrate_advanced_features():
    """Demonstrate advanced composite feature capabilities"""
    print("\n" + "=" * 40)
    print("ADVANCED FEATURES DEMONSTRATION")
    print("=" * 40)
    
    # Create sample data with specific patterns
    behavioral_data = create_advanced_sample_data()
    
    # Initialize builder with custom configuration
    custom_thresholds = {
        'conversion_rate': [0.20, 0.40, 0.70],  # More aggressive thresholds
        'engagement_intensity': [0.25, 0.50, 0.85],
        'visitor_loyalty': [0.15, 0.45, 0.75]
    }
    
    builder = CompositeFeatureBuilder(performance_thresholds=custom_thresholds)
    
    # Build features
    composite_features = builder.build_all_composite_features(behavioral_data)
    
    # Analyze feature validation results
    validation_counts = composite_features['feature_validation_status'].value_counts()
    print(f"\nFEATURE VALIDATION RESULTS:")
    for status, count in validation_counts.items():
        print(f"  {status}: {count} items")
    
    # Show items with validation issues
    problematic_items = composite_features[
        composite_features['feature_validation_status'] != 'valid'
    ]
    
    if len(problematic_items) > 0:
        print(f"\nITEMS WITH VALIDATION ISSUES ({len(problematic_items)}):")
        issue_cols = ['itemid', 'feature_validation_status', 'view_count', 'unique_visitors', 'conversion_rate']
        available_cols = [col for col in issue_cols if col in problematic_items.columns]
        print(problematic_items[available_cols].head(10).to_string(index=False))
    
    # Demonstrate bucket distribution with custom thresholds
    print(f"\nBUCKET DISTRIBUTIONS WITH CUSTOM THRESHOLDS:")
    for bucket_col in ['performance_bucket', 'engagement_bucket', 'loyalty_bucket']:
        if bucket_col in composite_features.columns:
            dist = composite_features[bucket_col].value_counts()
            print(f"\n{bucket_col}:")
            for bucket, count in dist.items():
                percentage = (count / len(composite_features)) * 100
                print(f"  {bucket}: {count} ({percentage:.1f}%)")
    
    return composite_features


def create_advanced_sample_data():
    """Create sample data with specific patterns for advanced demonstration"""
    np.random.seed(123)
    
    # Create 50 items with varied behavioral patterns
    n_items = 50
    
    data = []
    
    for i in range(n_items):
        itemid = i + 1
        
        # Create different item archetypes
        if i < 10:  # High performers
            view_count = np.random.randint(100, 500)
            unique_visitors = np.random.randint(80, min(400, view_count))
            addtocart_count = np.random.randint(20, min(100, view_count))
            transaction_count = np.random.randint(10, min(50, addtocart_count))
        elif i < 20:  # Medium performers
            view_count = np.random.randint(50, 150)
            unique_visitors = np.random.randint(30, min(120, view_count))
            addtocart_count = np.random.randint(5, min(30, view_count))
            transaction_count = np.random.randint(2, min(15, addtocart_count))
        elif i < 35:  # Low performers
            view_count = np.random.randint(10, 60)
            unique_visitors = np.random.randint(8, min(50, view_count))
            addtocart_count = np.random.randint(0, min(10, view_count))
            transaction_count = np.random.randint(0, min(5, addtocart_count))
        else:  # Edge cases with potential validation issues
            view_count = np.random.randint(1, 20)
            unique_visitors = np.random.randint(1, min(25, view_count + 5))  # Might exceed views
            addtocart_count = np.random.randint(0, min(30, view_count + 10))  # Might exceed views
            transaction_count = np.random.randint(0, min(10, addtocart_count + 5))
        
        # Calculate rates
        addtocart_rate = addtocart_count / view_count if view_count > 0 else 0
        conversion_rate = transaction_count / view_count if view_count > 0 else 0
        cart_conversion_rate = transaction_count / addtocart_count if addtocart_count > 0 else 0
        
        data.append({
            'itemid': itemid,
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
        # Main demonstration
        composite_features = demonstrate_composite_features()
        
        # Advanced features demonstration
        advanced_features = demonstrate_advanced_features()
        
        print(f"\n✓ Successfully demonstrated composite features for {len(composite_features)} items")
        print("✓ All composite feature capabilities have been showcased")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {str(e)}")
        import traceback
        traceback.print_exc()