"""
Example script demonstrating temporal and engagement feature extraction
for the Smart Product Re-Ranking System
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from data.csv_loader import load_events_csv
from data.feature_engineering import (
    FeatureEngineeringPipeline,
    TemporalFeatureExtractor,
    EngagementFeatureExtractor,
    extract_temporal_and_engagement_features,
    get_top_items_by_feature
)
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def demonstrate_temporal_features():
    """Demonstrate temporal feature extraction"""
    print("\n" + "="*60)
    print("TEMPORAL FEATURE EXTRACTION DEMONSTRATION")
    print("="*60)
    
    try:
        # Load events data
        print("Loading events data...")
        events_df = load_events_csv("events.csv")
        print(f"✓ Loaded {len(events_df)} events")
        print(f"✓ Date range: {pd.to_datetime(events_df['timestamp'], unit='ms').min()} to {pd.to_datetime(events_df['timestamp'], unit='ms').max()}")
        print(f"✓ Unique items: {events_df['itemid'].nunique()}")
        print(f"✓ Unique visitors: {events_df['visitorid'].nunique()}")
        
        # Initialize temporal extractor
        temporal_extractor = TemporalFeatureExtractor()
        
        # Extract time-based patterns
        print("\nExtracting time-based patterns...")
        temporal_patterns = temporal_extractor.extract_time_based_patterns(events_df)
        print(f"✓ Extracted temporal patterns for {len(temporal_patterns)} items")
        
        # Show sample temporal patterns
        print("\nSample temporal patterns:")
        sample_patterns = temporal_patterns.head(5)
        for _, row in sample_patterns.iterrows():
            print(f"  Item {row['itemid']}: Peak hour={row['peak_activity_hour']}, "
                  f"Activity span={row['activity_span_hours']:.1f}h, "
                  f"Avg daily events={row['avg_daily_events']:.1f}")
        
        # Extract time-to-action metrics
        print("\nCalculating time-to-action metrics...")
        time_metrics = temporal_extractor.calculate_time_to_action_metrics(events_df)
        print(f"✓ Calculated time-to-action metrics for {len(time_metrics)} items")
        
        # Show sample time-to-action metrics
        print("\nSample time-to-action metrics:")
        sample_metrics = time_metrics.head(5)
        for _, row in sample_metrics.iterrows():
            cart_time = f"{row['avg_time_to_cart']:.0f}s" if pd.notna(row['avg_time_to_cart']) else "N/A"
            purchase_time = f"{row['avg_time_to_purchase']:.0f}s" if pd.notna(row['avg_time_to_purchase']) else "N/A"
            print(f"  Item {row['itemid']}: Avg time to cart={cart_time}, "
                  f"Avg time to purchase={purchase_time}, "
                  f"Avg session depth={row['avg_session_depth']:.1f}")
        
        return temporal_patterns, time_metrics
        
    except Exception as e:
        print(f"✗ Error in temporal feature extraction: {str(e)}")
        return None, None


def demonstrate_engagement_features():
    """Demonstrate engagement feature extraction"""
    print("\n" + "="*60)
    print("ENGAGEMENT FEATURE EXTRACTION DEMONSTRATION")
    print("="*60)
    
    try:
        # Load events data
        print("Loading events data...")
        events_df = load_events_csv("events.csv")
        print(f"✓ Loaded {len(events_df)} events")
        
        # Initialize engagement extractor
        engagement_extractor = EngagementFeatureExtractor()
        
        # Calculate unique visitor counts
        print("\nCalculating unique visitor counts...")
        visitor_counts = engagement_extractor.calculate_unique_visitor_counts(events_df)
        print(f"✓ Calculated visitor counts for {len(visitor_counts)} items")
        
        # Show top items by unique visitors
        print("\nTop 5 items by unique visitors:")
        top_visitors = visitor_counts.nlargest(5, 'unique_visitors')
        for _, row in top_visitors.iterrows():
            print(f"  Item {row['itemid']}: {row['unique_visitors']} unique visitors, "
                  f"{row['total_events']} total events, "
                  f"{row['events_per_visitor']:.1f} events/visitor")
        
        # Calculate popularity scoring
        print("\nCalculating popularity scores...")
        popularity_scores = engagement_extractor.calculate_popularity_scoring(events_df)
        print(f"✓ Calculated popularity scores for {len(popularity_scores)} items")
        
        # Show top items by popularity
        print("\nTop 5 items by popularity score:")
        top_popular = popularity_scores.nlargest(5, 'popularity_score')
        for _, row in top_popular.iterrows():
            print(f"  Item {row['itemid']}: Popularity={row['popularity_score']:.3f}, "
                  f"Engagement={row['weighted_engagement_score']:.0f}, "
                  f"Intensity={row['engagement_intensity']:.1f}")
        
        # Calculate visitor loyalty metrics
        print("\nCalculating visitor loyalty metrics...")
        loyalty_metrics = engagement_extractor.calculate_visitor_loyalty_metrics(events_df)
        print(f"✓ Calculated loyalty metrics for {len(loyalty_metrics)} items")
        
        # Show top items by loyalty
        print("\nTop 5 items by repeat visitor rate:")
        top_loyalty = loyalty_metrics.nlargest(5, 'repeat_visitor_rate')
        for _, row in top_loyalty.iterrows():
            print(f"  Item {row['itemid']}: Repeat rate={row['repeat_visitor_rate']:.3f}, "
                  f"Avg events/visitor={row['avg_events_per_visitor']:.1f}, "
                  f"Avg duration={row['avg_engagement_duration_seconds']:.0f}s")
        
        return visitor_counts, popularity_scores, loyalty_metrics
        
    except Exception as e:
        print(f"✗ Error in engagement feature extraction: {str(e)}")
        return None, None, None


def demonstrate_complete_pipeline():
    """Demonstrate the complete feature engineering pipeline"""
    print("\n" + "="*60)
    print("COMPLETE FEATURE ENGINEERING PIPELINE DEMONSTRATION")
    print("="*60)
    
    try:
        # Load events data
        print("Loading events data...")
        events_df = load_events_csv("events.csv")
        print(f"✓ Loaded {len(events_df)} events")
        
        # Extract all features using the pipeline
        print("\nExtracting all temporal and engagement features...")
        features = extract_temporal_and_engagement_features(events_df)
        print(f"✓ Extracted comprehensive features for {len(features)} items")
        print(f"✓ Total feature columns: {len(features.columns) - 1}")  # Exclude itemid
        
        # Show feature columns
        feature_cols = [col for col in features.columns if col != 'itemid']
        print(f"\nExtracted features:")
        for i, col in enumerate(feature_cols, 1):
            print(f"  {i:2d}. {col}")
        
        # Get feature summary
        pipeline = FeatureEngineeringPipeline()
        summary = pipeline.get_feature_summary(features)
        
        print(f"\nFeature Summary:")
        print(f"  Total items with features: {summary['total_items']}")
        print(f"  Total feature count: {summary['feature_count']}")
        
        # Show temporal feature statistics
        if 'temporal_features' in summary:
            print(f"\n  Temporal Features Statistics:")
            for feature, stats in summary['temporal_features'].items():
                if stats['mean'] is not None:
                    print(f"    {feature}: mean={stats['mean']:.2f}, median={stats['median']:.2f}")
        
        # Show engagement feature statistics
        if 'engagement_features' in summary:
            print(f"\n  Engagement Features Statistics:")
            for feature, stats in summary['engagement_features'].items():
                if stats['mean'] is not None:
                    print(f"    {feature}: mean={stats['mean']:.2f}, median={stats['median']:.2f}")
        
        # Demonstrate top items by different features
        print(f"\nTop 3 items by different features:")
        
        key_features = ['popularity_score', 'unique_visitors', 'engagement_intensity', 
                       'repeat_visitor_rate', 'avg_session_depth']
        
        for feature in key_features:
            if feature in features.columns:
                top_items = get_top_items_by_feature(features, feature, 3)
                print(f"\n  Top 3 by {feature}:")
                for _, row in top_items.iterrows():
                    print(f"    Item {row['itemid']}: {feature}={row[feature]:.3f}")
        
        # Save features to CSV for further analysis
        output_file = "extracted_features.csv"
        features.to_csv(output_file, index=False)
        print(f"\n✓ Saved extracted features to {output_file}")
        
        return features
        
    except Exception as e:
        print(f"✗ Error in complete pipeline demonstration: {str(e)}")
        return None


def analyze_feature_correlations(features_df):
    """Analyze correlations between features"""
    print("\n" + "="*60)
    print("FEATURE CORRELATION ANALYSIS")
    print("="*60)
    
    try:
        # Select numeric features for correlation analysis
        numeric_features = features_df.select_dtypes(include=[np.number]).drop(columns=['itemid'])
        
        if numeric_features.empty:
            print("No numeric features found for correlation analysis")
            return
        
        # Calculate correlation matrix
        correlation_matrix = numeric_features.corr()
        
        # Find highly correlated feature pairs
        print("Highly correlated feature pairs (|correlation| > 0.7):")
        high_corr_pairs = []
        
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:
                    feature1 = correlation_matrix.columns[i]
                    feature2 = correlation_matrix.columns[j]
                    high_corr_pairs.append((feature1, feature2, corr_value))
        
        if high_corr_pairs:
            for feature1, feature2, corr in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True):
                print(f"  {feature1} <-> {feature2}: {corr:.3f}")
        else:
            print("  No highly correlated feature pairs found")
        
        # Show feature statistics
        print(f"\nFeature Statistics Summary:")
        print(f"  Total numeric features: {len(numeric_features.columns)}")
        print(f"  Features with missing values: {numeric_features.isnull().any().sum()}")
        print(f"  Average correlation magnitude: {abs(correlation_matrix).mean().mean():.3f}")
        
    except Exception as e:
        print(f"✗ Error in correlation analysis: {str(e)}")


def main():
    """Main function to run all demonstrations"""
    print("Smart Product Re-Ranking System")
    print("Temporal and Engagement Feature Engineering Demonstration")
    print("="*80)
    
    # Check if events.csv exists
    if not os.path.exists("events.csv"):
        print("✗ events.csv file not found in current directory")
        print("  Please ensure the events.csv file is available")
        return
    
    # Run demonstrations
    temporal_patterns, time_metrics = demonstrate_temporal_features()
    visitor_counts, popularity_scores, loyalty_metrics = demonstrate_engagement_features()
    complete_features = demonstrate_complete_pipeline()
    
    if complete_features is not None:
        analyze_feature_correlations(complete_features)
    
    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETE")
    print("="*80)
    print("✓ All temporal and engagement features have been successfully extracted")
    print("✓ Features saved to 'extracted_features.csv' for further analysis")
    print("✓ Ready for machine learning model training")


if __name__ == "__main__":
    main()