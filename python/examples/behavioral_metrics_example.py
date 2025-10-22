"""
Behavioral Metrics Processing Example
Demonstrates how to use the behavioral metrics pipeline with real event data
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import json

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.behavioral_pipeline import BehavioralMetricsPipeline, run_behavioral_metrics_pipeline
from data.behavioral_metrics import BehavioralMetricsCalculator, get_top_performing_items
from config.behavioral_metrics_config import ConfigProfiles, BehavioralMetricsConfig


def load_and_analyze_events(events_file: str = "events.csv"):
    """
    Load and analyze events using the behavioral metrics pipeline
    
    Args:
        events_file: Path to events CSV file
    """
    print(f"Loading and analyzing events from: {events_file}")
    
    if not os.path.exists(events_file):
        print(f"Error: {events_file} not found. Please ensure the file exists.")
        return None
    
    try:
        # Run the complete pipeline
        results = run_behavioral_metrics_pipeline(
            events_file,
            include_temporal=True,
            save_results=True,
            output_dir="behavioral_analysis_output"
        )
        
        print("\nPipeline Results:")
        print(f"  Total events processed: {len(results['events_df']):,}")
        print(f"  Items with metrics: {len(results['metrics_df']):,}")
        
        # Display summary statistics
        summary = results['summary']
        if 'metrics' in summary:
            metrics_summary = summary['metrics']
            print(f"\nBehavioral Metrics Summary:")
            print(f"  Total views: {metrics_summary.get('total_views', 0):,}")
            print(f"  Total add-to-carts: {metrics_summary.get('total_addtocarts', 0):,}")
            print(f"  Total transactions: {metrics_summary.get('total_transactions', 0):,}")
            print(f"  Average conversion rate: {metrics_summary.get('avg_conversion_rate', 0):.3f}")
            print(f"  Items with purchases: {metrics_summary.get('items_with_purchases', 0):,}")
        
        return results
        
    except Exception as e:
        print(f"Error processing events: {str(e)}")
        return None


def analyze_top_performers(metrics_df: pd.DataFrame, top_n: int = 10):
    """
    Analyze top performing items based on different metrics
    
    Args:
        metrics_df: DataFrame with behavioral metrics
        top_n: Number of top items to show
    """
    print(f"\nTop {top_n} Performing Items Analysis:")
    print("=" * 50)
    
    # Top by conversion rate
    print(f"\nTop {top_n} by Conversion Rate:")
    top_conversion = get_top_performing_items(metrics_df, 'conversion_rate', top_n)
    if not top_conversion.empty:
        display_cols = ['itemid', 'view_count', 'transaction_count', 'conversion_rate']
        available_cols = [col for col in display_cols if col in top_conversion.columns]
        print(top_conversion[available_cols].to_string(index=False))
    
    # Top by view count
    print(f"\nTop {top_n} by View Count:")
    top_views = get_top_performing_items(metrics_df, 'view_count', top_n)
    if not top_views.empty:
        display_cols = ['itemid', 'view_count', 'addtocart_count', 'transaction_count']
        available_cols = [col for col in display_cols if col in top_views.columns]
        print(top_views[available_cols].to_string(index=False))
    
    # Top by add-to-cart rate
    print(f"\nTop {top_n} by Add-to-Cart Rate:")
    top_addtocart = get_top_performing_items(metrics_df, 'addtocart_rate', top_n)
    if not top_addtocart.empty:
        display_cols = ['itemid', 'view_count', 'addtocart_count', 'addtocart_rate']
        available_cols = [col for col in display_cols if col in top_addtocart.columns]
        print(top_addtocart[available_cols].to_string(index=False))


def analyze_conversion_patterns(metrics_df: pd.DataFrame):
    """
    Analyze conversion patterns and distributions
    
    Args:
        metrics_df: DataFrame with behavioral metrics
    """
    print("\nConversion Pattern Analysis:")
    print("=" * 40)
    
    # Conversion rate distribution
    if 'conversion_rate' in metrics_df.columns:
        conversion_stats = metrics_df['conversion_rate'].describe()
        print("\nConversion Rate Distribution:")
        print(conversion_stats.to_string())
        
        # Conversion rate buckets
        metrics_df['conversion_bucket'] = pd.cut(
            metrics_df['conversion_rate'],
            bins=[0, 0.01, 0.05, 0.1, 0.2, 1.0],
            labels=['0-1%', '1-5%', '5-10%', '10-20%', '20%+']
        )
        
        bucket_counts = metrics_df['conversion_bucket'].value_counts().sort_index()
        print(f"\nItems by Conversion Rate Bucket:")
        for bucket, count in bucket_counts.items():
            percentage = (count / len(metrics_df)) * 100
            print(f"  {bucket}: {count:,} items ({percentage:.1f}%)")
    
    # Add-to-cart rate analysis
    if 'addtocart_rate' in metrics_df.columns:
        print(f"\nAdd-to-Cart Rate Statistics:")
        addtocart_stats = metrics_df['addtocart_rate'].describe()
        print(addtocart_stats.to_string())
    
    # View count analysis
    if 'view_count' in metrics_df.columns:
        print(f"\nView Count Statistics:")
        view_stats = metrics_df['view_count'].describe()
        print(view_stats.to_string())
        
        # High-traffic items
        high_traffic_threshold = metrics_df['view_count'].quantile(0.9)
        high_traffic_items = metrics_df[metrics_df['view_count'] >= high_traffic_threshold]
        print(f"\nHigh-Traffic Items (top 10% by views): {len(high_traffic_items):,}")
        if len(high_traffic_items) > 0:
            avg_conversion = high_traffic_items['conversion_rate'].mean()
            print(f"  Average conversion rate: {avg_conversion:.3f}")


def demonstrate_custom_configuration():
    """Demonstrate using custom configuration for behavioral metrics"""
    print("\nCustom Configuration Demonstration:")
    print("=" * 45)
    
    # Create custom configuration
    config = BehavioralMetricsConfig()
    
    # Customize settings
    config.metrics.min_views_for_conversion_rate = 5  # Require at least 5 views
    config.metrics.enable_temporal_features = True
    config.processing.enable_engagement_features = True
    config.output.generate_summary_report = True
    
    print("Custom Configuration Settings:")
    print(f"  Min views for conversion rate: {config.metrics.min_views_for_conversion_rate}")
    print(f"  Temporal features enabled: {config.metrics.enable_temporal_features}")
    print(f"  Engagement features enabled: {config.processing.enable_engagement_features}")
    
    # Save configuration for reference
    config.save_to_file("custom_behavioral_config.json")
    print("  Configuration saved to: custom_behavioral_config.json")
    
    return config


def generate_insights_report(metrics_df: pd.DataFrame, output_file: str = "behavioral_insights.txt"):
    """
    Generate a comprehensive insights report
    
    Args:
        metrics_df: DataFrame with behavioral metrics
        output_file: Path to save insights report
    """
    print(f"\nGenerating insights report: {output_file}")
    
    insights = []
    insights.append("BEHAVIORAL METRICS INSIGHTS REPORT")
    insights.append("=" * 50)
    insights.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    insights.append(f"Total items analyzed: {len(metrics_df):,}")
    insights.append("")
    
    # Key metrics
    if 'conversion_rate' in metrics_df.columns:
        avg_conversion = metrics_df['conversion_rate'].mean()
        median_conversion = metrics_df['conversion_rate'].median()
        insights.append(f"CONVERSION METRICS:")
        insights.append(f"  Average conversion rate: {avg_conversion:.3f} ({avg_conversion*100:.1f}%)")
        insights.append(f"  Median conversion rate: {median_conversion:.3f} ({median_conversion*100:.1f}%)")
        
        # High performers
        high_performers = metrics_df[metrics_df['conversion_rate'] > avg_conversion * 2]
        insights.append(f"  High-performing items (>2x avg): {len(high_performers):,}")
        insights.append("")
    
    # Traffic analysis
    if 'view_count' in metrics_df.columns:
        total_views = metrics_df['view_count'].sum()
        avg_views = metrics_df['view_count'].mean()
        insights.append(f"TRAFFIC ANALYSIS:")
        insights.append(f"  Total views: {total_views:,}")
        insights.append(f"  Average views per item: {avg_views:.1f}")
        
        # Traffic concentration
        top_10_percent = int(len(metrics_df) * 0.1)
        top_items_views = metrics_df.nlargest(top_10_percent, 'view_count')['view_count'].sum()
        traffic_concentration = (top_items_views / total_views) * 100
        insights.append(f"  Traffic concentration (top 10%): {traffic_concentration:.1f}%")
        insights.append("")
    
    # Engagement analysis
    if 'addtocart_rate' in metrics_df.columns:
        avg_addtocart = metrics_df['addtocart_rate'].mean()
        insights.append(f"ENGAGEMENT ANALYSIS:")
        insights.append(f"  Average add-to-cart rate: {avg_addtocart:.3f} ({avg_addtocart*100:.1f}%)")
        
        # Items with good engagement
        good_engagement = metrics_df[metrics_df['addtocart_rate'] > 0.1]  # >10% add-to-cart rate
        insights.append(f"  Items with >10% add-to-cart rate: {len(good_engagement):,}")
        insights.append("")
    
    # Recommendations
    insights.append("RECOMMENDATIONS:")
    
    if 'conversion_rate' in metrics_df.columns:
        low_conversion = metrics_df[metrics_df['conversion_rate'] < 0.01]  # <1% conversion
        if len(low_conversion) > 0:
            insights.append(f"  • Review {len(low_conversion):,} items with <1% conversion rate")
    
    if 'view_count' in metrics_df.columns and 'conversion_rate' in metrics_df.columns:
        high_traffic_low_conversion = metrics_df[
            (metrics_df['view_count'] > metrics_df['view_count'].quantile(0.8)) &
            (metrics_df['conversion_rate'] < metrics_df['conversion_rate'].median())
        ]
        if len(high_traffic_low_conversion) > 0:
            insights.append(f"  • Optimize {len(high_traffic_low_conversion):,} high-traffic, low-conversion items")
    
    # Save report
    with open(output_file, 'w') as f:
        f.write('\n'.join(insights))
    
    print("Insights report generated successfully!")
    
    # Also print key insights
    print("\nKey Insights:")
    for line in insights:
        if line.startswith("  Average") or line.startswith("  •"):
            print(line)


def main():
    """Main demonstration function"""
    print("BEHAVIORAL METRICS PROCESSING EXAMPLE")
    print("=" * 50)
    
    # Check if events.csv exists
    events_file = "events.csv"
    if not os.path.exists(events_file):
        print(f"Warning: {events_file} not found.")
        print("This example requires an events.csv file with columns:")
        print("  - timestamp, visitorid, event, itemid, transactionid")
        print("\nYou can:")
        print("  1. Use the sample_data_processing_demo.py to generate sample data")
        print("  2. Provide your own events.csv file")
        return
    
    # Load and analyze events
    results = load_and_analyze_events(events_file)
    
    if results is None:
        print("Failed to process events. Exiting.")
        return
    
    metrics_df = results['metrics_df']
    
    # Perform detailed analysis
    analyze_top_performers(metrics_df, top_n=5)
    analyze_conversion_patterns(metrics_df)
    
    # Demonstrate custom configuration
    custom_config = demonstrate_custom_configuration()
    
    # Generate insights report
    generate_insights_report(metrics_df)
    
    print("\n" + "=" * 50)
    print("EXAMPLE COMPLETE")
    print("=" * 50)
    print("Files generated:")
    print("  - behavioral_analysis_output/ (pipeline results)")
    print("  - custom_behavioral_config.json (configuration)")
    print("  - behavioral_insights.txt (insights report)")


if __name__ == "__main__":
    main()