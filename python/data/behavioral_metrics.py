"""
Behavioral Metrics Aggregation Module for Smart Product Re-Ranking System
Calculates behavioral metrics from event data including view counts, conversion rates, etc.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ItemBehavioralMetrics:
    """Structured representation of behavioral metrics for an item"""
    itemid: int
    view_count: int
    addtocart_count: int
    transaction_count: int
    unique_visitors: int
    addtocart_rate: float  # addtocart/view
    conversion_rate: float  # transaction/view
    cart_conversion_rate: float  # transaction/addtocart
    avg_time_to_cart: Optional[float] = None
    avg_time_to_purchase: Optional[float] = None
    popularity_score: Optional[float] = None


class BehavioralMetricsCalculator:
    """Calculator for behavioral metrics from event data"""
    
    def __init__(self):
        """Initialize the behavioral metrics calculator"""
        self.required_columns = ['timestamp', 'visitorid', 'event', 'itemid']
        
    def validate_event_data(self, df: pd.DataFrame) -> bool:
        """
        Validate that the event data has required columns
        
        Args:
            df: DataFrame with event data
            
        Returns:
            True if valid, False otherwise
        """
        missing_cols = set(self.required_columns) - set(df.columns)
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return False
        
        if df.empty:
            logger.error("Event data is empty")
            return False
            
        return True
    
    def calculate_view_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate view counts per item from 'view' events
        
        Args:
            df: DataFrame with event data
            
        Returns:
            DataFrame with itemid, view_count, and unique_visitors
        """
        if not self.validate_event_data(df):
            raise ValueError("Invalid event data")
        
        # Filter to view events only
        view_events = df[df['event'] == 'view'].copy()
        
        if view_events.empty:
            logger.warning("No view events found in data")
            return pd.DataFrame(columns=['itemid', 'view_count', 'unique_visitors'])
        
        # Calculate view counts and unique visitors per item
        view_metrics = view_events.groupby('itemid').agg({
            'visitorid': ['count', 'nunique']
        }).reset_index()
        
        # Flatten column names
        view_metrics.columns = ['itemid', 'view_count', 'unique_visitors']
        
        logger.info(f"Calculated view metrics for {len(view_metrics)} items")
        return view_metrics
    
    def calculate_addtocart_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate addtocart counts per item from 'addtocart' events
        
        Args:
            df: DataFrame with event data
            
        Returns:
            DataFrame with itemid and addtocart_count
        """
        if not self.validate_event_data(df):
            raise ValueError("Invalid event data")
        
        # Filter to addtocart events only
        addtocart_events = df[df['event'] == 'addtocart'].copy()
        
        if addtocart_events.empty:
            logger.warning("No addtocart events found in data")
            return pd.DataFrame(columns=['itemid', 'addtocart_count'])
        
        # Calculate addtocart counts per item
        addtocart_metrics = addtocart_events.groupby('itemid').size().reset_index(name='addtocart_count')
        
        logger.info(f"Calculated addtocart metrics for {len(addtocart_metrics)} items")
        return addtocart_metrics
    
    def calculate_transaction_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate transaction counts per item from 'transaction' events
        
        Args:
            df: DataFrame with event data
            
        Returns:
            DataFrame with itemid and transaction_count
        """
        if not self.validate_event_data(df):
            raise ValueError("Invalid event data")
        
        # Filter to transaction events only
        transaction_events = df[df['event'] == 'transaction'].copy()
        
        if transaction_events.empty:
            logger.warning("No transaction events found in data")
            return pd.DataFrame(columns=['itemid', 'transaction_count'])
        
        # Calculate transaction counts per item
        transaction_metrics = transaction_events.groupby('itemid').size().reset_index(name='transaction_count')
        
        logger.info(f"Calculated transaction metrics for {len(transaction_metrics)} items")
        return transaction_metrics
    
    def calculate_conversion_rates(self, view_metrics: pd.DataFrame, 
                                 addtocart_metrics: pd.DataFrame,
                                 transaction_metrics: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate conversion rates by combining view, addtocart, and transaction metrics
        
        Args:
            view_metrics: DataFrame with view counts per item
            addtocart_metrics: DataFrame with addtocart counts per item
            transaction_metrics: DataFrame with transaction counts per item
            
        Returns:
            DataFrame with all behavioral metrics and conversion rates
        """
        # Start with view metrics as the base (all items should have views)
        metrics = view_metrics.copy()
        
        # Merge addtocart metrics
        metrics = metrics.merge(addtocart_metrics, on='itemid', how='left')
        metrics['addtocart_count'] = metrics['addtocart_count'].fillna(0).astype(int)
        
        # Merge transaction metrics
        metrics = metrics.merge(transaction_metrics, on='itemid', how='left')
        metrics['transaction_count'] = metrics['transaction_count'].fillna(0).astype(int)
        
        # Calculate conversion rates
        # Addtocart rate: addtocart events / view events
        metrics['addtocart_rate'] = np.where(
            metrics['view_count'] > 0,
            metrics['addtocart_count'] / metrics['view_count'],
            0.0
        )
        
        # Conversion rate: transaction events / view events
        metrics['conversion_rate'] = np.where(
            metrics['view_count'] > 0,
            metrics['transaction_count'] / metrics['view_count'],
            0.0
        )
        
        # Cart conversion rate: transaction events / addtocart events
        metrics['cart_conversion_rate'] = np.where(
            metrics['addtocart_count'] > 0,
            metrics['transaction_count'] / metrics['addtocart_count'],
            0.0
        )
        
        logger.info(f"Calculated conversion rates for {len(metrics)} items")
        return metrics
    
    def calculate_all_behavioral_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all behavioral metrics from event data
        
        Args:
            df: DataFrame with event data
            
        Returns:
            DataFrame with comprehensive behavioral metrics per item
        """
        logger.info("Starting behavioral metrics calculation")
        
        if not self.validate_event_data(df):
            raise ValueError("Invalid event data")
        
        # Calculate individual metrics
        view_metrics = self.calculate_view_metrics(df)
        addtocart_metrics = self.calculate_addtocart_metrics(df)
        transaction_metrics = self.calculate_transaction_metrics(df)
        
        # Combine and calculate conversion rates
        all_metrics = self.calculate_conversion_rates(
            view_metrics, addtocart_metrics, transaction_metrics
        )
        
        # Add popularity score (simple view-based popularity)
        if len(all_metrics) > 0:
            max_views = all_metrics['view_count'].max()
            all_metrics['popularity_score'] = all_metrics['view_count'] / max_views if max_views > 0 else 0.0
        
        logger.info(f"Completed behavioral metrics calculation for {len(all_metrics)} items")
        return all_metrics
    
    def calculate_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate temporal features like average time to cart and purchase
        Note: This method is deprecated. Use feature_engineering.py for comprehensive temporal features.
        
        Args:
            df: DataFrame with event data
            
        Returns:
            DataFrame with temporal features per item
        """
        logger.warning("calculate_temporal_features is deprecated. Use feature_engineering.TemporalFeatureExtractor for comprehensive temporal features.")
        
        if not self.validate_event_data(df):
            raise ValueError("Invalid event data")
        
        # Convert timestamp to datetime for easier manipulation
        df_temp = df.copy()
        df_temp['datetime'] = pd.to_datetime(df_temp['timestamp'], unit='ms')
        
        temporal_features = []
        
        # Group by visitor and item to calculate time differences
        for (visitorid, itemid), group in df_temp.groupby(['visitorid', 'itemid']):
            group_sorted = group.sort_values('datetime')
            
            # Find first view event
            view_events = group_sorted[group_sorted['event'] == 'view']
            if view_events.empty:
                continue
                
            first_view_time = view_events.iloc[0]['datetime']
            
            # Calculate time to addtocart
            addtocart_events = group_sorted[group_sorted['event'] == 'addtocart']
            time_to_cart = None
            if not addtocart_events.empty:
                first_addtocart_time = addtocart_events.iloc[0]['datetime']
                time_to_cart = (first_addtocart_time - first_view_time).total_seconds()
            
            # Calculate time to purchase
            transaction_events = group_sorted[group_sorted['event'] == 'transaction']
            time_to_purchase = None
            if not transaction_events.empty:
                first_transaction_time = transaction_events.iloc[0]['datetime']
                time_to_purchase = (first_transaction_time - first_view_time).total_seconds()
            
            temporal_features.append({
                'itemid': itemid,
                'visitorid': visitorid,
                'time_to_cart': time_to_cart,
                'time_to_purchase': time_to_purchase
            })
        
        if not temporal_features:
            return pd.DataFrame(columns=['itemid', 'avg_time_to_cart', 'avg_time_to_purchase'])
        
        # Convert to DataFrame and aggregate by item
        temporal_df = pd.DataFrame(temporal_features)
        
        # Calculate average times per item
        temporal_aggregated = temporal_df.groupby('itemid').agg({
            'time_to_cart': 'mean',
            'time_to_purchase': 'mean'
        }).reset_index()
        
        temporal_aggregated.columns = ['itemid', 'avg_time_to_cart', 'avg_time_to_purchase']
        
        logger.info(f"Calculated temporal features for {len(temporal_aggregated)} items")
        return temporal_aggregated
    
    def get_metrics_summary(self, metrics_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get summary statistics of behavioral metrics
        
        Args:
            metrics_df: DataFrame with behavioral metrics
            
        Returns:
            Dictionary with summary statistics
        """
        if metrics_df.empty:
            return {"error": "Metrics DataFrame is empty"}
        
        summary = {
            "total_items": len(metrics_df),
            "total_views": int(metrics_df['view_count'].sum()),
            "total_addtocarts": int(metrics_df['addtocart_count'].sum()),
            "total_transactions": int(metrics_df['transaction_count'].sum()),
            "avg_addtocart_rate": float(metrics_df['addtocart_rate'].mean()),
            "avg_conversion_rate": float(metrics_df['conversion_rate'].mean()),
            "avg_cart_conversion_rate": float(metrics_df['cart_conversion_rate'].mean()),
            "items_with_purchases": int((metrics_df['transaction_count'] > 0).sum()),
            "items_with_addtocarts": int((metrics_df['addtocart_count'] > 0).sum())
        }
        
        # Add percentile information
        for metric in ['view_count', 'addtocart_rate', 'conversion_rate', 'cart_conversion_rate']:
            if metric in metrics_df.columns:
                summary[f"{metric}_percentiles"] = {
                    "25th": float(metrics_df[metric].quantile(0.25)),
                    "50th": float(metrics_df[metric].quantile(0.50)),
                    "75th": float(metrics_df[metric].quantile(0.75)),
                    "95th": float(metrics_df[metric].quantile(0.95))
                }
        
        return summary


# Convenience functions
def calculate_behavioral_metrics(events_df: pd.DataFrame, 
                               include_temporal: bool = False) -> pd.DataFrame:
    """
    Convenience function to calculate behavioral metrics from events DataFrame
    
    Args:
        events_df: DataFrame with event data
        include_temporal: Whether to include temporal features
        
    Returns:
        DataFrame with behavioral metrics
    """
    calculator = BehavioralMetricsCalculator()
    
    # Calculate core behavioral metrics
    metrics = calculator.calculate_all_behavioral_metrics(events_df)
    
    # Add temporal features if requested
    if include_temporal and not events_df.empty:
        temporal_features = calculator.calculate_temporal_features(events_df)
        if not temporal_features.empty:
            metrics = metrics.merge(temporal_features, on='itemid', how='left')
    
    return metrics


def get_top_performing_items(metrics_df: pd.DataFrame, 
                           metric: str = 'conversion_rate',
                           top_n: int = 10) -> pd.DataFrame:
    """
    Get top performing items based on a specific metric
    
    Args:
        metrics_df: DataFrame with behavioral metrics
        metric: Metric to sort by
        top_n: Number of top items to return
        
    Returns:
        DataFrame with top performing items
    """
    if metric not in metrics_df.columns:
        raise ValueError(f"Metric '{metric}' not found in metrics DataFrame")
    
    return metrics_df.nlargest(top_n, metric)


if __name__ == "__main__":
    # Example usage
    try:
        from csv_loader import load_events_csv
        
        # Load events data
        events_df = load_events_csv("events.csv")
        print(f"Loaded {len(events_df)} events")
        
        # Calculate behavioral metrics
        metrics = calculate_behavioral_metrics(events_df, include_temporal=True)
        print(f"Calculated metrics for {len(metrics)} items")
        
        # Get summary
        calculator = BehavioralMetricsCalculator()
        summary = calculator.get_metrics_summary(metrics)
        print("\nBehavioral Metrics Summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        
        # Show top performing items
        top_items = get_top_performing_items(metrics, 'conversion_rate', 5)
        print(f"\nTop 5 items by conversion rate:")
        print(top_items[['itemid', 'view_count', 'conversion_rate', 'addtocart_rate']])
        
    except FileNotFoundError:
        print("events.csv file not found. Please ensure the file exists in the current directory.")
    except Exception as e:
        print(f"Error: {str(e)}")