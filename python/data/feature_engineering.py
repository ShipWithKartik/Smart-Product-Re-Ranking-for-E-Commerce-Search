"""
Feature Engineering Module for Smart Product Re-Ranking System
Implements temporal and engagement features from event data
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TemporalFeatures:
    """Structured representation of temporal features for an item"""
    itemid: int
    avg_time_to_cart: Optional[float] = None
    avg_time_to_purchase: Optional[float] = None
    median_time_to_cart: Optional[float] = None
    median_time_to_purchase: Optional[float] = None
    hour_of_day_pattern: Optional[Dict[int, int]] = None
    day_of_week_pattern: Optional[Dict[int, int]] = None
    peak_activity_hour: Optional[int] = None
    activity_span_hours: Optional[float] = None


@dataclass
class EngagementFeatures:
    """Structured representation of engagement features for an item"""
    itemid: int
    unique_visitors: int
    total_engagement_score: float
    popularity_score: float
    engagement_intensity: float
    visitor_loyalty_score: float
    repeat_visitor_rate: float
    session_depth_avg: float


class TemporalFeatureExtractor:
    """Extracts temporal features from event data"""
    
    def __init__(self):
        """Initialize the temporal feature extractor"""
        self.required_columns = ['timestamp', 'visitorid', 'event', 'itemid']
    
    def validate_event_data(self, df: pd.DataFrame) -> bool:
        """Validate that the event data has required columns"""
        missing_cols = set(self.required_columns) - set(df.columns)
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return False
        
        if df.empty:
            logger.error("Event data is empty")
            return False
            
        return True
    
    def extract_time_based_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract time-based patterns from timestamp data
        
        Args:
            df: DataFrame with event data
            
        Returns:
            DataFrame with time-based patterns per item
        """
        if not self.validate_event_data(df):
            raise ValueError("Invalid event data")
        
        logger.info("Extracting time-based patterns from timestamp data")
        
        # Convert timestamp to datetime
        df_temp = df.copy()
        df_temp['datetime'] = pd.to_datetime(df_temp['timestamp'], unit='ms')
        df_temp['hour'] = df_temp['datetime'].dt.hour
        df_temp['day_of_week'] = df_temp['datetime'].dt.dayofweek
        df_temp['date'] = df_temp['datetime'].dt.date
        
        temporal_patterns = []
        
        # Group by item to calculate patterns
        for itemid, item_group in df_temp.groupby('itemid'):
            # Hour of day pattern
            hour_pattern = item_group['hour'].value_counts().to_dict()
            peak_hour = item_group['hour'].mode().iloc[0] if not item_group['hour'].mode().empty else None
            
            # Day of week pattern
            dow_pattern = item_group['day_of_week'].value_counts().to_dict()
            
            # Activity span (hours between first and last event)
            time_span = (item_group['datetime'].max() - item_group['datetime'].min()).total_seconds() / 3600
            
            # Daily activity patterns
            daily_events = item_group.groupby('date').size()
            avg_daily_events = daily_events.mean() if len(daily_events) > 0 else 0
            
            temporal_patterns.append({
                'itemid': itemid,
                'peak_activity_hour': peak_hour,
                'activity_span_hours': time_span,
                'avg_daily_events': avg_daily_events,
                'total_days_active': len(daily_events),
                'hour_pattern_entropy': self._calculate_entropy(list(hour_pattern.values())),
                'dow_pattern_entropy': self._calculate_entropy(list(dow_pattern.values()))
            })
        
        temporal_df = pd.DataFrame(temporal_patterns)
        logger.info(f"Extracted time-based patterns for {len(temporal_df)} items")
        
        return temporal_df
    
    def calculate_time_to_action_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate average time-to-cart and time-to-purchase metrics
        
        Args:
            df: DataFrame with event data
            
        Returns:
            DataFrame with time-to-action metrics per item
        """
        if not self.validate_event_data(df):
            raise ValueError("Invalid event data")
        
        logger.info("Calculating time-to-cart and time-to-purchase metrics")
        
        # Convert timestamp to datetime
        df_temp = df.copy()
        df_temp['datetime'] = pd.to_datetime(df_temp['timestamp'], unit='ms')
        
        time_metrics = []
        
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
            
            # Calculate session depth (number of events in this visitor-item session)
            session_depth = len(group_sorted)
            
            time_metrics.append({
                'itemid': itemid,
                'visitorid': visitorid,
                'time_to_cart': time_to_cart,
                'time_to_purchase': time_to_purchase,
                'session_depth': session_depth,
                'total_events_in_session': len(group_sorted)
            })
        
        if not time_metrics:
            return pd.DataFrame(columns=[
                'itemid', 'avg_time_to_cart', 'avg_time_to_purchase', 
                'median_time_to_cart', 'median_time_to_purchase',
                'avg_session_depth', 'cart_conversion_sessions', 'purchase_conversion_sessions'
            ])
        
        # Convert to DataFrame and aggregate by item
        time_df = pd.DataFrame(time_metrics)
        
        # Calculate aggregated metrics per item
        aggregated_metrics = []
        
        for itemid, item_group in time_df.groupby('itemid'):
            # Time to cart metrics
            cart_times = item_group['time_to_cart'].dropna()
            avg_time_to_cart = cart_times.mean() if len(cart_times) > 0 else None
            median_time_to_cart = cart_times.median() if len(cart_times) > 0 else None
            
            # Time to purchase metrics
            purchase_times = item_group['time_to_purchase'].dropna()
            avg_time_to_purchase = purchase_times.mean() if len(purchase_times) > 0 else None
            median_time_to_purchase = purchase_times.median() if len(purchase_times) > 0 else None
            
            # Session metrics
            avg_session_depth = item_group['session_depth'].mean()
            
            # Conversion session counts
            cart_conversion_sessions = len(cart_times)
            purchase_conversion_sessions = len(purchase_times)
            total_sessions = len(item_group)
            
            aggregated_metrics.append({
                'itemid': itemid,
                'avg_time_to_cart': avg_time_to_cart,
                'avg_time_to_purchase': avg_time_to_purchase,
                'median_time_to_cart': median_time_to_cart,
                'median_time_to_purchase': median_time_to_purchase,
                'avg_session_depth': avg_session_depth,
                'cart_conversion_sessions': cart_conversion_sessions,
                'purchase_conversion_sessions': purchase_conversion_sessions,
                'total_sessions': total_sessions,
                'session_cart_rate': cart_conversion_sessions / total_sessions if total_sessions > 0 else 0,
                'session_purchase_rate': purchase_conversion_sessions / total_sessions if total_sessions > 0 else 0
            })
        
        result_df = pd.DataFrame(aggregated_metrics)
        logger.info(f"Calculated time-to-action metrics for {len(result_df)} items")
        
        return result_df
    
    def _calculate_entropy(self, values: List[int]) -> float:
        """Calculate entropy of a distribution"""
        if not values or sum(values) == 0:
            return 0.0
        
        total = sum(values)
        probabilities = [v / total for v in values if v > 0]
        
        entropy = -sum(p * np.log2(p) for p in probabilities)
        return entropy
    
    def extract_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract all temporal features from event data
        
        Args:
            df: DataFrame with event data
            
        Returns:
            DataFrame with comprehensive temporal features
        """
        if not self.validate_event_data(df):
            raise ValueError("Invalid event data")
        
        logger.info("Extracting comprehensive temporal features")
        
        # Get time-based patterns
        time_patterns = self.extract_time_based_patterns(df)
        
        # Get time-to-action metrics
        time_to_action = self.calculate_time_to_action_metrics(df)
        
        # Merge temporal features
        temporal_features = time_patterns.merge(time_to_action, on='itemid', how='outer')
        
        # Fill missing values with appropriate defaults
        temporal_features = temporal_features.fillna(0)
        
        logger.info(f"Extracted temporal features for {len(temporal_features)} items")
        return temporal_features


class EngagementFeatureExtractor:
    """Extracts engagement features from event data"""
    
    def __init__(self):
        """Initialize the engagement feature extractor"""
        self.required_columns = ['timestamp', 'visitorid', 'event', 'itemid']
    
    def validate_event_data(self, df: pd.DataFrame) -> bool:
        """Validate that the event data has required columns"""
        missing_cols = set(self.required_columns) - set(df.columns)
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return False
        
        if df.empty:
            logger.error("Event data is empty")
            return False
            
        return True
    
    def calculate_unique_visitor_counts(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate unique visitor counts per item
        
        Args:
            df: DataFrame with event data
            
        Returns:
            DataFrame with unique visitor counts per item
        """
        if not self.validate_event_data(df):
            raise ValueError("Invalid event data")
        
        logger.info("Calculating unique visitor counts per item")
        
        # Calculate unique visitors per item
        visitor_counts = df.groupby('itemid')['visitorid'].nunique().reset_index()
        visitor_counts.columns = ['itemid', 'unique_visitors']
        
        # Calculate total events per item for additional context
        event_counts = df.groupby('itemid').size().reset_index(name='total_events')
        
        # Merge the metrics
        result = visitor_counts.merge(event_counts, on='itemid', how='left')
        
        # Calculate events per visitor ratio
        result['events_per_visitor'] = result['total_events'] / result['unique_visitors']
        
        logger.info(f"Calculated unique visitor counts for {len(result)} items")
        return result
    
    def calculate_popularity_scoring(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Implement popularity scoring based on total engagement
        
        Args:
            df: DataFrame with event data
            
        Returns:
            DataFrame with popularity scores per item
        """
        if not self.validate_event_data(df):
            raise ValueError("Invalid event data")
        
        logger.info("Calculating popularity scoring based on total engagement")
        
        # Calculate engagement metrics per item
        engagement_metrics = []
        
        for itemid, item_group in df.groupby('itemid'):
            # Basic counts
            total_events = len(item_group)
            unique_visitors = item_group['visitorid'].nunique()
            
            # Event type breakdown
            event_counts = item_group['event'].value_counts()
            view_count = event_counts.get('view', 0)
            addtocart_count = event_counts.get('addtocart', 0)
            transaction_count = event_counts.get('transaction', 0)
            
            # Weighted engagement score (transactions worth more than views)
            engagement_weights = {'view': 1.0, 'addtocart': 3.0, 'transaction': 10.0}
            weighted_engagement = sum(event_counts.get(event, 0) * weight 
                                    for event, weight in engagement_weights.items())
            
            # Engagement intensity (events per unique visitor)
            engagement_intensity = total_events / unique_visitors if unique_visitors > 0 else 0
            
            # Visitor loyalty (repeat visitors)
            visitor_event_counts = item_group['visitorid'].value_counts()
            repeat_visitors = (visitor_event_counts > 1).sum()
            visitor_loyalty_score = repeat_visitors / unique_visitors if unique_visitors > 0 else 0
            
            engagement_metrics.append({
                'itemid': itemid,
                'total_events': total_events,
                'unique_visitors': unique_visitors,
                'view_count': view_count,
                'addtocart_count': addtocart_count,
                'transaction_count': transaction_count,
                'weighted_engagement_score': weighted_engagement,
                'engagement_intensity': engagement_intensity,
                'repeat_visitors': repeat_visitors,
                'visitor_loyalty_score': visitor_loyalty_score
            })
        
        engagement_df = pd.DataFrame(engagement_metrics)
        
        # Calculate normalized popularity scores
        if len(engagement_df) > 0:
            # Popularity score based on weighted engagement (normalized 0-1)
            max_engagement = engagement_df['weighted_engagement_score'].max()
            engagement_df['popularity_score'] = (
                engagement_df['weighted_engagement_score'] / max_engagement 
                if max_engagement > 0 else 0
            )
            
            # Alternative popularity score based on unique visitors (normalized 0-1)
            max_visitors = engagement_df['unique_visitors'].max()
            engagement_df['visitor_popularity_score'] = (
                engagement_df['unique_visitors'] / max_visitors 
                if max_visitors > 0 else 0
            )
            
            # Combined popularity score (weighted average)
            engagement_df['combined_popularity_score'] = (
                0.7 * engagement_df['popularity_score'] + 
                0.3 * engagement_df['visitor_popularity_score']
            )
        
        logger.info(f"Calculated popularity scores for {len(engagement_df)} items")
        return engagement_df
    
    def calculate_visitor_loyalty_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate visitor loyalty metrics (repeat engagement patterns)
        
        Args:
            df: DataFrame with event data
            
        Returns:
            DataFrame with visitor loyalty metrics per item
        """
        if not self.validate_event_data(df):
            raise ValueError("Invalid event data")
        
        logger.info("Calculating visitor loyalty metrics")
        
        loyalty_metrics = []
        
        for itemid, item_group in df.groupby('itemid'):
            # Calculate visitor engagement patterns
            visitor_patterns = item_group.groupby('visitorid').agg({
                'event': 'count',
                'timestamp': ['min', 'max']
            }).reset_index()
            
            visitor_patterns.columns = ['visitorid', 'event_count', 'first_timestamp', 'last_timestamp']
            
            # Calculate engagement duration per visitor (in seconds)
            visitor_patterns['engagement_duration'] = (
                visitor_patterns['last_timestamp'] - visitor_patterns['first_timestamp']
            ) / 1000  # Convert from milliseconds to seconds
            
            # Loyalty metrics
            total_visitors = len(visitor_patterns)
            repeat_visitors = (visitor_patterns['event_count'] > 1).sum()
            highly_engaged_visitors = (visitor_patterns['event_count'] >= 5).sum()
            
            # Average metrics
            avg_events_per_visitor = visitor_patterns['event_count'].mean()
            avg_engagement_duration = visitor_patterns['engagement_duration'].mean()
            
            # Loyalty scores
            repeat_visitor_rate = repeat_visitors / total_visitors if total_visitors > 0 else 0
            high_engagement_rate = highly_engaged_visitors / total_visitors if total_visitors > 0 else 0
            
            loyalty_metrics.append({
                'itemid': itemid,
                'total_visitors': total_visitors,
                'repeat_visitors': repeat_visitors,
                'highly_engaged_visitors': highly_engaged_visitors,
                'repeat_visitor_rate': repeat_visitor_rate,
                'high_engagement_rate': high_engagement_rate,
                'avg_events_per_visitor': avg_events_per_visitor,
                'avg_engagement_duration_seconds': avg_engagement_duration
            })
        
        loyalty_df = pd.DataFrame(loyalty_metrics)
        logger.info(f"Calculated visitor loyalty metrics for {len(loyalty_df)} items")
        
        return loyalty_df
    
    def extract_engagement_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract all engagement features from event data
        
        Args:
            df: DataFrame with event data
            
        Returns:
            DataFrame with comprehensive engagement features
        """
        if not self.validate_event_data(df):
            raise ValueError("Invalid event data")
        
        logger.info("Extracting comprehensive engagement features")
        
        # Get unique visitor counts
        visitor_counts = self.calculate_unique_visitor_counts(df)
        
        # Get popularity scores
        popularity_scores = self.calculate_popularity_scoring(df)
        
        # Get visitor loyalty metrics
        loyalty_metrics = self.calculate_visitor_loyalty_metrics(df)
        
        # Merge all engagement features
        engagement_features = visitor_counts.merge(popularity_scores, on='itemid', how='outer')
        engagement_features = engagement_features.merge(loyalty_metrics, on='itemid', how='outer')
        
        # Fill missing values with 0
        engagement_features = engagement_features.fillna(0)
        
        logger.info(f"Extracted engagement features for {len(engagement_features)} items")
        return engagement_features


class FeatureEngineeringPipeline:
    """Complete feature engineering pipeline combining temporal and engagement features"""
    
    def __init__(self):
        """Initialize the feature engineering pipeline"""
        self.temporal_extractor = TemporalFeatureExtractor()
        self.engagement_extractor = EngagementFeatureExtractor()
    
    def extract_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract all temporal and engagement features from event data
        
        Args:
            df: DataFrame with event data
            
        Returns:
            DataFrame with comprehensive features per item
        """
        logger.info("Starting comprehensive feature extraction")
        
        if df.empty:
            logger.warning("Empty DataFrame provided")
            return pd.DataFrame()
        
        # Extract temporal features
        temporal_patterns = self.temporal_extractor.extract_time_based_patterns(df)
        time_to_action = self.temporal_extractor.calculate_time_to_action_metrics(df)
        
        # Extract engagement features
        visitor_counts = self.engagement_extractor.calculate_unique_visitor_counts(df)
        popularity_scores = self.engagement_extractor.calculate_popularity_scoring(df)
        loyalty_metrics = self.engagement_extractor.calculate_visitor_loyalty_metrics(df)
        
        # Merge all features
        features = temporal_patterns
        
        # Merge time-to-action metrics
        if not time_to_action.empty:
            features = features.merge(time_to_action, on='itemid', how='left')
        
        # Merge engagement features
        if not visitor_counts.empty:
            features = features.merge(visitor_counts, on='itemid', how='left')
        
        if not popularity_scores.empty:
            features = features.merge(popularity_scores, on='itemid', how='left')
        
        if not loyalty_metrics.empty:
            features = features.merge(loyalty_metrics, on='itemid', how='left')
        
        logger.info(f"Extracted comprehensive features for {len(features)} items")
        logger.info(f"Feature columns: {list(features.columns)}")
        
        return features
    
    def get_feature_summary(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get summary statistics of extracted features
        
        Args:
            features_df: DataFrame with extracted features
            
        Returns:
            Dictionary with feature summary statistics
        """
        if features_df.empty:
            return {"error": "Features DataFrame is empty"}
        
        summary = {
            "total_items": len(features_df),
            "feature_count": len(features_df.columns) - 1,  # Exclude itemid
            "feature_columns": [col for col in features_df.columns if col != 'itemid']
        }
        
        # Temporal feature summaries
        temporal_cols = [col for col in features_df.columns if 'time' in col.lower() or 'hour' in col.lower()]
        if temporal_cols:
            summary["temporal_features"] = {}
            for col in temporal_cols:
                if features_df[col].dtype in ['int64', 'float64']:
                    summary["temporal_features"][col] = {
                        "mean": float(features_df[col].mean()) if features_df[col].notna().any() else None,
                        "median": float(features_df[col].median()) if features_df[col].notna().any() else None,
                        "std": float(features_df[col].std()) if features_df[col].notna().any() else None
                    }
        
        # Engagement feature summaries
        engagement_cols = [col for col in features_df.columns if any(word in col.lower() 
                          for word in ['visitor', 'engagement', 'popularity', 'loyalty'])]
        if engagement_cols:
            summary["engagement_features"] = {}
            for col in engagement_cols:
                if features_df[col].dtype in ['int64', 'float64']:
                    summary["engagement_features"][col] = {
                        "mean": float(features_df[col].mean()) if features_df[col].notna().any() else None,
                        "median": float(features_df[col].median()) if features_df[col].notna().any() else None,
                        "std": float(features_df[col].std()) if features_df[col].notna().any() else None
                    }
        
        return summary


# Convenience functions
def extract_temporal_and_engagement_features(events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function to extract all temporal and engagement features
    
    Args:
        events_df: DataFrame with event data
        
    Returns:
        DataFrame with extracted features
    """
    pipeline = FeatureEngineeringPipeline()
    return pipeline.extract_all_features(events_df)


def get_top_items_by_feature(features_df: pd.DataFrame, 
                           feature_name: str, 
                           top_n: int = 10) -> pd.DataFrame:
    """
    Get top items by a specific feature
    
    Args:
        features_df: DataFrame with features
        feature_name: Name of the feature to sort by
        top_n: Number of top items to return
        
    Returns:
        DataFrame with top items
    """
    if feature_name not in features_df.columns:
        raise ValueError(f"Feature '{feature_name}' not found in features DataFrame")
    
    return features_df.nlargest(top_n, feature_name)


if __name__ == "__main__":
    # Example usage
    try:
        from csv_loader import load_events_csv
        
        # Load events data
        events_df = load_events_csv("events.csv")
        print(f"Loaded {len(events_df)} events")
        
        # Extract all features
        features = extract_temporal_and_engagement_features(events_df)
        print(f"Extracted features for {len(features)} items")
        
        # Get feature summary
        pipeline = FeatureEngineeringPipeline()
        summary = pipeline.get_feature_summary(features)
        print("\nFeature Summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        
        # Show top items by popularity
        if 'popularity_score' in features.columns:
            top_popular = get_top_items_by_feature(features, 'popularity_score', 5)
            print(f"\nTop 5 items by popularity score:")
            print(top_popular[['itemid', 'popularity_score', 'unique_visitors', 'total_events']])
        
    except FileNotFoundError:
        print("events.csv file not found. Please ensure the file exists in the current directory.")
    except Exception as e:
        print(f"Error: {str(e)}")