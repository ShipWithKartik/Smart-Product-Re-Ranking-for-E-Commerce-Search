"""
Composite Behavioral Features Module for Smart Product Re-Ranking System
Implements composite features combining behavioral metrics for enhanced ML model performance
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from dataclasses import dataclass
from enum import Enum
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CompositeFeatures:
    """Structured representation of composite behavioral features for an item"""
    itemid: int
    engagement_intensity_score: float
    visitor_loyalty_score: float
    repeat_engagement_rate: float
    performance_bucket: str
    engagement_bucket: str
    loyalty_bucket: str
    composite_score: float
    feature_validation_status: str


class PerformanceBucket(Enum):
    """Performance bucket categories based on behavioral metrics"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    PREMIUM = "premium"


class EngagementBucket(Enum):
    """Engagement bucket categories"""
    MINIMAL = "minimal"
    MODERATE = "moderate"
    HIGH = "high"
    EXCEPTIONAL = "exceptional"


class LoyaltyBucket(Enum):
    """Loyalty bucket categories"""
    TRANSIENT = "transient"
    CASUAL = "casual"
    LOYAL = "loyal"
    DEVOTED = "devoted"


class CompositeFeatureBuilder:
    """Builder for composite behavioral features"""
    
    def __init__(self, 
                 engagement_weights: Optional[Dict[str, float]] = None,
                 performance_thresholds: Optional[Dict[str, List[float]]] = None):
        """
        Initialize the composite feature builder
        
        Args:
            engagement_weights: Weights for different event types in engagement scoring
            performance_thresholds: Thresholds for performance bucketing
        """
        # Default engagement weights (transaction > addtocart > view)
        self.engagement_weights = engagement_weights or {
            'view': 1.0,
            'addtocart': 3.0,
            'transaction': 10.0
        }
        
        # Default performance thresholds (percentiles)
        self.performance_thresholds = performance_thresholds or {
            'conversion_rate': [0.25, 0.50, 0.75],  # 25th, 50th, 75th percentiles
            'engagement_intensity': [0.33, 0.66, 0.90],
            'visitor_loyalty': [0.20, 0.50, 0.80]
        }
        
        self.required_columns = [
            'itemid', 'view_count', 'addtocart_count', 'transaction_count',
            'unique_visitors', 'addtocart_rate', 'conversion_rate', 'cart_conversion_rate'
        ]
    
    def validate_input_data(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate input data for composite feature generation
        
        Args:
            df: DataFrame with behavioral metrics
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check for required columns
        missing_cols = set(self.required_columns) - set(df.columns)
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
        
        # Check for empty DataFrame
        if df.empty:
            errors.append("Input DataFrame is empty")
        
        # Check for negative values in count columns
        count_columns = ['view_count', 'addtocart_count', 'transaction_count', 'unique_visitors']
        for col in count_columns:
            if col in df.columns and (df[col] < 0).any():
                errors.append(f"Negative values found in {col}")
        
        # Check for invalid rates (should be between 0 and 1, or reasonable for conversion rates)
        rate_columns = ['addtocart_rate', 'conversion_rate', 'cart_conversion_rate']
        for col in rate_columns:
            if col in df.columns:
                if (df[col] < 0).any():
                    errors.append(f"Negative values found in {col}")
                if (df[col] > 10).any():  # Allow for rates > 1 in case of multiple events per visitor
                    warnings.warn(f"Unusually high values found in {col} (>10)")
        
        # Check for logical inconsistencies
        if 'view_count' in df.columns and 'unique_visitors' in df.columns:
            if (df['unique_visitors'] > df['view_count']).any():
                errors.append("Logical inconsistency: unique_visitors > view_count for some items")
        
        return len(errors) == 0, errors
    
    def calculate_engagement_intensity_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build engagement intensity scores combining views, carts, and purchases
        
        Args:
            df: DataFrame with behavioral metrics
            
        Returns:
            DataFrame with engagement intensity scores
        """
        logger.info("Calculating engagement intensity scores")
        
        # Validate input
        is_valid, errors = self.validate_input_data(df)
        if not is_valid:
            raise ValueError(f"Input validation failed: {errors}")
        
        result_df = df.copy()
        
        # Calculate weighted engagement score
        result_df['weighted_engagement_raw'] = (
            result_df['view_count'] * self.engagement_weights['view'] +
            result_df['addtocart_count'] * self.engagement_weights['addtocart'] +
            result_df['transaction_count'] * self.engagement_weights['transaction']
        )
        
        # Calculate engagement intensity (weighted engagement per unique visitor)
        result_df['engagement_intensity_raw'] = np.where(
            result_df['unique_visitors'] > 0,
            result_df['weighted_engagement_raw'] / result_df['unique_visitors'],
            0.0
        )
        
        # Normalize engagement intensity score (0-1 scale)
        max_intensity = result_df['engagement_intensity_raw'].max()
        result_df['engagement_intensity_score'] = np.where(
            max_intensity > 0,
            result_df['engagement_intensity_raw'] / max_intensity,
            0.0
        )
        
        # Calculate engagement diversity (how balanced the engagement types are)
        total_events = result_df['view_count'] + result_df['addtocart_count'] + result_df['transaction_count']
        
        # Shannon entropy for engagement diversity
        result_df['engagement_diversity'] = result_df.apply(
            lambda row: self._calculate_engagement_entropy(
                row['view_count'], row['addtocart_count'], row['transaction_count']
            ) if total_events.loc[row.name] > 0 else 0.0,
            axis=1
        )
        
        # Combined engagement intensity (weighted average of intensity and diversity)
        result_df['engagement_intensity_score'] = (
            0.8 * result_df['engagement_intensity_score'] +
            0.2 * result_df['engagement_diversity']
        )
        
        logger.info(f"Calculated engagement intensity scores for {len(result_df)} items")
        return result_df[['itemid', 'engagement_intensity_score', 'weighted_engagement_raw', 
                        'engagement_intensity_raw', 'engagement_diversity']]
    
    def calculate_visitor_loyalty_metrics(self, df: pd.DataFrame, events_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Create visitor loyalty metrics (repeat engagement patterns)
        
        Args:
            df: DataFrame with behavioral metrics
            events_df: Optional raw events DataFrame for detailed loyalty analysis
            
        Returns:
            DataFrame with visitor loyalty metrics
        """
        logger.info("Calculating visitor loyalty metrics")
        
        # Validate input
        is_valid, errors = self.validate_input_data(df)
        if not is_valid:
            raise ValueError(f"Input validation failed: {errors}")
        
        result_df = df.copy()
        
        # Basic loyalty score based on events per visitor ratio
        result_df['events_per_visitor'] = np.where(
            result_df['unique_visitors'] > 0,
            (result_df['view_count'] + result_df['addtocart_count'] + result_df['transaction_count']) / result_df['unique_visitors'],
            0.0
        )
        
        # Repeat engagement rate (proxy based on conversion funnel)
        # Items with higher addtocart_rate suggest more repeat/engaged visitors
        result_df['repeat_engagement_rate'] = np.minimum(
            result_df['addtocart_rate'] * 2.0,  # Scale addtocart rate as proxy
            1.0  # Cap at 1.0
        )
        
        # Visitor loyalty score combining multiple factors
        # Higher conversion rates and events per visitor indicate more loyal visitors
        result_df['visitor_loyalty_score'] = (
            0.4 * np.minimum(result_df['events_per_visitor'] / 5.0, 1.0) +  # Normalize events per visitor
            0.3 * result_df['conversion_rate'] * 10.0 +  # Scale conversion rate
            0.3 * result_df['cart_conversion_rate']  # Cart conversion indicates commitment
        )
        
        # Cap loyalty score at 1.0
        result_df['visitor_loyalty_score'] = np.minimum(result_df['visitor_loyalty_score'], 1.0)
        
        # If detailed events data is provided, calculate more sophisticated loyalty metrics
        if events_df is not None:
            detailed_loyalty = self._calculate_detailed_loyalty_metrics(events_df)
            if not detailed_loyalty.empty:
                result_df = result_df.merge(detailed_loyalty, on='itemid', how='left', suffixes=('', '_detailed'))
                
                # Update loyalty score with detailed metrics
                result_df['visitor_loyalty_score'] = np.where(
                    result_df['detailed_loyalty_score'].notna(),
                    0.7 * result_df['visitor_loyalty_score'] + 0.3 * result_df['detailed_loyalty_score'],
                    result_df['visitor_loyalty_score']
                )
        
        logger.info(f"Calculated visitor loyalty metrics for {len(result_df)} items")
        return result_df[['itemid', 'visitor_loyalty_score', 'repeat_engagement_rate', 
                        'events_per_visitor']]
    
    def implement_performance_bucketing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Implement item performance bucketing based on behavioral metrics
        
        Args:
            df: DataFrame with behavioral metrics and composite features
            
        Returns:
            DataFrame with performance buckets
        """
        logger.info("Implementing performance bucketing based on behavioral metrics")
        
        result_df = df.copy()
        
        # Calculate performance buckets based on conversion rate
        conversion_thresholds = np.percentile(
            result_df['conversion_rate'], 
            [p * 100 for p in self.performance_thresholds['conversion_rate']]
        )
        
        result_df['performance_bucket'] = pd.cut(
            result_df['conversion_rate'],
            bins=[-np.inf] + list(conversion_thresholds) + [np.inf],
            labels=[PerformanceBucket.LOW.value, PerformanceBucket.MEDIUM.value, 
                   PerformanceBucket.HIGH.value, PerformanceBucket.PREMIUM.value]
        )
        
        # Calculate engagement buckets based on engagement intensity
        if 'engagement_intensity_score' in result_df.columns:
            engagement_thresholds = np.percentile(
                result_df['engagement_intensity_score'],
                [p * 100 for p in self.performance_thresholds['engagement_intensity']]
            )
            
            result_df['engagement_bucket'] = pd.cut(
                result_df['engagement_intensity_score'],
                bins=[-np.inf] + list(engagement_thresholds) + [np.inf],
                labels=[EngagementBucket.MINIMAL.value, EngagementBucket.MODERATE.value,
                       EngagementBucket.HIGH.value, EngagementBucket.EXCEPTIONAL.value]
            )
        
        # Calculate loyalty buckets based on visitor loyalty score
        if 'visitor_loyalty_score' in result_df.columns:
            loyalty_thresholds = np.percentile(
                result_df['visitor_loyalty_score'],
                [p * 100 for p in self.performance_thresholds['visitor_loyalty']]
            )
            
            result_df['loyalty_bucket'] = pd.cut(
                result_df['visitor_loyalty_score'],
                bins=[-np.inf] + list(loyalty_thresholds) + [np.inf],
                labels=[LoyaltyBucket.TRANSIENT.value, LoyaltyBucket.CASUAL.value,
                       LoyaltyBucket.LOYAL.value, LoyaltyBucket.DEVOTED.value]
            )
        
        # Create composite performance score
        result_df['composite_score'] = self._calculate_composite_score(result_df)
        
        logger.info(f"Implemented performance bucketing for {len(result_df)} items")
        return result_df
    
    def add_feature_validation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add feature validation and consistency checks
        
        Args:
            df: DataFrame with composite features
            
        Returns:
            DataFrame with validation status
        """
        logger.info("Adding feature validation and consistency checks")
        
        result_df = df.copy()
        validation_results = []
        
        for _, row in result_df.iterrows():
            validation_status = self._validate_item_features(row)
            validation_results.append(validation_status)
        
        result_df['feature_validation_status'] = validation_results
        
        # Add validation summary statistics
        validation_summary = pd.Series(validation_results).value_counts()
        logger.info(f"Feature validation summary: {validation_summary.to_dict()}")
        
        return result_df
    
    def build_all_composite_features(self, df: pd.DataFrame, events_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Build all composite behavioral features
        
        Args:
            df: DataFrame with behavioral metrics
            events_df: Optional raw events DataFrame for detailed analysis
            
        Returns:
            DataFrame with all composite features
        """
        logger.info("Building all composite behavioral features")
        
        # Start with input data
        result_df = df.copy()
        
        # Calculate engagement intensity scores
        engagement_features = self.calculate_engagement_intensity_scores(df)
        result_df = result_df.merge(engagement_features, on='itemid', how='left')
        
        # Calculate visitor loyalty metrics
        loyalty_features = self.calculate_visitor_loyalty_metrics(df, events_df)
        result_df = result_df.merge(loyalty_features, on='itemid', how='left')
        
        # Implement performance bucketing
        result_df = self.implement_performance_bucketing(result_df)
        
        # Add feature validation
        result_df = self.add_feature_validation(result_df)
        
        logger.info(f"Built all composite features for {len(result_df)} items")
        return result_df
    
    def _calculate_engagement_entropy(self, views: int, carts: int, transactions: int) -> float:
        """Calculate Shannon entropy for engagement diversity"""
        total = views + carts + transactions
        if total == 0:
            return 0.0
        
        # Calculate probabilities
        probs = [count / total for count in [views, carts, transactions] if count > 0]
        
        # Calculate entropy
        entropy = -sum(p * np.log2(p) for p in probs)
        
        # Normalize to 0-1 scale (max entropy for 3 categories is log2(3))
        max_entropy = np.log2(3)
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    def _calculate_detailed_loyalty_metrics(self, events_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate detailed loyalty metrics from raw events data"""
        if events_df.empty:
            return pd.DataFrame()
        
        logger.info("Calculating detailed loyalty metrics from events data")
        
        # Group by item and visitor to analyze repeat behavior
        visitor_item_patterns = events_df.groupby(['itemid', 'visitorid']).agg({
            'event': 'count',
            'timestamp': ['min', 'max']
        }).reset_index()
        
        visitor_item_patterns.columns = ['itemid', 'visitorid', 'event_count', 'first_timestamp', 'last_timestamp']
        
        # Calculate engagement duration per visitor-item pair
        visitor_item_patterns['engagement_duration_ms'] = (
            visitor_item_patterns['last_timestamp'] - visitor_item_patterns['first_timestamp']
        )
        
        # Aggregate by item
        loyalty_metrics = []
        
        for itemid, item_group in visitor_item_patterns.groupby('itemid'):
            total_visitors = len(item_group)
            repeat_visitors = (item_group['event_count'] > 1).sum()
            highly_engaged_visitors = (item_group['event_count'] >= 5).sum()
            
            # Calculate detailed loyalty score
            repeat_rate = repeat_visitors / total_visitors if total_visitors > 0 else 0
            high_engagement_rate = highly_engaged_visitors / total_visitors if total_visitors > 0 else 0
            avg_events_per_visitor = item_group['event_count'].mean()
            avg_engagement_duration = item_group['engagement_duration_ms'].mean() / 1000  # Convert to seconds
            
            detailed_loyalty_score = (
                0.4 * repeat_rate +
                0.3 * high_engagement_rate +
                0.2 * min(avg_events_per_visitor / 10.0, 1.0) +  # Normalize events per visitor
                0.1 * min(avg_engagement_duration / 3600.0, 1.0)  # Normalize duration (max 1 hour)
            )
            
            loyalty_metrics.append({
                'itemid': itemid,
                'detailed_loyalty_score': detailed_loyalty_score,
                'repeat_visitor_rate': repeat_rate,
                'high_engagement_visitor_rate': high_engagement_rate,
                'avg_events_per_visitor_detailed': avg_events_per_visitor,
                'avg_engagement_duration_seconds': avg_engagement_duration
            })
        
        return pd.DataFrame(loyalty_metrics)
    
    def _calculate_composite_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate composite performance score"""
        # Weights for different components
        weights = {
            'conversion_rate': 0.4,
            'engagement_intensity_score': 0.3,
            'visitor_loyalty_score': 0.3
        }
        
        composite_score = pd.Series(0.0, index=df.index)
        
        for feature, weight in weights.items():
            if feature in df.columns:
                # Normalize feature to 0-1 scale if needed
                normalized_feature = df[feature]
                if feature == 'conversion_rate':
                    # Scale conversion rate (typically small values)
                    normalized_feature = np.minimum(normalized_feature * 10.0, 1.0)
                
                composite_score += weight * normalized_feature
        
        return composite_score
    
    def _validate_item_features(self, row: pd.Series) -> str:
        """Validate features for a single item"""
        issues = []
        
        # Check for missing values in key features
        key_features = ['engagement_intensity_score', 'visitor_loyalty_score', 'conversion_rate']
        for feature in key_features:
            if feature in row and (pd.isna(row[feature]) or row[feature] < 0):
                issues.append(f"invalid_{feature}")
        
        # Check for logical inconsistencies
        if 'unique_visitors' in row and 'view_count' in row:
            if row['unique_visitors'] > row['view_count']:
                issues.append("visitors_exceed_views")
        
        if 'addtocart_count' in row and 'transaction_count' in row and 'view_count' in row:
            if row['addtocart_count'] > row['view_count'] * 2:  # Allow some flexibility
                issues.append("excessive_addtocarts")
            if row['transaction_count'] > row['view_count']:
                issues.append("excessive_transactions")
        
        # Check for extreme values
        if 'engagement_intensity_score' in row and row['engagement_intensity_score'] > 1.1:
            issues.append("extreme_engagement")
        
        if 'visitor_loyalty_score' in row and row['visitor_loyalty_score'] > 1.1:
            issues.append("extreme_loyalty")
        
        # Return validation status
        if not issues:
            return "valid"
        elif len(issues) == 1:
            return f"warning_{issues[0]}"
        else:
            return f"error_multiple_issues"


# Convenience functions
def build_composite_features(behavioral_metrics_df: pd.DataFrame, 
                           events_df: Optional[pd.DataFrame] = None,
                           engagement_weights: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    """
    Convenience function to build all composite behavioral features
    
    Args:
        behavioral_metrics_df: DataFrame with behavioral metrics
        events_df: Optional raw events DataFrame for detailed analysis
        engagement_weights: Optional custom weights for engagement scoring
        
    Returns:
        DataFrame with composite features
    """
    builder = CompositeFeatureBuilder(engagement_weights=engagement_weights)
    return builder.build_all_composite_features(behavioral_metrics_df, events_df)


def get_feature_distribution_summary(composite_features_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get distribution summary of composite features
    
    Args:
        composite_features_df: DataFrame with composite features
        
    Returns:
        Dictionary with distribution summaries
    """
    if composite_features_df.empty:
        return {"error": "Features DataFrame is empty"}
    
    summary = {}
    
    # Numeric feature distributions
    numeric_features = [
        'engagement_intensity_score', 'visitor_loyalty_score', 'repeat_engagement_rate', 'composite_score'
    ]
    
    for feature in numeric_features:
        if feature in composite_features_df.columns:
            summary[feature] = {
                "mean": float(composite_features_df[feature].mean()),
                "median": float(composite_features_df[feature].median()),
                "std": float(composite_features_df[feature].std()),
                "min": float(composite_features_df[feature].min()),
                "max": float(composite_features_df[feature].max()),
                "percentiles": {
                    "25th": float(composite_features_df[feature].quantile(0.25)),
                    "75th": float(composite_features_df[feature].quantile(0.75)),
                    "95th": float(composite_features_df[feature].quantile(0.95))
                }
            }
    
    # Categorical feature distributions
    categorical_features = ['performance_bucket', 'engagement_bucket', 'loyalty_bucket', 'feature_validation_status']
    
    for feature in categorical_features:
        if feature in composite_features_df.columns:
            value_counts = composite_features_df[feature].value_counts()
            summary[feature] = {
                "distribution": value_counts.to_dict(),
                "most_common": value_counts.index[0] if len(value_counts) > 0 else None,
                "unique_values": len(value_counts)
            }
    
    return summary


if __name__ == "__main__":
    # Example usage
    try:
        from behavioral_metrics import calculate_behavioral_metrics
        from csv_loader import load_events_csv
        
        # Load events data
        events_df = load_events_csv("events.csv")
        print(f"Loaded {len(events_df)} events")
        
        # Calculate behavioral metrics
        behavioral_metrics = calculate_behavioral_metrics(events_df)
        print(f"Calculated behavioral metrics for {len(behavioral_metrics)} items")
        
        # Build composite features
        composite_features = build_composite_features(behavioral_metrics, events_df)
        print(f"Built composite features for {len(composite_features)} items")
        
        # Get feature summary
        summary = get_feature_distribution_summary(composite_features)
        print("\nComposite Features Summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        
        # Show sample of composite features
        print(f"\nSample composite features:")
        sample_cols = ['itemid', 'engagement_intensity_score', 'visitor_loyalty_score', 
                      'performance_bucket', 'composite_score', 'feature_validation_status']
        available_cols = [col for col in sample_cols if col in composite_features.columns]
        print(composite_features[available_cols].head(10))
        
    except FileNotFoundError:
        print("events.csv file not found. Please ensure the file exists in the current directory.")
    except Exception as e:
        print(f"Error: {str(e)}")