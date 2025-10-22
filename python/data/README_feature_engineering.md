# Feature Engineering Module

This module implements temporal and engagement feature extraction for the Smart Product Re-Ranking System, as specified in task 3.1.

## Overview

The feature engineering module extracts comprehensive temporal and engagement features from event data to support machine learning models for product re-ranking. It implements the requirements from the design specification:

- Extract time-based patterns from timestamp data
- Calculate unique visitor counts per item
- Implement popularity scoring based on total engagement
- Create average time-to-cart and time-to-purchase metrics

## Components

### 1. TemporalFeatureExtractor

Extracts temporal features from event timestamp data.

#### Key Features:
- **Time-based patterns**: Hour-of-day and day-of-week activity patterns
- **Time-to-action metrics**: Average and median time from view to cart/purchase
- **Activity patterns**: Peak activity hours, activity span, daily event patterns
- **Session metrics**: Session depth and conversion rates

#### Methods:
- `extract_time_based_patterns(df)`: Extracts temporal patterns from timestamps
- `calculate_time_to_action_metrics(df)`: Calculates time-to-cart and time-to-purchase metrics

### 2. EngagementFeatureExtractor

Extracts engagement features from user interaction data.

#### Key Features:
- **Unique visitor counts**: Number of unique visitors per item
- **Popularity scoring**: Weighted engagement scores based on event types
- **Visitor loyalty**: Repeat visitor rates and engagement intensity
- **Engagement metrics**: Events per visitor, session patterns

#### Methods:
- `calculate_unique_visitor_counts(df)`: Calculates unique visitors per item
- `calculate_popularity_scoring(df)`: Implements popularity scoring algorithm
- `calculate_visitor_loyalty_metrics(df)`: Calculates visitor loyalty patterns

### 3. FeatureEngineeringPipeline

Complete pipeline that combines all feature extraction components.

#### Methods:
- `extract_all_features(df)`: Extracts all temporal and engagement features
- `get_feature_summary(features_df)`: Provides summary statistics of extracted features

## Extracted Features

### Temporal Features
1. **peak_activity_hour**: Hour of day with highest activity
2. **activity_span_hours**: Time span between first and last event
3. **avg_daily_events**: Average number of events per day
4. **total_days_active**: Number of days with activity
5. **hour_pattern_entropy**: Entropy of hour-of-day distribution
6. **dow_pattern_entropy**: Entropy of day-of-week distribution
7. **avg_time_to_cart**: Average time from view to addtocart (seconds)
8. **avg_time_to_purchase**: Average time from view to transaction (seconds)
9. **median_time_to_cart**: Median time from view to addtocart (seconds)
10. **median_time_to_purchase**: Median time from view to transaction (seconds)
11. **avg_session_depth**: Average number of events per session
12. **session_cart_rate**: Proportion of sessions resulting in addtocart
13. **session_purchase_rate**: Proportion of sessions resulting in purchase

### Engagement Features
1. **unique_visitors**: Number of unique visitors for the item
2. **total_events**: Total number of events for the item
3. **events_per_visitor**: Average events per unique visitor
4. **view_count**: Number of view events
5. **addtocart_count**: Number of addtocart events
6. **transaction_count**: Number of transaction events
7. **weighted_engagement_score**: Weighted sum of events (view=1, cart=3, transaction=10)
8. **engagement_intensity**: Total events per unique visitor
9. **repeat_visitors**: Number of visitors with multiple events
10. **visitor_loyalty_score**: Proportion of repeat visitors
11. **popularity_score**: Normalized popularity based on weighted engagement
12. **visitor_popularity_score**: Normalized popularity based on unique visitors
13. **combined_popularity_score**: Weighted combination of popularity scores
14. **repeat_visitor_rate**: Rate of visitors who return multiple times
15. **high_engagement_rate**: Rate of highly engaged visitors (5+ events)
16. **avg_events_per_visitor**: Average number of events per visitor
17. **avg_engagement_duration_seconds**: Average engagement duration per visitor

## Usage Examples

### Basic Usage

```python
from data.feature_engineering import extract_temporal_and_engagement_features
from data.csv_loader import load_events_csv

# Load events data
events_df = load_events_csv("events.csv")

# Extract all features
features = extract_temporal_and_engagement_features(events_df)

print(f"Extracted {len(features.columns) - 1} features for {len(features)} items")
```

### Advanced Usage

```python
from data.feature_engineering import (
    TemporalFeatureExtractor,
    EngagementFeatureExtractor,
    FeatureEngineeringPipeline
)

# Initialize extractors
temporal_extractor = TemporalFeatureExtractor()
engagement_extractor = EngagementFeatureExtractor()

# Extract specific feature types
temporal_patterns = temporal_extractor.extract_time_based_patterns(events_df)
time_metrics = temporal_extractor.calculate_time_to_action_metrics(events_df)
popularity_scores = engagement_extractor.calculate_popularity_scoring(events_df)

# Use complete pipeline
pipeline = FeatureEngineeringPipeline()
all_features = pipeline.extract_all_features(events_df)
summary = pipeline.get_feature_summary(all_features)
```

### Finding Top Items

```python
from data.feature_engineering import get_top_items_by_feature

# Get top items by different metrics
top_popular = get_top_items_by_feature(features, 'popularity_score', 10)
top_engaging = get_top_items_by_feature(features, 'engagement_intensity', 10)
top_loyal = get_top_items_by_feature(features, 'visitor_loyalty_score', 10)
```

## Data Requirements

### Input Data Format
The module expects event data with the following columns:
- `timestamp`: Event timestamp in milliseconds
- `visitorid`: Unique visitor identifier
- `event`: Event type ('view', 'addtocart', 'transaction')
- `itemid`: Unique item identifier
- `transactionid`: Transaction identifier (optional)

### Data Validation
The module includes comprehensive data validation:
- Checks for required columns
- Validates data types and ranges
- Handles missing values appropriately
- Filters invalid events

## Performance Considerations

### Memory Usage
- Uses efficient pandas operations for large datasets
- Implements chunked processing for very large files
- Optimizes memory usage with appropriate data types

### Processing Time
- Temporal features: O(n log n) where n is number of events
- Engagement features: O(n) where n is number of events
- Complete pipeline: O(n log n) overall complexity

### Scalability
- Designed to handle millions of events
- Uses vectorized operations where possible
- Implements efficient groupby operations

## Integration with Existing Systems

### Behavioral Metrics Integration
The feature engineering module extends the existing `behavioral_metrics.py` module:
- Provides more comprehensive temporal features
- Adds advanced engagement metrics
- Maintains compatibility with existing interfaces

### Machine Learning Pipeline
Features are designed for direct use in ML models:
- All features are numeric and normalized where appropriate
- Missing values are handled consistently
- Feature scaling recommendations provided

## Testing

### Unit Tests
Run the test suite to validate functionality:

```bash
python python/examples/test_temporal_engagement.py
```

### Integration Tests
Test with real data:

```bash
python python/examples/feature_engineering_example.py
```

## Requirements Mapping

This implementation addresses the following requirements from the specification:

- **Requirement 2.1**: Extract time-based patterns from timestamp data
  - Implemented in `TemporalFeatureExtractor.extract_time_based_patterns()`
  
- **Requirement 2.2**: Calculate unique visitor counts per item
  - Implemented in `EngagementFeatureExtractor.calculate_unique_visitor_counts()`
  
- **Requirement 2.3**: Implement popularity scoring based on total engagement
  - Implemented in `EngagementFeatureExtractor.calculate_popularity_scoring()`
  
- **Additional**: Create average time-to-cart and time-to-purchase metrics
  - Implemented in `TemporalFeatureExtractor.calculate_time_to_action_metrics()`

## Future Enhancements

### Planned Features
1. **Seasonal patterns**: Monthly and yearly activity patterns
2. **Cross-item features**: Item similarity based on visitor overlap
3. **Real-time features**: Streaming feature calculation
4. **Advanced temporal**: Time series decomposition and trend analysis

### Performance Optimizations
1. **Caching**: Feature caching for repeated calculations
2. **Parallel processing**: Multi-threaded feature extraction
3. **Incremental updates**: Update features with new data only

## Troubleshooting

### Common Issues

1. **Memory errors with large datasets**
   - Use chunked processing in `csv_loader.py`
   - Reduce feature complexity for initial testing

2. **Missing temporal features**
   - Ensure timestamp data is in milliseconds
   - Check for sufficient time span in data

3. **Zero engagement scores**
   - Verify event type values ('view', 'addtocart', 'transaction')
   - Check for data filtering issues

### Debug Mode
Enable debug logging for detailed processing information:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Dependencies

- pandas >= 1.3.0
- numpy >= 1.21.0
- python >= 3.8

## License

This module is part of the Smart Product Re-Ranking System and follows the same licensing terms.