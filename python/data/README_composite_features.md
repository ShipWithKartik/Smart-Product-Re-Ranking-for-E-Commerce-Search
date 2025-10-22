# Composite Behavioral Features Module

This module implements composite behavioral features for the Smart Product Re-Ranking System, combining multiple behavioral signals to create enhanced features for machine learning models.

## Overview

The composite features module builds upon basic behavioral metrics (views, addtocarts, transactions) to create sophisticated features that capture:

- **Engagement Intensity**: How intensively users engage with products
- **Visitor Loyalty**: Patterns of repeat engagement and user commitment
- **Performance Bucketing**: Categorical groupings based on behavioral performance
- **Feature Validation**: Consistency checks and data quality validation

## Key Components

### CompositeFeatureBuilder

Main class for building composite behavioral features.

```python
from data.composite_features import CompositeFeatureBuilder

builder = CompositeFeatureBuilder()
composite_features = builder.build_all_composite_features(behavioral_metrics_df, events_df)
```

### Features Generated

#### 1. Engagement Intensity Score (0-1)
Combines views, addtocarts, and transactions with weighted scoring:
- Views: weight 1.0
- Addtocarts: weight 3.0  
- Transactions: weight 10.0

Normalized by unique visitors and includes engagement diversity (Shannon entropy).

#### 2. Visitor Loyalty Score (0-1)
Measures repeat engagement patterns:
- Events per visitor ratio
- Conversion funnel progression
- Repeat visitor behavior (when events data available)

#### 3. Performance Buckets
Categorical groupings based on conversion rates:
- **LOW**: Bottom 25th percentile
- **MEDIUM**: 25th-50th percentile
- **HIGH**: 50th-75th percentile
- **PREMIUM**: Top 25th percentile

#### 4. Engagement Buckets
Based on engagement intensity scores:
- **MINIMAL**: Bottom 33rd percentile
- **MODERATE**: 33rd-66th percentile
- **HIGH**: 66th-90th percentile
- **EXCEPTIONAL**: Top 10th percentile

#### 5. Loyalty Buckets
Based on visitor loyalty scores:
- **TRANSIENT**: Bottom 20th percentile
- **CASUAL**: 20th-50th percentile
- **LOYAL**: 50th-80th percentile
- **DEVOTED**: Top 20th percentile

#### 6. Composite Score (0-1)
Weighted combination of key metrics:
- Conversion rate: 40%
- Engagement intensity: 30%
- Visitor loyalty: 30%

#### 7. Feature Validation Status
Quality checks for data consistency:
- **valid**: All checks passed
- **warning_[issue]**: Minor issues detected
- **error_multiple_issues**: Multiple validation failures

## Usage Examples

### Basic Usage

```python
from data.composite_features import build_composite_features
from data.behavioral_metrics import calculate_behavioral_metrics
from data.csv_loader import load_events_csv

# Load and process data
events_df = load_events_csv("events.csv")
behavioral_metrics = calculate_behavioral_metrics(events_df)

# Build composite features
composite_features = build_composite_features(behavioral_metrics, events_df)
```

### Advanced Usage with Custom Weights

```python
from data.composite_features import CompositeFeatureBuilder

# Custom engagement weights
custom_weights = {
    'view': 1.0,
    'addtocart': 5.0,  # Higher weight for addtocarts
    'transaction': 15.0  # Higher weight for transactions
}

builder = CompositeFeatureBuilder(engagement_weights=custom_weights)
composite_features = builder.build_all_composite_features(behavioral_metrics, events_df)
```

### Feature Analysis

```python
from data.composite_features import get_feature_distribution_summary

# Get distribution summary
summary = get_feature_distribution_summary(composite_features)

# Print key statistics
for feature in ['engagement_intensity_score', 'visitor_loyalty_score']:
    if feature in summary:
        stats = summary[feature]
        print(f"{feature}: mean={stats['mean']:.4f}, std={stats['std']:.4f}")
```

## Input Requirements

### Required Columns in Behavioral Metrics DataFrame
- `itemid`: Unique item identifier
- `view_count`: Number of view events
- `addtocart_count`: Number of addtocart events  
- `transaction_count`: Number of transaction events
- `unique_visitors`: Number of unique visitors
- `addtocart_rate`: Addtocart rate (addtocart/view)
- `conversion_rate`: Conversion rate (transaction/view)
- `cart_conversion_rate`: Cart conversion rate (transaction/addtocart)

### Optional Events DataFrame
For enhanced loyalty metrics, provide raw events data with:
- `timestamp`: Event timestamp
- `visitorid`: Visitor identifier
- `event`: Event type ('view', 'addtocart', 'transaction')
- `itemid`: Item identifier

## Output Features

The module generates a comprehensive DataFrame with:

### Numeric Features
- `engagement_intensity_score`: Weighted engagement per visitor (0-1)
- `visitor_loyalty_score`: Repeat engagement patterns (0-1)
- `repeat_engagement_rate`: Proxy for repeat visitors (0-1)
- `composite_score`: Combined performance score (0-1)
- `events_per_visitor`: Average events per unique visitor
- `weighted_engagement_raw`: Raw weighted engagement score
- `engagement_diversity`: Shannon entropy of engagement types (0-1)

### Categorical Features
- `performance_bucket`: Performance category (low/medium/high/premium)
- `engagement_bucket`: Engagement category (minimal/moderate/high/exceptional)
- `loyalty_bucket`: Loyalty category (transient/casual/loyal/devoted)
- `feature_validation_status`: Data quality status

## Validation and Quality Checks

The module includes comprehensive validation:

### Input Validation
- Required column presence
- Data type consistency
- Logical relationships (e.g., unique_visitors ≤ view_count)
- Negative value detection

### Feature Validation
- Range checks (scores should be 0-1)
- Consistency checks across related features
- Extreme value detection
- Missing value handling

### Quality Warnings
- Unusual conversion rates (>10x normal)
- Logical inconsistencies in event counts
- Extreme engagement or loyalty scores

## Performance Considerations

- **Memory Usage**: Processes large datasets efficiently using pandas operations
- **Computation Time**: O(n) complexity for most operations where n = number of items
- **Detailed Loyalty Metrics**: O(m) additional complexity where m = number of events (optional)

## Integration with ML Pipeline

The composite features are designed for direct use in machine learning models:

```python
# Select features for ML model
ml_features = composite_features[[
    'engagement_intensity_score',
    'visitor_loyalty_score', 
    'composite_score',
    'events_per_visitor'
]]

# Or use categorical features with encoding
categorical_features = composite_features[[
    'performance_bucket',
    'engagement_bucket', 
    'loyalty_bucket'
]]
```

## Error Handling

The module provides robust error handling:
- Input validation with detailed error messages
- Graceful handling of missing optional data
- Warning system for data quality issues
- Fallback calculations when detailed data unavailable

## Testing

Run the test suite to validate functionality:

```bash
python python/examples/test_composite_features.py
```

Run the example to see the module in action:

```bash
python python/examples/composite_features_example.py
```