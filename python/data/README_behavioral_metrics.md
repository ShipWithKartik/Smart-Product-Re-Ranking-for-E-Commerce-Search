# Behavioral Metrics Module

This module provides functionality to calculate behavioral metrics from e-commerce event data for the Smart Product Re-Ranking System.

## Overview

The behavioral metrics module processes event data (views, addtocart, transactions) to calculate key performance indicators for products, including:

- **View counts**: Number of times each product was viewed
- **Addtocart rates**: Ratio of addtocart events to view events per product
- **Conversion rates**: Ratio of transaction events to view events per product  
- **Cart conversion rates**: Ratio of transaction events to addtocart events per product

## Key Components

### BehavioralMetricsCalculator

Main class for calculating behavioral metrics from event data.

```python
from data.behavioral_metrics import BehavioralMetricsCalculator

calculator = BehavioralMetricsCalculator()
metrics = calculator.calculate_all_behavioral_metrics(events_df)
```

### Key Methods

- `calculate_view_metrics(df)`: Calculate view counts and unique visitors per item
- `calculate_addtocart_metrics(df)`: Calculate addtocart counts per item
- `calculate_transaction_metrics(df)`: Calculate transaction counts per item
- `calculate_conversion_rates(...)`: Calculate all conversion rates
- `calculate_all_behavioral_metrics(df)`: Calculate comprehensive metrics
- `get_metrics_summary(metrics_df)`: Get summary statistics

### BehavioralMetricsPipeline

Complete pipeline for loading events and calculating metrics.

```python
from data.behavioral_pipeline import run_behavioral_metrics_pipeline

results = run_behavioral_metrics_pipeline("events.csv")
```

## Data Requirements

### Input Data Format (events.csv)

The module expects event data with the following columns:

- `timestamp`: Unix timestamp (milliseconds)
- `visitorid`: Unique visitor identifier
- `event`: Event type ('view', 'addtocart', 'transaction')
- `itemid`: Unique product identifier
- `transactionid`: Transaction ID (optional, for transaction events)

### Example Data

```csv
timestamp,visitorid,event,itemid,transactionid
1433221332117,257597,view,355908,
1433224214164,992329,addtocart,248676,
1433221999827,111016,transaction,318965,T123
```

## Output Metrics

The module generates the following metrics for each product:

| Metric | Description | Formula |
|--------|-------------|---------|
| `view_count` | Total number of views | Count of 'view' events |
| `addtocart_count` | Total number of addtocart events | Count of 'addtocart' events |
| `transaction_count` | Total number of transactions | Count of 'transaction' events |
| `unique_visitors` | Number of unique visitors | Unique count of visitorid |
| `addtocart_rate` | Addtocart conversion rate | addtocart_count / view_count |
| `conversion_rate` | Purchase conversion rate | transaction_count / view_count |
| `cart_conversion_rate` | Cart-to-purchase rate | transaction_count / addtocart_count |
| `popularity_score` | Normalized popularity | view_count / max_view_count |

## Usage Examples

### Basic Usage

```python
from data.csv_loader import load_events_csv
from data.behavioral_metrics import calculate_behavioral_metrics

# Load events data
events_df = load_events_csv("events.csv")

# Calculate behavioral metrics
metrics_df = calculate_behavioral_metrics(events_df)

# Display top performing items
top_items = metrics_df.nlargest(10, 'conversion_rate')
print(top_items[['itemid', 'view_count', 'conversion_rate']])
```

### Pipeline Usage

```python
from data.behavioral_pipeline import BehavioralMetricsPipeline

# Initialize pipeline
pipeline = BehavioralMetricsPipeline("events.csv")

# Run complete pipeline
results = pipeline.run_full_pipeline(
    include_temporal=False,
    save_results=True,
    output_dir="output"
)

# Access results
events_df = results['events_df']
metrics_df = results['metrics_df']
summary = results['summary']
```

### Individual Metric Calculations

```python
from data.behavioral_metrics import BehavioralMetricsCalculator

calculator = BehavioralMetricsCalculator()

# Calculate individual metrics
view_metrics = calculator.calculate_view_metrics(events_df)
addtocart_metrics = calculator.calculate_addtocart_metrics(events_df)
transaction_metrics = calculator.calculate_transaction_metrics(events_df)

# Combine metrics
all_metrics = calculator.calculate_conversion_rates(
    view_metrics, addtocart_metrics, transaction_metrics
)
```

## Performance Considerations

- **Large Files**: Use chunked processing for files > 100MB
- **Memory Usage**: Monitor memory consumption with large datasets
- **Processing Time**: Expect ~1-2 seconds per 100K events

## Error Handling

The module includes comprehensive error handling for:

- Missing required columns
- Invalid data types
- Empty datasets
- Division by zero in rate calculations

## Integration

This module integrates with:

- `csv_loader.py`: For loading and validating event data
- `feature_engineering.py`: For creating ML-ready features
- `model_training.py`: For training ranking models

## Requirements Mapping

This module implements the following requirements:

- **Requirement 1.1**: Calculate CTR as clicks divided by impressions (adapted for e-commerce)
- **Requirement 1.2**: Calculate CVR as purchases divided by clicks (adapted for e-commerce)
- **Requirement 1.3**: Aggregate cart-to-purchase ratios for each product
- **Requirement 1.5**: Store aggregated metrics in ML-suitable format

## Testing

Run the test suite to validate functionality:

```python
python examples/test_behavioral_metrics.py
```

The test creates sample data and validates all metric calculations.