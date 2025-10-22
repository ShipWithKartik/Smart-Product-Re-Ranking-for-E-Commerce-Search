# CSV Data Loader Documentation

## Overview

The CSV Data Loader module provides robust loading and validation functionality for the events.csv file used in the Smart Product Re-Ranking System. It handles data type validation, event type classification, and data quality checks.

## Features

### Data Loading
- **Efficient CSV Reading**: Proper data type handling for large files
- **Chunked Processing**: Memory-efficient loading for large datasets
- **Error Handling**: Comprehensive error handling and logging

### Data Validation
- **Required Columns**: Validates presence of timestamp, visitorid, event, itemid
- **Data Types**: Ensures proper integer types for IDs and timestamps
- **Event Types**: Validates event types (view, addtocart, transaction)
- **Data Quality**: Checks for invalid values, duplicates, and missing data

### Event Processing
- **Event Classification**: Separates events by type for targeted processing
- **Event Filtering**: Filter datasets by specific event types
- **Data Summary**: Comprehensive statistics and data profiling

## Usage

### Basic Loading
```python
from python.data.csv_loader import load_events_csv

# Load and validate events
events_df = load_events_csv("events.csv", validate=True, filter_invalid=True)
```

### Advanced Usage
```python
from python.data.csv_loader import CSVDataLoader, EventType

# Create loader instance
loader = CSVDataLoader("events.csv")

# Load with validation
events_df = loader.load_events(validate=True, filter_invalid=True)

# Get data summary
summary = loader.get_data_summary(events_df)

# Classify events by type
classified = loader.classify_events(events_df)
view_events = classified[EventType.VIEW.value]
cart_events = classified[EventType.ADDTOCART.value]
purchase_events = classified[EventType.TRANSACTION.value]

# Filter specific event types
conversion_events = loader.filter_by_event_type(events_df, ['addtocart', 'transaction'])
```

## Data Structure

The CSV loader expects events.csv with the following structure:

| Column | Type | Required | Description |
|--------|------|----------|-------------|
| timestamp | int64 | Yes | Unix timestamp in milliseconds |
| visitorid | int64 | Yes | Unique visitor identifier |
| event | str | Yes | Event type: 'view', 'addtocart', 'transaction' |
| itemid | int64 | Yes | Unique item/product identifier |
| transactionid | str | No | Transaction ID (for purchase events) |

## Validation Rules

- **Timestamps**: Must be positive integers
- **Visitor IDs**: Must be positive integers
- **Item IDs**: Must be positive integers  
- **Event Types**: Must be one of: view, addtocart, transaction
- **Missing Values**: Handled gracefully with filtering options

## Integration

The CSV loader integrates with existing components:

- **RetailrocketDataExtractor**: Can use CSV loader for initial data loading
- **Feature Engineering**: Provides clean, validated data for feature calculation
- **Model Training**: Ensures data quality for machine learning pipelines

## Error Handling

- **File Validation**: Checks file existence and format
- **Data Validation**: Comprehensive validation with detailed error reporting
- **Memory Management**: Chunked processing for large files
- **Logging**: Structured logging for monitoring and debugging

## Performance

- **Chunked Loading**: Handles files larger than available memory
- **Data Type Optimization**: Efficient pandas data types
- **Validation Caching**: Reuses validation results where possible
- **Memory Monitoring**: Reports memory usage statistics