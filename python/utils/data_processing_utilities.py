"""
Data Processing Utilities
Common utilities for data processing, validation, and error handling
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
import json
import time
from datetime import datetime
import warnings
import psutil
import os


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataValidator:
    """Utilities for data validation and quality checks"""
    
    @staticmethod
    def validate_events_schema(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate events DataFrame schema and data quality
        
        Args:
            df: Events DataFrame to validate
            
        Returns:
            Dictionary with validation results
        """
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {}
        }
        
        # Required columns check
        required_cols = ['timestamp', 'visitorid', 'event', 'itemid']
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            validation_result['errors'].append(f"Missing required columns: {missing_cols}")
            validation_result['is_valid'] = False
        
        if df.empty:
            validation_result['errors'].append("DataFrame is empty")
            validation_result['is_valid'] = False
            return validation_result
        
        # Data type validation
        if 'timestamp' in df.columns:
            non_numeric_timestamps = df['timestamp'].apply(lambda x: not isinstance(x, (int, float, np.integer, np.floating)))
            if non_numeric_timestamps.any():
                validation_result['warnings'].append(f"Non-numeric timestamps found: {non_numeric_timestamps.sum()}")
        
        # Event type validation
        if 'event' in df.columns:
            valid_events = {'view', 'addtocart', 'transaction'}
            invalid_events = set(df['event'].unique()) - valid_events
            if invalid_events:
                validation_result['warnings'].append(f"Invalid event types found: {invalid_events}")
        
        # Null value check
        null_counts = df.isnull().sum()
        if null_counts.any():
            validation_result['warnings'].append(f"Null values found: {null_counts.to_dict()}")
        
        # Statistics
        validation_result['statistics'] = {
            'total_rows': len(df),
            'unique_visitors': df['visitorid'].nunique() if 'visitorid' in df.columns else 0,
            'unique_items': df['itemid'].nunique() if 'itemid' in df.columns else 0,
            'event_distribution': df['event'].value_counts().to_dict() if 'event' in df.columns else {},
            'date_range': {
                'min_timestamp': int(df['timestamp'].min()) if 'timestamp' in df.columns else None,
                'max_timestamp': int(df['timestamp'].max()) if 'timestamp' in df.columns else None
            }
        }
        
        return validation_result
    
    @staticmethod
    def validate_metrics_quality(metrics_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate behavioral metrics quality
        
        Args:
            metrics_df: Behavioral metrics DataFrame
            
        Returns:
            Dictionary with validation results
        """
        validation_result = {
            'is_valid': True,
            'warnings': [],
            'quality_scores': {}
        }
        
        # Check for reasonable conversion rates
        if 'conversion_rate' in metrics_df.columns:
            high_conversion = metrics_df['conversion_rate'] > 0.5  # >50% conversion
            if high_conversion.any():
                validation_result['warnings'].append(
                    f"Unusually high conversion rates found: {high_conversion.sum()} items"
                )
        
        # Check for items with no engagement
        if 'view_count' in metrics_df.columns:
            no_views = metrics_df['view_count'] == 0
            if no_views.any():
                validation_result['warnings'].append(
                    f"Items with no views found: {no_views.sum()}"
                )
        
        # Calculate quality scores
        if 'conversion_rate' in metrics_df.columns:
            validation_result['quality_scores']['conversion_rate_distribution'] = {
                'mean': float(metrics_df['conversion_rate'].mean()),
                'std': float(metrics_df['conversion_rate'].std()),
                'median': float(metrics_df['conversion_rate'].median())
            }
        
        return validation_result


class MemoryMonitor:
    """Monitor memory usage during data processing"""
    
    def __init__(self, warning_threshold_mb: int = 500, error_threshold_mb: int = 1000):
        """
        Initialize memory monitor
        
        Args:
            warning_threshold_mb: Memory usage threshold for warnings
            error_threshold_mb: Memory usage threshold for errors
        """
        self.warning_threshold = warning_threshold_mb * 1024 * 1024  # Convert to bytes
        self.error_threshold = error_threshold_mb * 1024 * 1024
        self.process = psutil.Process(os.getpid())
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics"""
        memory_info = self.process.memory_info()
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size
            'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
            'percent': self.process.memory_percent()
        }
    
    def check_memory_usage(self) -> Optional[str]:
        """
        Check current memory usage and return warning/error message if needed
        
        Returns:
            Warning or error message, or None if memory usage is acceptable
        """
        memory_info = self.process.memory_info()
        
        if memory_info.rss > self.error_threshold:
            return f"ERROR: High memory usage: {memory_info.rss / 1024 / 1024:.1f}MB"
        elif memory_info.rss > self.warning_threshold:
            return f"WARNING: Elevated memory usage: {memory_info.rss / 1024 / 1024:.1f}MB"
        
        return None
    
    def log_memory_usage(self, context: str = ""):
        """Log current memory usage"""
        usage = self.get_memory_usage()
        logger.info(f"Memory usage {context}: {usage['rss_mb']:.1f}MB RSS, {usage['percent']:.1f}%")


class ProcessingTimer:
    """Timer for tracking processing performance"""
    
    def __init__(self):
        """Initialize timer"""
        self.start_time = None
        self.end_time = None
        self.checkpoints = {}
    
    def start(self):
        """Start timing"""
        self.start_time = time.time()
        logger.info("Processing timer started")
    
    def checkpoint(self, name: str):
        """Record a checkpoint"""
        if self.start_time is None:
            logger.warning("Timer not started, cannot record checkpoint")
            return
        
        current_time = time.time()
        elapsed = current_time - self.start_time
        self.checkpoints[name] = elapsed
        logger.info(f"Checkpoint '{name}': {elapsed:.2f} seconds")
    
    def stop(self) -> float:
        """Stop timing and return total elapsed time"""
        if self.start_time is None:
            logger.warning("Timer not started")
            return 0.0
        
        self.end_time = time.time()
        total_time = self.end_time - self.start_time
        logger.info(f"Processing completed in {total_time:.2f} seconds")
        return total_time
    
    def get_summary(self) -> Dict[str, float]:
        """Get timing summary"""
        summary = {'checkpoints': self.checkpoints.copy()}
        
        if self.start_time and self.end_time:
            summary['total_time'] = self.end_time - self.start_time
        
        return summary


class DataCleaner:
    """Utilities for cleaning and preprocessing data"""
    
    @staticmethod
    def clean_events_data(df: pd.DataFrame, 
                         remove_duplicates: bool = True,
                         filter_invalid_timestamps: bool = True,
                         filter_invalid_ids: bool = True) -> pd.DataFrame:
        """
        Clean events data by removing invalid entries
        
        Args:
            df: Events DataFrame to clean
            remove_duplicates: Whether to remove duplicate rows
            filter_invalid_timestamps: Whether to filter invalid timestamps
            filter_invalid_ids: Whether to filter invalid IDs
            
        Returns:
            Cleaned DataFrame
        """
        logger.info(f"Cleaning events data: {len(df)} rows")
        cleaned_df = df.copy()
        
        # Remove duplicates
        if remove_duplicates:
            initial_count = len(cleaned_df)
            cleaned_df = cleaned_df.drop_duplicates()
            removed = initial_count - len(cleaned_df)
            if removed > 0:
                logger.info(f"Removed {removed} duplicate rows")
        
        # Filter invalid timestamps
        if filter_invalid_timestamps and 'timestamp' in cleaned_df.columns:
            initial_count = len(cleaned_df)
            cleaned_df = cleaned_df[
                cleaned_df['timestamp'].notna() & 
                (cleaned_df['timestamp'] > 0)
            ]
            removed = initial_count - len(cleaned_df)
            if removed > 0:
                logger.info(f"Removed {removed} rows with invalid timestamps")
        
        # Filter invalid IDs
        if filter_invalid_ids:
            initial_count = len(cleaned_df)
            
            if 'visitorid' in cleaned_df.columns:
                cleaned_df = cleaned_df[
                    cleaned_df['visitorid'].notna() & 
                    (cleaned_df['visitorid'] > 0)
                ]
            
            if 'itemid' in cleaned_df.columns:
                cleaned_df = cleaned_df[
                    cleaned_df['itemid'].notna() & 
                    (cleaned_df['itemid'] > 0)
                ]
            
            removed = initial_count - len(cleaned_df)
            if removed > 0:
                logger.info(f"Removed {removed} rows with invalid IDs")
        
        logger.info(f"Cleaning complete: {len(cleaned_df)} rows remaining")
        return cleaned_df
    
    @staticmethod
    def handle_outliers(df: pd.DataFrame, 
                       columns: List[str],
                       method: str = 'iqr',
                       threshold: float = 3.0) -> pd.DataFrame:
        """
        Handle outliers in specified columns
        
        Args:
            df: DataFrame to process
            columns: List of columns to check for outliers
            method: Method for outlier detection ('iqr', 'zscore', 'percentile')
            threshold: Threshold for outlier detection
            
        Returns:
            DataFrame with outliers handled
        """
        logger.info(f"Handling outliers in columns: {columns}")
        cleaned_df = df.copy()
        
        for col in columns:
            if col not in cleaned_df.columns:
                logger.warning(f"Column {col} not found, skipping")
                continue
            
            initial_count = len(cleaned_df)
            
            if method == 'iqr':
                Q1 = cleaned_df[col].quantile(0.25)
                Q3 = cleaned_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                cleaned_df = cleaned_df[
                    (cleaned_df[col] >= lower_bound) & 
                    (cleaned_df[col] <= upper_bound)
                ]
            
            elif method == 'zscore':
                z_scores = np.abs((cleaned_df[col] - cleaned_df[col].mean()) / cleaned_df[col].std())
                cleaned_df = cleaned_df[z_scores <= threshold]
            
            elif method == 'percentile':
                lower_percentile = (100 - threshold * 10) / 2
                upper_percentile = 100 - lower_percentile
                lower_bound = cleaned_df[col].quantile(lower_percentile / 100)
                upper_bound = cleaned_df[col].quantile(upper_percentile / 100)
                cleaned_df = cleaned_df[
                    (cleaned_df[col] >= lower_bound) & 
                    (cleaned_df[col] <= upper_bound)
                ]
            
            removed = initial_count - len(cleaned_df)
            if removed > 0:
                logger.info(f"Removed {removed} outliers from column {col}")
        
        return cleaned_df


class FileManager:
    """Utilities for file management and I/O operations"""
    
    @staticmethod
    def ensure_directory(path: Union[str, Path]) -> Path:
        """
        Ensure directory exists, create if it doesn't
        
        Args:
            path: Directory path
            
        Returns:
            Path object for the directory
        """
        dir_path = Path(path)
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path
    
    @staticmethod
    def save_dataframe(df: pd.DataFrame, 
                      file_path: Union[str, Path],
                      format: str = 'csv',
                      **kwargs) -> str:
        """
        Save DataFrame to file with specified format
        
        Args:
            df: DataFrame to save
            file_path: Output file path
            format: File format ('csv', 'parquet', 'json')
            **kwargs: Additional arguments for pandas save methods
            
        Returns:
            Path to saved file
        """
        file_path = Path(file_path)
        
        # Ensure directory exists
        FileManager.ensure_directory(file_path.parent)
        
        if format.lower() == 'csv':
            df.to_csv(file_path, index=False, **kwargs)
        elif format.lower() == 'parquet':
            df.to_parquet(file_path, index=False, **kwargs)
        elif format.lower() == 'json':
            df.to_json(file_path, orient='records', **kwargs)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Saved DataFrame to: {file_path}")
        return str(file_path)
    
    @staticmethod
    def save_json(data: Dict[str, Any], 
                  file_path: Union[str, Path],
                  indent: int = 2) -> str:
        """
        Save dictionary to JSON file
        
        Args:
            data: Dictionary to save
            file_path: Output file path
            indent: JSON indentation
            
        Returns:
            Path to saved file
        """
        file_path = Path(file_path)
        FileManager.ensure_directory(file_path.parent)
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=indent, default=str)
        
        logger.info(f"Saved JSON to: {file_path}")
        return str(file_path)


class ProgressTracker:
    """Track progress of long-running operations"""
    
    def __init__(self, total_items: int, description: str = "Processing"):
        """
        Initialize progress tracker
        
        Args:
            total_items: Total number of items to process
            description: Description of the operation
        """
        self.total_items = total_items
        self.description = description
        self.processed_items = 0
        self.start_time = time.time()
        self.last_update = self.start_time
    
    def update(self, increment: int = 1):
        """
        Update progress
        
        Args:
            increment: Number of items processed
        """
        self.processed_items += increment
        current_time = time.time()
        
        # Log progress every 10 seconds or at completion
        if (current_time - self.last_update > 10) or (self.processed_items >= self.total_items):
            percentage = (self.processed_items / self.total_items) * 100
            elapsed_time = current_time - self.start_time
            
            if self.processed_items > 0:
                estimated_total_time = elapsed_time * (self.total_items / self.processed_items)
                remaining_time = estimated_total_time - elapsed_time
                
                logger.info(
                    f"{self.description}: {self.processed_items:,}/{self.total_items:,} "
                    f"({percentage:.1f}%) - ETA: {remaining_time:.0f}s"
                )
            
            self.last_update = current_time
    
    def complete(self):
        """Mark processing as complete"""
        total_time = time.time() - self.start_time
        logger.info(
            f"{self.description} completed: {self.processed_items:,} items "
            f"in {total_time:.1f} seconds"
        )


# Convenience functions
def validate_and_clean_events(df: pd.DataFrame, 
                            strict_validation: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Convenience function to validate and clean events data
    
    Args:
        df: Events DataFrame
        strict_validation: Whether to use strict validation
        
    Returns:
        Tuple of (cleaned_df, validation_result)
    """
    # Validate
    validator = DataValidator()
    validation_result = validator.validate_events_schema(df)
    
    if not validation_result['is_valid'] and strict_validation:
        raise ValueError(f"Validation failed: {validation_result['errors']}")
    
    # Clean
    cleaner = DataCleaner()
    cleaned_df = cleaner.clean_events_data(df)
    
    return cleaned_df, validation_result


def monitor_processing(func):
    """
    Decorator to monitor memory and timing for processing functions
    
    Args:
        func: Function to monitor
        
    Returns:
        Wrapped function with monitoring
    """
    def wrapper(*args, **kwargs):
        # Initialize monitoring
        timer = ProcessingTimer()
        memory_monitor = MemoryMonitor()
        
        # Start monitoring
        timer.start()
        memory_monitor.log_memory_usage("before processing")
        
        try:
            # Execute function
            result = func(*args, **kwargs)
            
            # Log completion
            timer.stop()
            memory_monitor.log_memory_usage("after processing")
            
            return result
            
        except Exception as e:
            logger.error(f"Processing failed: {str(e)}")
            timer.stop()
            raise
    
    return wrapper


if __name__ == "__main__":
    # Example usage of utilities
    print("Data Processing Utilities - Example Usage")
    
    # Create sample data for demonstration
    sample_data = pd.DataFrame({
        'timestamp': [1633221332117, 1633224214164, 1633221999827],
        'visitorid': [1, 2, 3],
        'event': ['view', 'addtocart', 'transaction'],
        'itemid': [101, 102, 103],
        'transactionid': [None, None, 'txn_123']
    })
    
    # Validate data
    validator = DataValidator()
    validation_result = validator.validate_events_schema(sample_data)
    print(f"Validation result: {validation_result['is_valid']}")
    
    # Monitor memory
    memory_monitor = MemoryMonitor()
    memory_usage = memory_monitor.get_memory_usage()
    print(f"Memory usage: {memory_usage['rss_mb']:.1f}MB")
    
    # Clean data
    cleaner = DataCleaner()
    cleaned_data = cleaner.clean_events_data(sample_data)
    print(f"Cleaned data shape: {cleaned_data.shape}")
    
    print("Utilities demonstration complete!")