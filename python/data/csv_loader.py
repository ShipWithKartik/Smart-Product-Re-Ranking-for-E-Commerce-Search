"""
CSV Data Loader and Validator for Smart Product Re-Ranking System
Handles loading and validation of events.csv with proper data type handling
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EventType(Enum):
    """Enumeration of valid event types"""
    VIEW = "view"
    ADDTOCART = "addtocart"
    TRANSACTION = "transaction"


@dataclass
class ValidationResult:
    """Result of data validation"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    row_count: int
    valid_row_count: int


@dataclass
class EventData:
    """Structured representation of event data"""
    timestamp: int
    visitorid: int
    event: str
    itemid: int
    transactionid: Optional[str] = None


class CSVDataLoader:
    """CSV data loader with validation for events.csv"""
    
    REQUIRED_COLUMNS = ['timestamp', 'visitorid', 'event', 'itemid']
    OPTIONAL_COLUMNS = ['transactionid']
    VALID_EVENT_TYPES = {e.value for e in EventType}
    
    def __init__(self, file_path: str, chunk_size: int = 10000):
        """
        Initialize CSV data loader
        
        Args:
            file_path: Path to the events.csv file
            chunk_size: Number of rows to process at once for large files
        """
        self.file_path = Path(file_path)
        self.chunk_size = chunk_size
        self._validate_file_exists()
    
    def _validate_file_exists(self) -> None:
        """Validate that the CSV file exists"""
        if not self.file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.file_path}")
        
        if not self.file_path.suffix.lower() == '.csv':
            raise ValueError(f"File must be a CSV file: {self.file_path}")
    
    def load_events(self, validate: bool = True, 
                   filter_invalid: bool = True) -> pd.DataFrame:
        """
        Load events from CSV file with optional validation
        
        Args:
            validate: Whether to validate the data
            filter_invalid: Whether to filter out invalid rows
            
        Returns:
            DataFrame with loaded and optionally validated events
        """
        logger.info(f"Loading events from {self.file_path}")
        
        try:
            # First try with flexible data types to handle invalid data
            try:
                # Define data types for efficient loading
                dtype_dict = {
                    'timestamp': 'int64',
                    'visitorid': 'int64', 
                    'event': 'str',
                    'itemid': 'int64',
                    'transactionid': 'str'
                }
                
                # Load CSV with proper data types
                df = pd.read_csv(
                    self.file_path,
                    dtype=dtype_dict,
                    na_values=['', 'NULL', 'null', 'NaN'],
                    keep_default_na=True
                )
            except (ValueError, TypeError) as e:
                # If strict typing fails, load with flexible types and convert later
                logger.warning(f"Strict data type loading failed: {str(e)}. Trying flexible loading...")
                df = pd.read_csv(
                    self.file_path,
                    na_values=['', 'NULL', 'null', 'NaN'],
                    keep_default_na=True
                )
                
                # Convert columns to appropriate types, handling errors
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce').astype('Int64')
                if 'visitorid' in df.columns:
                    df['visitorid'] = pd.to_numeric(df['visitorid'], errors='coerce').astype('Int64')
                if 'itemid' in df.columns:
                    df['itemid'] = pd.to_numeric(df['itemid'], errors='coerce').astype('Int64')
                if 'event' in df.columns:
                    df['event'] = df['event'].astype('str')
                if 'transactionid' in df.columns:
                    df['transactionid'] = df['transactionid'].astype('str')
            
            logger.info(f"Loaded {len(df)} rows from CSV")
            
            if validate:
                validation_result = self.validate_data(df)
                self._log_validation_results(validation_result)
                
                if filter_invalid and not validation_result.is_valid:
                    df = self._filter_invalid_rows(df)
                    logger.info(f"Filtered to {len(df)} valid rows")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading CSV file: {str(e)}")
            raise
    
    def load_events_chunked(self, validate: bool = True, 
                           filter_invalid: bool = True) -> pd.DataFrame:
        """
        Load events in chunks for large files
        
        Args:
            validate: Whether to validate each chunk
            filter_invalid: Whether to filter out invalid rows
            
        Returns:
            DataFrame with all loaded events
        """
        logger.info(f"Loading events in chunks of {self.chunk_size}")
        
        chunks = []
        chunk_count = 0
        
        try:
            # Try with strict data types first
            try:
                dtype_dict = {
                    'timestamp': 'int64',
                    'visitorid': 'int64',
                    'event': 'str', 
                    'itemid': 'int64',
                    'transactionid': 'str'
                }
                
                chunk_reader = pd.read_csv(
                    self.file_path,
                    chunksize=self.chunk_size,
                    dtype=dtype_dict,
                    na_values=['', 'NULL', 'null', 'NaN'],
                    keep_default_na=True
                )
            except (ValueError, TypeError) as e:
                # If strict typing fails, use flexible types
                logger.warning(f"Strict data type loading failed: {str(e)}. Using flexible loading...")
                chunk_reader = pd.read_csv(
                    self.file_path,
                    chunksize=self.chunk_size,
                    na_values=['', 'NULL', 'null', 'NaN'],
                    keep_default_na=True
                )
            
            for chunk in chunk_reader:
                chunk_count += 1
                logger.info(f"Processing chunk {chunk_count} ({len(chunk)} rows)")
                
                # Convert data types if we used flexible loading
                if 'dtype_dict' not in locals():
                    # Convert columns to appropriate types, handling errors
                    if 'timestamp' in chunk.columns:
                        chunk['timestamp'] = pd.to_numeric(chunk['timestamp'], errors='coerce').astype('Int64')
                    if 'visitorid' in chunk.columns:
                        chunk['visitorid'] = pd.to_numeric(chunk['visitorid'], errors='coerce').astype('Int64')
                    if 'itemid' in chunk.columns:
                        chunk['itemid'] = pd.to_numeric(chunk['itemid'], errors='coerce').astype('Int64')
                    if 'event' in chunk.columns:
                        chunk['event'] = chunk['event'].astype('str')
                    if 'transactionid' in chunk.columns:
                        chunk['transactionid'] = chunk['transactionid'].astype('str')
                
                if validate:
                    validation_result = self.validate_data(chunk)
                    if filter_invalid and not validation_result.is_valid:
                        chunk = self._filter_invalid_rows(chunk)
                
                chunks.append(chunk)
            
            # Combine all chunks
            df = pd.concat(chunks, ignore_index=True)
            logger.info(f"Combined {chunk_count} chunks into {len(df)} total rows")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading CSV file in chunks: {str(e)}")
            raise
    
    def validate_data(self, df: pd.DataFrame) -> ValidationResult:
        """
        Validate the loaded data structure and content
        
        Args:
            df: DataFrame to validate
            
        Returns:
            ValidationResult with validation details
        """
        errors = []
        warnings = []
        
        # Check required columns
        missing_cols = set(self.REQUIRED_COLUMNS) - set(df.columns)
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
        
        # Check for empty DataFrame
        if df.empty:
            errors.append("DataFrame is empty")
            return ValidationResult(False, errors, warnings, 0, 0)
        
        original_count = len(df)
        valid_rows = df.copy()
        
        # Validate data types and content
        if 'timestamp' in df.columns:
            invalid_timestamps = df['timestamp'].isna() | (df['timestamp'] <= 0)
            if invalid_timestamps.any():
                count = invalid_timestamps.sum()
                warnings.append(f"Found {count} invalid timestamps")
                valid_rows = valid_rows[~invalid_timestamps]
        
        if 'visitorid' in df.columns:
            invalid_visitors = df['visitorid'].isna() | (df['visitorid'] <= 0)
            if invalid_visitors.any():
                count = invalid_visitors.sum()
                warnings.append(f"Found {count} invalid visitor IDs")
                valid_rows = valid_rows[~invalid_visitors]
        
        if 'itemid' in df.columns:
            invalid_items = df['itemid'].isna() | (df['itemid'] <= 0)
            if invalid_items.any():
                count = invalid_items.sum()
                warnings.append(f"Found {count} invalid item IDs")
                valid_rows = valid_rows[~invalid_items]
        
        if 'event' in df.columns:
            invalid_events = ~df['event'].isin(self.VALID_EVENT_TYPES)
            if invalid_events.any():
                count = invalid_events.sum()
                unique_invalid = df.loc[invalid_events, 'event'].unique()
                warnings.append(f"Found {count} invalid event types: {unique_invalid}")
                valid_rows = valid_rows[~invalid_events]
        
        # Check for duplicate rows
        duplicates = df.duplicated()
        if duplicates.any():
            count = duplicates.sum()
            warnings.append(f"Found {count} duplicate rows")
        
        valid_count = len(valid_rows)
        is_valid = len(errors) == 0 and valid_count > 0
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            row_count=original_count,
            valid_row_count=valid_count
        )
    
    def _filter_invalid_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter out invalid rows from the DataFrame
        
        Args:
            df: DataFrame to filter
            
        Returns:
            DataFrame with only valid rows
        """
        # Start with all rows
        valid_mask = pd.Series(True, index=df.index)
        
        # Filter invalid timestamps
        if 'timestamp' in df.columns:
            valid_mask &= df['timestamp'].notna() & (df['timestamp'] > 0)
        
        # Filter invalid visitor IDs
        if 'visitorid' in df.columns:
            visitor_mask = df['visitorid'].notna() & (df['visitorid'] > 0)
            valid_mask = valid_mask & visitor_mask
        
        # Filter invalid item IDs
        if 'itemid' in df.columns:
            item_mask = df['itemid'].notna() & (df['itemid'] > 0)
            valid_mask = valid_mask & item_mask
        
        # Filter invalid event types
        if 'event' in df.columns:
            event_mask = df['event'].isin(self.VALID_EVENT_TYPES)
            valid_mask = valid_mask & event_mask
        
        return df[valid_mask].copy()
    
    def _log_validation_results(self, result: ValidationResult) -> None:
        """Log validation results"""
        if result.is_valid:
            logger.info(f"Data validation passed: {result.valid_row_count}/{result.row_count} valid rows")
        else:
            logger.error(f"Data validation failed: {result.errors}")
        
        if result.warnings:
            for warning in result.warnings:
                logger.warning(warning)
    
    def classify_events(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Classify events by type for separate processing
        
        Args:
            df: DataFrame with events
            
        Returns:
            Dictionary with event type as key and filtered DataFrame as value
        """
        if 'event' not in df.columns:
            raise ValueError("DataFrame must contain 'event' column")
        
        classified = {}
        for event_type in EventType:
            event_df = df[df['event'] == event_type.value].copy()
            classified[event_type.value] = event_df
            logger.info(f"Classified {len(event_df)} {event_type.value} events")
        
        return classified
    
    def filter_by_event_type(self, df: pd.DataFrame, 
                           event_types: List[str]) -> pd.DataFrame:
        """
        Filter DataFrame to include only specified event types
        
        Args:
            df: DataFrame to filter
            event_types: List of event types to include
            
        Returns:
            Filtered DataFrame
        """
        if 'event' not in df.columns:
            raise ValueError("DataFrame must contain 'event' column")
        
        # Validate event types
        invalid_types = set(event_types) - self.VALID_EVENT_TYPES
        if invalid_types:
            raise ValueError(f"Invalid event types: {invalid_types}")
        
        filtered_df = df[df['event'].isin(event_types)].copy()
        logger.info(f"Filtered to {len(filtered_df)} rows with event types: {event_types}")
        
        return filtered_df
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get summary statistics of the loaded data
        
        Args:
            df: DataFrame to summarize
            
        Returns:
            Dictionary with summary statistics
        """
        if df.empty:
            return {"error": "DataFrame is empty"}
        
        summary = {
            "total_rows": len(df),
            "columns": list(df.columns),
            "data_types": df.dtypes.to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024
        }
        
        # Event-specific summaries
        if 'event' in df.columns:
            summary["event_distribution"] = df['event'].value_counts().to_dict()
        
        if 'timestamp' in df.columns:
            summary["timestamp_range"] = {
                "min": int(df['timestamp'].min()) if df['timestamp'].notna().any() else None,
                "max": int(df['timestamp'].max()) if df['timestamp'].notna().any() else None
            }
        
        if 'visitorid' in df.columns:
            summary["unique_visitors"] = df['visitorid'].nunique()
        
        if 'itemid' in df.columns:
            summary["unique_items"] = df['itemid'].nunique()
        
        return summary


# Convenience functions for common use cases
def load_events_csv(file_path: str, validate: bool = True, 
                   filter_invalid: bool = True, 
                   chunk_size: int = 10000) -> pd.DataFrame:
    """
    Convenience function to load events CSV file
    
    Args:
        file_path: Path to events.csv file
        validate: Whether to validate the data
        filter_invalid: Whether to filter out invalid rows
        chunk_size: Chunk size for large files
        
    Returns:
        DataFrame with loaded events
    """
    loader = CSVDataLoader(file_path, chunk_size)
    
    # Try regular loading first, fall back to chunked for large files
    try:
        return loader.load_events(validate=validate, filter_invalid=filter_invalid)
    except MemoryError:
        logger.warning("Memory error with regular loading, switching to chunked loading")
        return loader.load_events_chunked(validate=validate, filter_invalid=filter_invalid)


def validate_events_csv(file_path: str) -> ValidationResult:
    """
    Convenience function to validate events CSV file
    
    Args:
        file_path: Path to events.csv file
        
    Returns:
        ValidationResult with validation details
    """
    loader = CSVDataLoader(file_path)
    df = pd.read_csv(file_path, nrows=1000)  # Sample for validation
    return loader.validate_data(df)


if __name__ == "__main__":
    # Example usage
    try:
        # Load and validate events
        events_df = load_events_csv("events.csv")
        print(f"Loaded {len(events_df)} events")
        
        # Get data summary
        loader = CSVDataLoader("events.csv")
        summary = loader.get_data_summary(events_df)
        print("Data Summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        
        # Classify events by type
        classified_events = loader.classify_events(events_df)
        for event_type, event_df in classified_events.items():
            print(f"{event_type}: {len(event_df)} events")
            
    except FileNotFoundError:
        print("events.csv file not found. Please ensure the file exists in the current directory.")
    except Exception as e:
        print(f"Error: {str(e)}")