"""
Behavioral Metrics Pipeline
Integrates CSV loading with behavioral metrics calculation with comprehensive error handling and logging
"""

import pandas as pd
import logging
import sys
import time
from typing import Optional, Dict, Any
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from .csv_loader import CSVDataLoader, load_events_csv
from .behavioral_metrics import BehavioralMetricsCalculator, calculate_behavioral_metrics
from utils.logging_config import get_logger, with_error_handling, with_performance_logging, SystemMonitor


class BehavioralMetricsPipeline:
    """Pipeline for loading events and calculating behavioral metrics with comprehensive error handling"""
    
    def __init__(self, events_file_path: str, chunk_size: int = 10000):
        """
        Initialize the behavioral metrics pipeline
        
        Args:
            events_file_path: Path to the events.csv file
            chunk_size: Chunk size for processing large files
        """
        self.logger = get_logger("behavioral_pipeline")
        self.monitor = SystemMonitor(self.logger)
        
        self.events_file_path = events_file_path
        self.chunk_size = chunk_size
        
        # Initialize components with error handling
        try:
            self.csv_loader = CSVDataLoader(events_file_path, chunk_size)
            self.metrics_calculator = BehavioralMetricsCalculator()
            self.logger.info(f"Initialized BehavioralMetricsPipeline with file: {events_file_path}")
        except Exception as e:
            self.logger.error(f"Failed to initialize BehavioralMetricsPipeline: {str(e)}", exc_info=True)
            raise
        
        self.events_df = None
        self.metrics_df = None
        self.processing_stats = {}
    
    @with_error_handling(get_logger("behavioral_pipeline"), "load_and_validate_events", max_retries=2)
    @with_performance_logging(get_logger("behavioral_pipeline"), "load_and_validate_events")
    def load_and_validate_events(self, filter_invalid: bool = True) -> pd.DataFrame:
        """
        Load and validate events data with comprehensive error handling
        
        Args:
            filter_invalid: Whether to filter out invalid rows
            
        Returns:
            DataFrame with validated events
            
        Raises:
            FileNotFoundError: If events file doesn't exist
            ValueError: If data validation fails
            MemoryError: If file is too large to process
        """
        self.monitor.checkpoint("start_load_validation")
        self.logger.info(f"Loading and validating events data from: {self.events_file_path}")
        
        try:
            # Validate file exists
            if not Path(self.events_file_path).exists():
                raise FileNotFoundError(f"Events file not found: {self.events_file_path}")
            
            # Check file size
            file_size_mb = Path(self.events_file_path).stat().st_size / (1024 * 1024)
            self.logger.info(f"Processing file size: {file_size_mb:.2f} MB")
            
            if file_size_mb > 1000:  # 1GB limit
                self.logger.warning(f"Large file detected ({file_size_mb:.2f} MB). Consider using chunked processing.")
            
            self.monitor.checkpoint("file_validation_complete")
            # Load events with validation
            self.events_df = self.csv_loader.load_events(
                validate=True, 
                filter_invalid=filter_invalid
            )
            
            self.monitor.checkpoint("events_loaded")
            
            # Validate loaded data
            if self.events_df is None or self.events_df.empty:
                raise ValueError("No valid events data loaded")
            
            # Log data quality metrics
            data_quality = {
                'total_events': len(self.events_df),
                'unique_items': self.events_df['itemid'].nunique(),
                'unique_visitors': self.events_df['visitorid'].nunique(),
                'missing_values': self.events_df.isnull().sum().sum(),
                'duplicate_rows': self.events_df.duplicated().sum()
            }
            
            self.logger.log_data_quality("events_dataset", data_quality)
            self.logger.info(f"Successfully loaded {len(self.events_df):,} events")
            
            # Log event distribution
            if not self.events_df.empty:
                event_counts = self.events_df['event'].value_counts()
                self.logger.info("Event distribution:")
                for event_type, count in event_counts.items():
                    self.logger.info(f"  {event_type}: {count:,}")
            
            self.processing_stats['events_loaded'] = len(self.events_df)
            self.monitor.checkpoint("validation_complete")
            
            return self.events_df
            
        except FileNotFoundError as e:
            self.logger.error(f"Events file not found: {str(e)}")
            raise
        except MemoryError as e:
            self.logger.error(f"Insufficient memory to load events: {str(e)}")
            raise
        except pd.errors.EmptyDataError as e:
            self.logger.error(f"Events file is empty or corrupted: {str(e)}")
            raise ValueError("Events file contains no valid data")
        except Exception as e:
            self.logger.error(f"Unexpected error loading events: {str(e)}", exc_info=True)
            raise
    
    @with_error_handling(get_logger("behavioral_pipeline"), "calculate_metrics", max_retries=1)
    @with_performance_logging(get_logger("behavioral_pipeline"), "calculate_metrics")
    def calculate_metrics(self, include_temporal: bool = False) -> pd.DataFrame:
        """
        Calculate behavioral metrics from loaded events with error handling
        
        Args:
            include_temporal: Whether to include temporal features
            
        Returns:
            DataFrame with behavioral metrics
            
        Raises:
            ValueError: If events data not loaded or invalid
            RuntimeError: If metrics calculation fails
        """
        if self.events_df is None:
            raise ValueError("Events data not loaded. Call load_and_validate_events() first.")
        
        if self.events_df.empty:
            raise ValueError("Events data is empty. Cannot calculate metrics.")
        
        self.monitor.checkpoint("start_metrics_calculation")
        self.logger.info("Calculating behavioral metrics...")
        
        try:
            # Validate required columns
            required_columns = ['itemid', 'event', 'visitorid']
            missing_columns = [col for col in required_columns if col not in self.events_df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Calculate comprehensive behavioral metrics
            self.metrics_df = calculate_behavioral_metrics(
                self.events_df, 
                include_temporal=include_temporal
            )
            
            self.monitor.checkpoint("metrics_calculated")
            
            # Validate results
            if self.metrics_df is None or self.metrics_df.empty:
                raise RuntimeError("Metrics calculation returned empty results")
            
            # Log metrics quality
            metrics_quality = {
                'total_items': len(self.metrics_df),
                'features_calculated': len(self.metrics_df.columns) - 1,  # Exclude itemid
                'missing_values': self.metrics_df.isnull().sum().sum(),
                'zero_values': (self.metrics_df == 0).sum().sum()
            }
            
            self.logger.log_data_quality("behavioral_metrics", metrics_quality)
            self.logger.info(f"Successfully calculated metrics for {len(self.metrics_df):,} items")
            
            self.processing_stats['metrics_calculated'] = len(self.metrics_df)
            self.monitor.checkpoint("metrics_validation_complete")
            
            return self.metrics_df
            
        except ValueError as e:
            self.logger.error(f"Data validation error in metrics calculation: {str(e)}")
            raise
        except RuntimeError as e:
            self.logger.error(f"Runtime error in metrics calculation: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error calculating metrics: {str(e)}", exc_info=True)
            raise RuntimeError(f"Metrics calculation failed: {str(e)}")
    
    def get_pipeline_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive summary of the pipeline results
        
        Returns:
            Dictionary with pipeline summary
        """
        summary = {}
        
        # Events summary
        if self.events_df is not None:
            summary['events'] = {
                'total_events': len(self.events_df),
                'unique_items': self.events_df['itemid'].nunique(),
                'unique_visitors': self.events_df['visitorid'].nunique(),
                'event_distribution': self.events_df['event'].value_counts().to_dict(),
                'date_range': {
                    'start': int(self.events_df['timestamp'].min()),
                    'end': int(self.events_df['timestamp'].max())
                } if 'timestamp' in self.events_df.columns else None
            }
        
        # Metrics summary
        if self.metrics_df is not None:
            metrics_summary = self.metrics_calculator.get_metrics_summary(self.metrics_df)
            summary['metrics'] = metrics_summary
        
        return summary
    
    def save_results(self, output_dir: str = "output") -> Dict[str, str]:
        """
        Save pipeline results to files
        
        Args:
            output_dir: Directory to save results
            
        Returns:
            Dictionary with saved file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        saved_files = {}
        
        # Save events data
        if self.events_df is not None:
            events_file = output_path / "processed_events.csv"
            self.events_df.to_csv(events_file, index=False)
            saved_files['events'] = str(events_file)
            logger.info(f"Saved processed events to: {events_file}")
        
        # Save metrics data
        if self.metrics_df is not None:
            metrics_file = output_path / "behavioral_metrics.csv"
            self.metrics_df.to_csv(metrics_file, index=False)
            saved_files['metrics'] = str(metrics_file)
            logger.info(f"Saved behavioral metrics to: {metrics_file}")
        
        # Save summary
        summary = self.get_pipeline_summary()
        if summary:
            import json
            summary_file = output_path / "pipeline_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            saved_files['summary'] = str(summary_file)
            logger.info(f"Saved pipeline summary to: {summary_file}")
        
        return saved_files
    
    def run_full_pipeline(self, include_temporal: bool = False, 
                         save_results: bool = True,
                         output_dir: str = "output") -> Dict[str, Any]:
        """
        Run the complete behavioral metrics pipeline
        
        Args:
            include_temporal: Whether to include temporal features
            save_results: Whether to save results to files
            output_dir: Directory to save results
            
        Returns:
            Dictionary with pipeline results and summary
        """
        logger.info("Starting behavioral metrics pipeline...")
        
        try:
            # Step 1: Load and validate events
            events_df = self.load_and_validate_events()
            
            # Step 2: Calculate behavioral metrics
            metrics_df = self.calculate_metrics(include_temporal=include_temporal)
            
            # Step 3: Get summary
            summary = self.get_pipeline_summary()
            
            # Step 4: Save results if requested
            saved_files = {}
            if save_results:
                saved_files = self.save_results(output_dir)
            
            logger.info("Behavioral metrics pipeline completed successfully")
            
            return {
                'events_df': events_df,
                'metrics_df': metrics_df,
                'summary': summary,
                'saved_files': saved_files
            }
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise


# Convenience function
def run_behavioral_metrics_pipeline(events_file_path: str,
                                  include_temporal: bool = False,
                                  save_results: bool = True,
                                  output_dir: str = "output") -> Dict[str, Any]:
    """
    Convenience function to run the complete behavioral metrics pipeline
    
    Args:
        events_file_path: Path to events.csv file
        include_temporal: Whether to include temporal features
        save_results: Whether to save results to files
        output_dir: Directory to save results
        
    Returns:
        Dictionary with pipeline results
    """
    pipeline = BehavioralMetricsPipeline(events_file_path)
    return pipeline.run_full_pipeline(
        include_temporal=include_temporal,
        save_results=save_results,
        output_dir=output_dir
    )


if __name__ == "__main__":
    # Example usage
    try:
        # Run the complete pipeline
        results = run_behavioral_metrics_pipeline(
            "events.csv",
            include_temporal=False,
            save_results=True,
            output_dir="behavioral_metrics_output"
        )
        
        print("Pipeline completed successfully!")
        print(f"Processed {len(results['events_df'])} events")
        print(f"Calculated metrics for {len(results['metrics_df'])} items")
        
        if results['saved_files']:
            print("Saved files:")
            for file_type, file_path in results['saved_files'].items():
                print(f"  {file_type}: {file_path}")
        
    except FileNotFoundError:
        print("Error: events.csv file not found. Please ensure the file exists in the current directory.")
    except Exception as e:
        print(f"Error: {str(e)}")