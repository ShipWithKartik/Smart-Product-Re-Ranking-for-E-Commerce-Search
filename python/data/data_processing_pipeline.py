"""
Finalized Event Data Processing Pipeline with Error Handling
Comprehensive pipeline for processing event data with robust error handling and logging
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import json
import traceback
from datetime import datetime
import warnings

from .csv_loader import CSVDataLoader, ValidationResult
from .behavioral_metrics import BehavioralMetricsCalculator
from .feature_engineering import TemporalFeatureExtractor, EngagementFeatureExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataProcessingError(Exception):
    """Custom exception for data processing errors"""
    pass


class EventDataProcessor:
    """
    Finalized event data processing pipeline with comprehensive error handling
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the event data processor
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or self._get_default_config()
        self.csv_loader = None
        self.metrics_calculator = BehavioralMetricsCalculator()
        self.temporal_extractor = TemporalFeatureExtractor()
        self.engagement_extractor = EngagementFeatureExtractor()
        
        # Processing state
        self.events_df = None
        self.metrics_df = None
        self.features_df = None
        self.processing_log = []
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default processing configuration"""
        return {
            'validation': {
                'strict_mode': True,
                'filter_invalid_rows': True,
                'min_events_per_item': 1,
                'max_processing_time_minutes': 30
            },
            'processing': {
                'chunk_size': 10000,
                'memory_limit_mb': 1000,
                'enable_temporal_features': True,
                'enable_engagement_features': True
            },
            'error_handling': {
                'max_retries': 3,
                'continue_on_warnings': True,
                'save_error_logs': True
            },
            'output': {
                'save_intermediate_results': True,
                'output_directory': 'data_processing_output',
                'file_format': 'csv'
            }
        }
    
    def _log_step(self, step: str, status: str, details: Optional[str] = None):
        """Log processing step with timestamp"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'step': step,
            'status': status,
            'details': details
        }
        self.processing_log.append(log_entry)
        
        if status == 'ERROR':
            logger.error(f"{step}: {details}")
        elif status == 'WARNING':
            logger.warning(f"{step}: {details}")
        else:
            logger.info(f"{step}: {status}")
    
    def load_events_with_error_handling(self, file_path: str) -> pd.DataFrame:
        """
        Load events with comprehensive error handling
        
        Args:
            file_path: Path to events CSV file
            
        Returns:
            DataFrame with loaded events
            
        Raises:
            DataProcessingError: If loading fails after retries
        """
        self._log_step("Event Loading", "STARTED", f"Loading from {file_path}")
        
        max_retries = self.config['error_handling']['max_retries']
        
        for attempt in range(max_retries):
            try:
                # Initialize CSV loader
                self.csv_loader = CSVDataLoader(
                    file_path, 
                    chunk_size=self.config['processing']['chunk_size']
                )
                
                # Load events with validation
                self.events_df = self.csv_loader.load_events(
                    validate=self.config['validation']['strict_mode'],
                    filter_invalid=self.config['validation']['filter_invalid_rows']
                )
                
                # Validate minimum requirements
                if len(self.events_df) == 0:
                    raise DataProcessingError("No valid events found in file")
                
                # Check memory usage
                memory_mb = self.events_df.memory_usage(deep=True).sum() / 1024 / 1024
                if memory_mb > self.config['processing']['memory_limit_mb']:
                    self._log_step("Memory Check", "WARNING", 
                                 f"High memory usage: {memory_mb:.1f}MB")
                
                self._log_step("Event Loading", "SUCCESS", 
                             f"Loaded {len(self.events_df):,} events")
                return self.events_df
                
            except FileNotFoundError as e:
                error_msg = f"File not found: {file_path}"
                self._log_step("Event Loading", "ERROR", error_msg)
                raise DataProcessingError(error_msg) from e
                
            except Exception as e:
                error_msg = f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}"
                self._log_step("Event Loading", "ERROR", error_msg)
                
                if attempt == max_retries - 1:
                    raise DataProcessingError(f"Failed to load events after {max_retries} attempts") from e
                
                logger.warning(f"Retrying in 1 second... ({attempt + 1}/{max_retries})")
                import time
                time.sleep(1)
        
        raise DataProcessingError("Unexpected error in event loading")
    
    def calculate_behavioral_metrics_with_error_handling(self) -> pd.DataFrame:
        """
        Calculate behavioral metrics with error handling
        
        Returns:
            DataFrame with behavioral metrics
            
        Raises:
            DataProcessingError: If calculation fails
        """
        if self.events_df is None:
            raise DataProcessingError("Events data not loaded. Call load_events_with_error_handling() first.")
        
        self._log_step("Behavioral Metrics", "STARTED")
        
        try:
            # Calculate core behavioral metrics
            self.metrics_df = self.metrics_calculator.calculate_all_behavioral_metrics(self.events_df)
            
            # Validate results
            if self.metrics_df.empty:
                raise DataProcessingError("No behavioral metrics calculated")
            
            # Check for minimum events per item if configured
            min_events = self.config['validation']['min_events_per_item']
            if min_events > 1:
                valid_items = self.metrics_df[self.metrics_df['view_count'] >= min_events]
                if len(valid_items) < len(self.metrics_df):
                    filtered_count = len(self.metrics_df) - len(valid_items)
                    self._log_step("Metrics Filtering", "WARNING", 
                                 f"Filtered {filtered_count} items with < {min_events} events")
                    self.metrics_df = valid_items
            
            self._log_step("Behavioral Metrics", "SUCCESS", 
                         f"Calculated metrics for {len(self.metrics_df):,} items")
            return self.metrics_df
            
        except Exception as e:
            error_msg = f"Failed to calculate behavioral metrics: {str(e)}"
            self._log_step("Behavioral Metrics", "ERROR", error_msg)
            raise DataProcessingError(error_msg) from e
    
    def extract_features_with_error_handling(self) -> pd.DataFrame:
        """
        Extract additional features with error handling
        
        Returns:
            DataFrame with enhanced features
            
        Raises:
            DataProcessingError: If feature extraction fails
        """
        if self.metrics_df is None:
            raise DataProcessingError("Behavioral metrics not calculated")
        
        self._log_step("Feature Extraction", "STARTED")
        
        try:
            # Start with behavioral metrics
            self.features_df = self.metrics_df.copy()
            
            # Add temporal features if enabled
            if self.config['processing']['enable_temporal_features']:
                try:
                    temporal_features = self.temporal_extractor.extract_temporal_features(self.events_df)
                    if not temporal_features.empty:
                        self.features_df = self.features_df.merge(
                            temporal_features, on='itemid', how='left'
                        )
                        self._log_step("Temporal Features", "SUCCESS", 
                                     f"Added temporal features for {len(temporal_features)} items")
                except Exception as e:
                    if self.config['error_handling']['continue_on_warnings']:
                        self._log_step("Temporal Features", "WARNING", 
                                     f"Failed to extract temporal features: {str(e)}")
                    else:
                        raise
            
            # Add engagement features if enabled
            if self.config['processing']['enable_engagement_features']:
                try:
                    engagement_features = self.engagement_extractor.extract_engagement_features(self.events_df)
                    if not engagement_features.empty:
                        self.features_df = self.features_df.merge(
                            engagement_features, on='itemid', how='left'
                        )
                        self._log_step("Engagement Features", "SUCCESS", 
                                     f"Added engagement features for {len(engagement_features)} items")
                except Exception as e:
                    if self.config['error_handling']['continue_on_warnings']:
                        self._log_step("Engagement Features", "WARNING", 
                                     f"Failed to extract engagement features: {str(e)}")
                    else:
                        raise
            
            self._log_step("Feature Extraction", "SUCCESS", 
                         f"Final feature matrix: {self.features_df.shape}")
            return self.features_df
            
        except Exception as e:
            error_msg = f"Failed to extract features: {str(e)}"
            self._log_step("Feature Extraction", "ERROR", error_msg)
            raise DataProcessingError(error_msg) from e
    
    def save_results_with_error_handling(self, output_dir: Optional[str] = None) -> Dict[str, str]:
        """
        Save processing results with error handling
        
        Args:
            output_dir: Optional output directory override
            
        Returns:
            Dictionary with saved file paths
        """
        output_path = Path(output_dir or self.config['output']['output_directory'])
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        try:
            # Save events data
            if self.events_df is not None:
                events_file = output_path / "processed_events.csv"
                self.events_df.to_csv(events_file, index=False)
                saved_files['events'] = str(events_file)
                self._log_step("Save Events", "SUCCESS", str(events_file))
            
            # Save behavioral metrics
            if self.metrics_df is not None:
                metrics_file = output_path / "behavioral_metrics.csv"
                self.metrics_df.to_csv(metrics_file, index=False)
                saved_files['metrics'] = str(metrics_file)
                self._log_step("Save Metrics", "SUCCESS", str(metrics_file))
            
            # Save feature matrix
            if self.features_df is not None:
                features_file = output_path / "feature_matrix.csv"
                self.features_df.to_csv(features_file, index=False)
                saved_files['features'] = str(features_file)
                self._log_step("Save Features", "SUCCESS", str(features_file))
            
            # Save processing log
            if self.config['error_handling']['save_error_logs']:
                log_file = output_path / "processing_log.json"
                with open(log_file, 'w') as f:
                    json.dump(self.processing_log, f, indent=2)
                saved_files['log'] = str(log_file)
                self._log_step("Save Log", "SUCCESS", str(log_file))
            
            return saved_files
            
        except Exception as e:
            error_msg = f"Failed to save results: {str(e)}"
            self._log_step("Save Results", "ERROR", error_msg)
            raise DataProcessingError(error_msg) from e
    
    def run_complete_pipeline(self, file_path: str, 
                            output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Run the complete data processing pipeline with error handling
        
        Args:
            file_path: Path to events CSV file
            output_dir: Optional output directory
            
        Returns:
            Dictionary with pipeline results
        """
        pipeline_start = datetime.now()
        self._log_step("Pipeline", "STARTED", f"Processing {file_path}")
        
        try:
            # Step 1: Load events
            events_df = self.load_events_with_error_handling(file_path)
            
            # Step 2: Calculate behavioral metrics
            metrics_df = self.calculate_behavioral_metrics_with_error_handling()
            
            # Step 3: Extract additional features
            features_df = self.extract_features_with_error_handling()
            
            # Step 4: Save results
            saved_files = self.save_results_with_error_handling(output_dir)
            
            # Calculate processing time
            processing_time = (datetime.now() - pipeline_start).total_seconds()
            
            # Generate summary
            summary = {
                'processing_time_seconds': processing_time,
                'events_processed': len(events_df),
                'items_with_metrics': len(metrics_df),
                'final_features': features_df.shape[1] if features_df is not None else 0,
                'saved_files': saved_files,
                'processing_log': self.processing_log
            }
            
            self._log_step("Pipeline", "SUCCESS", 
                         f"Completed in {processing_time:.1f} seconds")
            
            return {
                'events_df': events_df,
                'metrics_df': metrics_df,
                'features_df': features_df,
                'summary': summary
            }
            
        except Exception as e:
            processing_time = (datetime.now() - pipeline_start).total_seconds()
            error_msg = f"Pipeline failed after {processing_time:.1f} seconds: {str(e)}"
            self._log_step("Pipeline", "ERROR", error_msg)
            
            # Save error log even if pipeline fails
            if self.config['error_handling']['save_error_logs']:
                try:
                    output_path = Path(output_dir or self.config['output']['output_directory'])
                    output_path.mkdir(parents=True, exist_ok=True)
                    error_log_file = output_path / "error_log.json"
                    with open(error_log_file, 'w') as f:
                        json.dump({
                            'error': str(e),
                            'traceback': traceback.format_exc(),
                            'processing_log': self.processing_log
                        }, f, indent=2)
                    logger.info(f"Error log saved to: {error_log_file}")
                except Exception:
                    pass  # Don't fail on error log saving
            
            raise DataProcessingError(error_msg) from e
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """Get summary of processing results"""
        return {
            'events_loaded': len(self.events_df) if self.events_df is not None else 0,
            'metrics_calculated': len(self.metrics_df) if self.metrics_df is not None else 0,
            'features_extracted': self.features_df.shape[1] if self.features_df is not None else 0,
            'processing_steps': len(self.processing_log),
            'errors': len([log for log in self.processing_log if log['status'] == 'ERROR']),
            'warnings': len([log for log in self.processing_log if log['status'] == 'WARNING'])
        }


# Convenience function for simple usage
def process_events_data(file_path: str, 
                       config: Optional[Dict[str, Any]] = None,
                       output_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function to process events data with error handling
    
    Args:
        file_path: Path to events CSV file
        config: Optional configuration dictionary
        output_dir: Optional output directory
        
    Returns:
        Dictionary with processing results
    """
    processor = EventDataProcessor(config)
    return processor.run_complete_pipeline(file_path, output_dir)


if __name__ == "__main__":
    # Example usage with error handling
    try:
        results = process_events_data(
            "events.csv",
            output_dir="data_processing_output"
        )
        
        print("Data processing completed successfully!")
        print(f"Events processed: {results['summary']['events_processed']:,}")
        print(f"Items with metrics: {results['summary']['items_with_metrics']:,}")
        print(f"Processing time: {results['summary']['processing_time_seconds']:.1f} seconds")
        
        if results['summary']['saved_files']:
            print("\nSaved files:")
            for file_type, file_path in results['summary']['saved_files'].items():
                print(f"  {file_type}: {file_path}")
        
    except DataProcessingError as e:
        print(f"Data processing failed: {str(e)}")
    except FileNotFoundError:
        print("Error: events.csv file not found. Please ensure the file exists.")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")