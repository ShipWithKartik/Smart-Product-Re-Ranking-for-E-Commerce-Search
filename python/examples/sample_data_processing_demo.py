"""
Sample Data Processing Demonstration Script
Shows how to use the data processing pipeline with sample data
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_processing_pipeline import EventDataProcessor, process_events_data
from config.behavioral_metrics_config import ConfigProfiles, create_config_file
from utils.sample_data_generator import SampleDataGenerator


class SampleEventDataGenerator:
    """Generate sample event data for demonstration"""
    
    def __init__(self, seed: int = 42):
        """Initialize with random seed for reproducibility"""
        random.seed(seed)
        np.random.seed(seed)
        
    def generate_sample_events(self, num_items: int = 50, 
                             num_visitors: int = 200,
                             num_events: int = 5000) -> pd.DataFrame:
        """
        Generate realistic sample event data
        
        Args:
            num_items: Number of unique items
            num_visitors: Number of unique visitors
            num_events: Total number of events to generate
            
        Returns:
            DataFrame with sample events
        """
        print(f"Generating {num_events:,} sample events...")
        
        events = []
        
        # Generate base timestamp (30 days ago)
        base_time = datetime.now() - timedelta(days=30)
        base_timestamp = int(base_time.timestamp() * 1000)
        
        for i in range(num_events):
            # Random visitor and item
            visitorid = random.randint(1, num_visitors)
            itemid = random.randint(1, num_items)
            
            # Generate timestamp within 30-day window
            time_offset = random.randint(0, 30 * 24 * 60 * 60 * 1000)  # 30 days in ms
            timestamp = base_timestamp + time_offset
            
            # Determine event type with realistic probabilities
            event_prob = random.random()
            if event_prob < 0.7:  # 70% views
                event = "view"
                transactionid = None
            elif event_prob < 0.85:  # 15% addtocart
                event = "addtocart"
                transactionid = None
            else:  # 15% transactions
                event = "transaction"
                transactionid = f"txn_{i}_{random.randint(1000, 9999)}"
            
            events.append({
                'timestamp': timestamp,
                'visitorid': visitorid,
                'event': event,
                'itemid': itemid,
                'transactionid': transactionid
            })
        
        # Convert to DataFrame and sort by timestamp
        events_df = pd.DataFrame(events)
        events_df = events_df.sort_values('timestamp').reset_index(drop=True)
        
        print(f"Generated events distribution:")
        print(events_df['event'].value_counts())
        
        return events_df
    
    def save_sample_events(self, events_df: pd.DataFrame, 
                          file_path: str = "sample_events.csv") -> str:
        """Save sample events to CSV file"""
        events_df.to_csv(file_path, index=False)
        print(f"Sample events saved to: {file_path}")
        return file_path


def demonstrate_basic_processing():
    """Demonstrate basic event data processing"""
    print("\n" + "="*60)
    print("BASIC EVENT DATA PROCESSING DEMONSTRATION")
    print("="*60)
    
    # Generate sample data
    generator = SampleEventDataGenerator()
    events_df = generator.generate_sample_events(
        num_items=30,
        num_visitors=100, 
        num_events=2000
    )
    
    # Save to file for processing
    sample_file = generator.save_sample_events(events_df, "demo_events.csv")
    
    try:
        # Process with default configuration
        print("\nProcessing events with default configuration...")
        results = process_events_data(
            sample_file,
            output_dir="demo_output_basic"
        )
        
        print("\nProcessing Results:")
        print(f"  Events processed: {results['summary']['events_processed']:,}")
        print(f"  Items with metrics: {results['summary']['items_with_metrics']:,}")
        print(f"  Processing time: {results['summary']['processing_time_seconds']:.2f} seconds")
        
        # Show sample metrics
        if results['metrics_df'] is not None:
            print("\nSample Behavioral Metrics:")
            sample_metrics = results['metrics_df'].head()
            print(sample_metrics[['itemid', 'view_count', 'addtocart_rate', 'conversion_rate']].to_string())
        
        return True
        
    except Exception as e:
        print(f"Error in basic processing: {str(e)}")
        return False
    
    finally:
        # Clean up
        if os.path.exists(sample_file):
            os.remove(sample_file)


def demonstrate_advanced_processing():
    """Demonstrate advanced processing with custom configuration"""
    print("\n" + "="*60)
    print("ADVANCED EVENT DATA PROCESSING DEMONSTRATION")
    print("="*60)
    
    # Create custom configuration
    config = ConfigProfiles.development()
    config.processing.enable_temporal_features = True
    config.processing.enable_engagement_features = True
    config.output.generate_summary_report = True
    
    # Generate larger sample dataset
    generator = SampleEventDataGenerator()
    events_df = generator.generate_sample_events(
        num_items=100,
        num_visitors=500,
        num_events=10000
    )
    
    sample_file = generator.save_sample_events(events_df, "demo_events_advanced.csv")
    
    try:
        # Process with custom configuration
        print("\nProcessing events with advanced configuration...")
        processor = EventDataProcessor(config.to_dict())
        results = processor.run_complete_pipeline(
            sample_file,
            output_dir="demo_output_advanced"
        )
        
        print("\nAdvanced Processing Results:")
        print(f"  Events processed: {results['summary']['events_processed']:,}")
        print(f"  Items with metrics: {results['summary']['items_with_metrics']:,}")
        print(f"  Feature columns: {results['summary']['final_features']}")
        print(f"  Processing time: {results['summary']['processing_time_seconds']:.2f} seconds")
        
        # Show processing log summary
        log_summary = processor.get_processing_summary()
        print(f"\nProcessing Summary:")
        print(f"  Steps completed: {log_summary['processing_steps']}")
        print(f"  Warnings: {log_summary['warnings']}")
        print(f"  Errors: {log_summary['errors']}")
        
        # Show enhanced metrics
        if results['features_df'] is not None:
            print("\nSample Enhanced Features:")
            feature_cols = ['itemid', 'view_count', 'conversion_rate', 'popularity_score']
            available_cols = [col for col in feature_cols if col in results['features_df'].columns]
            sample_features = results['features_df'][available_cols].head()
            print(sample_features.to_string())
        
        return True
        
    except Exception as e:
        print(f"Error in advanced processing: {str(e)}")
        return False
    
    finally:
        # Clean up
        if os.path.exists(sample_file):
            os.remove(sample_file)


def demonstrate_error_handling():
    """Demonstrate error handling capabilities"""
    print("\n" + "="*60)
    print("ERROR HANDLING DEMONSTRATION")
    print("="*60)
    
    # Test with invalid file
    print("\n1. Testing with non-existent file...")
    try:
        results = process_events_data("nonexistent_file.csv")
        print("Unexpected success!")
    except Exception as e:
        print(f"Expected error caught: {type(e).__name__}: {str(e)}")
    
    # Test with invalid data
    print("\n2. Testing with invalid data...")
    invalid_data = pd.DataFrame({
        'timestamp': [None, -1, 'invalid'],
        'visitorid': [1, None, 3],
        'event': ['view', 'invalid_event', 'transaction'],
        'itemid': [1, 2, None]
    })
    
    invalid_file = "invalid_events.csv"
    invalid_data.to_csv(invalid_file, index=False)
    
    try:
        # Use configuration that continues on warnings
        config = ConfigProfiles.development()
        config.error_handling.continue_on_warnings = True
        config.validation.filter_invalid_rows = True
        
        results = process_events_data(
            invalid_file,
            config=config.to_dict(),
            output_dir="demo_output_error_handling"
        )
        
        print(f"Processing completed with data filtering:")
        print(f"  Events processed: {results['summary']['events_processed']}")
        
    except Exception as e:
        print(f"Error with invalid data: {str(e)}")
    
    finally:
        if os.path.exists(invalid_file):
            os.remove(invalid_file)


def demonstrate_configuration_profiles():
    """Demonstrate different configuration profiles"""
    print("\n" + "="*60)
    print("CONFIGURATION PROFILES DEMONSTRATION")
    print("="*60)
    
    profiles = {
        'minimal': ConfigProfiles.minimal(),
        'development': ConfigProfiles.development(),
        'production': ConfigProfiles.production()
    }
    
    # Generate sample data
    generator = SampleEventDataGenerator()
    events_df = generator.generate_sample_events(
        num_items=20,
        num_visitors=50,
        num_events=1000
    )
    sample_file = generator.save_sample_events(events_df, "demo_events_profiles.csv")
    
    try:
        for profile_name, config in profiles.items():
            print(f"\n{profile_name.upper()} Profile:")
            print(f"  Chunk size: {config.processing.chunk_size}")
            print(f"  Temporal features: {config.processing.enable_temporal_features}")
            print(f"  Strict validation: {config.validation.strict_mode}")
            
            try:
                results = process_events_data(
                    sample_file,
                    config=config.to_dict(),
                    output_dir=f"demo_output_{profile_name}"
                )
                
                print(f"  Processing time: {results['summary']['processing_time_seconds']:.2f}s")
                print(f"  Features: {results['summary']['final_features']}")
                
            except Exception as e:
                print(f"  Error: {str(e)}")
    
    finally:
        if os.path.exists(sample_file):
            os.remove(sample_file)


def main():
    """Run all demonstrations"""
    print("SMART PRODUCT RE-RANKING: DATA PROCESSING DEMONSTRATIONS")
    print("=" * 80)
    
    # Create configuration files for reference
    print("\nCreating configuration files...")
    create_config_file("development", "demo_config_development.json")
    create_config_file("production", "demo_config_production.json")
    print("Configuration files created for reference")
    
    # Run demonstrations
    demos = [
        ("Basic Processing", demonstrate_basic_processing),
        ("Advanced Processing", demonstrate_advanced_processing),
        ("Error Handling", demonstrate_error_handling),
        ("Configuration Profiles", demonstrate_configuration_profiles)
    ]
    
    results = {}
    for demo_name, demo_func in demos:
        try:
            print(f"\nRunning {demo_name}...")
            success = demo_func()
            results[demo_name] = "SUCCESS" if success else "FAILED"
        except Exception as e:
            print(f"Demo failed with error: {str(e)}")
            results[demo_name] = "ERROR"
    
    # Summary
    print("\n" + "="*80)
    print("DEMONSTRATION SUMMARY")
    print("="*80)
    for demo_name, status in results.items():
        print(f"  {demo_name}: {status}")
    
    print("\nDemonstration complete!")
    print("Check the demo_output_* directories for generated files.")


if __name__ == "__main__":
    main()