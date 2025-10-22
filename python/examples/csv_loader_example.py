"""
Example usage of CSV Data Loader for Smart Product Re-Ranking System
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.csv_loader import CSVDataLoader, load_events_csv, EventType

def main():
    """Example usage of CSV data loader"""
    
    # Path to events.csv file
    events_file = "events.csv"
    
    try:
        print("=== CSV Data Loader Example ===\n")
        
        # Method 1: Using convenience function
        print("1. Loading events using convenience function...")
        events_df = load_events_csv(events_file, validate=True, filter_invalid=True)
        print(f"   Loaded {len(events_df)} events")
        
        # Method 2: Using CSVDataLoader class
        print("\n2. Using CSVDataLoader class...")
        loader = CSVDataLoader(events_file)
        
        # Get data summary
        summary = loader.get_data_summary(events_df)
        print(f"   Total rows: {summary['total_rows']}")
        print(f"   Unique visitors: {summary['unique_visitors']}")
        print(f"   Unique items: {summary['unique_items']}")
        print(f"   Event distribution: {summary['event_distribution']}")
        
        # Classify events by type
        print("\n3. Classifying events by type...")
        classified_events = loader.classify_events(events_df)
        for event_type in EventType:
            event_df = classified_events[event_type.value]
            print(f"   {event_type.value}: {len(event_df)} events")
        
        # Filter specific event types
        print("\n4. Filtering specific event types...")
        view_events = loader.filter_by_event_type(events_df, ['view'])
        conversion_events = loader.filter_by_event_type(events_df, ['addtocart', 'transaction'])
        
        print(f"   View events: {len(view_events)}")
        print(f"   Conversion events: {len(conversion_events)}")
        
        # Show sample data
        print("\n5. Sample data:")
        print(events_df.head())
        
        print("\n✅ CSV loader example completed successfully!")
        
    except FileNotFoundError:
        print(f"❌ Error: {events_file} not found.")
        print("   Please ensure the events.csv file exists in the project root.")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()