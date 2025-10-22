"""
Test script for behavioral metrics calculation
Creates sample data and tests the behavioral metrics functions
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
from data.behavioral_metrics import BehavioralMetricsCalculator, calculate_behavioral_metrics


def create_sample_data():
    """Create sample event data for testing"""
    
    # Sample events data
    sample_events = [
        # Item 1: High performing item
        {'timestamp': 1433221332117, 'visitorid': 1, 'event': 'view', 'itemid': 100, 'transactionid': None},
        {'timestamp': 1433221332118, 'visitorid': 1, 'event': 'addtocart', 'itemid': 100, 'transactionid': None},
        {'timestamp': 1433221332119, 'visitorid': 1, 'event': 'transaction', 'itemid': 100, 'transactionid': 'T1'},
        {'timestamp': 1433221332120, 'visitorid': 2, 'event': 'view', 'itemid': 100, 'transactionid': None},
        {'timestamp': 1433221332121, 'visitorid': 2, 'event': 'addtocart', 'itemid': 100, 'transactionid': None},
        {'timestamp': 1433221332122, 'visitorid': 2, 'event': 'transaction', 'itemid': 100, 'transactionid': 'T2'},
        
        # Item 2: Medium performing item
        {'timestamp': 1433221332123, 'visitorid': 3, 'event': 'view', 'itemid': 200, 'transactionid': None},
        {'timestamp': 1433221332124, 'visitorid': 3, 'event': 'addtocart', 'itemid': 200, 'transactionid': None},
        {'timestamp': 1433221332125, 'visitorid': 4, 'event': 'view', 'itemid': 200, 'transactionid': None},
        {'timestamp': 1433221332126, 'visitorid': 5, 'event': 'view', 'itemid': 200, 'transactionid': None},
        
        # Item 3: Low performing item (views only)
        {'timestamp': 1433221332127, 'visitorid': 6, 'event': 'view', 'itemid': 300, 'transactionid': None},
        {'timestamp': 1433221332128, 'visitorid': 7, 'event': 'view', 'itemid': 300, 'transactionid': None},
        {'timestamp': 1433221332129, 'visitorid': 8, 'event': 'view', 'itemid': 300, 'transactionid': None},
    ]
    
    return pd.DataFrame(sample_events)


def test_behavioral_metrics():
    """Test the behavioral metrics calculation"""
    
    print("=== Testing Behavioral Metrics Calculation ===\n")
    
    # Create sample data
    print("1. Creating sample data...")
    events_df = create_sample_data()
    print(f"   Created {len(events_df)} sample events")
    print(f"   Event distribution:")
    for event_type, count in events_df['event'].value_counts().items():
        print(f"     {event_type}: {count}")
    
    # Initialize calculator
    calculator = BehavioralMetricsCalculator()
    
    # Test individual metric calculations
    print("\n2. Testing individual metric calculations...")
    
    # Test view metrics
    view_metrics = calculator.calculate_view_metrics(events_df)
    print(f"   View metrics calculated for {len(view_metrics)} items:")
    print(view_metrics.to_string(index=False))
    
    # Test addtocart metrics
    addtocart_metrics = calculator.calculate_addtocart_metrics(events_df)
    print(f"\n   Addtocart metrics calculated for {len(addtocart_metrics)} items:")
    print(addtocart_metrics.to_string(index=False))
    
    # Test transaction metrics
    transaction_metrics = calculator.calculate_transaction_metrics(events_df)
    print(f"\n   Transaction metrics calculated for {len(transaction_metrics)} items:")
    print(transaction_metrics.to_string(index=False))
    
    # Test comprehensive metrics
    print("\n3. Testing comprehensive metrics calculation...")
    all_metrics = calculate_behavioral_metrics(events_df)
    print(f"   Comprehensive metrics calculated for {len(all_metrics)} items:")
    print(all_metrics.to_string(index=False))
    
    # Validate expected results
    print("\n4. Validating results...")
    
    # Item 100 should have perfect conversion rates
    item_100 = all_metrics[all_metrics['itemid'] == 100].iloc[0]
    assert item_100['view_count'] == 2, f"Expected 2 views for item 100, got {item_100['view_count']}"
    assert item_100['addtocart_count'] == 2, f"Expected 2 addtocarts for item 100, got {item_100['addtocart_count']}"
    assert item_100['transaction_count'] == 2, f"Expected 2 transactions for item 100, got {item_100['transaction_count']}"
    assert item_100['addtocart_rate'] == 1.0, f"Expected addtocart rate 1.0 for item 100, got {item_100['addtocart_rate']}"
    assert item_100['conversion_rate'] == 1.0, f"Expected conversion rate 1.0 for item 100, got {item_100['conversion_rate']}"
    assert item_100['cart_conversion_rate'] == 1.0, f"Expected cart conversion rate 1.0 for item 100, got {item_100['cart_conversion_rate']}"
    print("   ✓ Item 100 metrics validated")
    
    # Item 200 should have partial conversion
    item_200 = all_metrics[all_metrics['itemid'] == 200].iloc[0]
    assert item_200['view_count'] == 3, f"Expected 3 views for item 200, got {item_200['view_count']}"
    assert item_200['addtocart_count'] == 1, f"Expected 1 addtocart for item 200, got {item_200['addtocart_count']}"
    assert item_200['transaction_count'] == 0, f"Expected 0 transactions for item 200, got {item_200['transaction_count']}"
    assert abs(item_200['addtocart_rate'] - 0.3333) < 0.01, f"Expected addtocart rate ~0.33 for item 200, got {item_200['addtocart_rate']}"
    assert item_200['conversion_rate'] == 0.0, f"Expected conversion rate 0.0 for item 200, got {item_200['conversion_rate']}"
    assert item_200['cart_conversion_rate'] == 0.0, f"Expected cart conversion rate 0.0 for item 200, got {item_200['cart_conversion_rate']}"
    print("   ✓ Item 200 metrics validated")
    
    # Item 300 should have no conversions
    item_300 = all_metrics[all_metrics['itemid'] == 300].iloc[0]
    assert item_300['view_count'] == 3, f"Expected 3 views for item 300, got {item_300['view_count']}"
    assert item_300['addtocart_count'] == 0, f"Expected 0 addtocarts for item 300, got {item_300['addtocart_count']}"
    assert item_300['transaction_count'] == 0, f"Expected 0 transactions for item 300, got {item_300['transaction_count']}"
    assert item_300['addtocart_rate'] == 0.0, f"Expected addtocart rate 0.0 for item 300, got {item_300['addtocart_rate']}"
    assert item_300['conversion_rate'] == 0.0, f"Expected conversion rate 0.0 for item 300, got {item_300['conversion_rate']}"
    assert item_300['cart_conversion_rate'] == 0.0, f"Expected cart conversion rate 0.0 for item 300, got {item_300['cart_conversion_rate']}"
    print("   ✓ Item 300 metrics validated")
    
    # Test summary statistics
    print("\n5. Testing summary statistics...")
    summary = calculator.get_metrics_summary(all_metrics)
    print(f"   Summary calculated with {len(summary)} metrics")
    
    expected_totals = {
        'total_items': 3,
        'total_views': 8,
        'total_addtocarts': 3,
        'total_transactions': 2,
        'items_with_purchases': 1,
        'items_with_addtocarts': 2
    }
    
    for key, expected_value in expected_totals.items():
        actual_value = summary[key]
        assert actual_value == expected_value, f"Expected {key}={expected_value}, got {actual_value}"
        print(f"   ✓ {key}: {actual_value}")
    
    print("\n=== All Tests Passed! ===")
    return True


if __name__ == "__main__":
    try:
        test_behavioral_metrics()
        print("\n✅ Behavioral metrics implementation is working correctly!")
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()