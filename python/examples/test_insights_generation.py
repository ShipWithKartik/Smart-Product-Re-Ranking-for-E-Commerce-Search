"""
Test script for Insights Generation and Reporting functionality
Tests core functionality with sample data
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from evaluation.insights_generator import (
    EventDataAnalyzer,
    ModelInsightsAnalyzer, 
    InsightsReportGenerator,
    BusinessInsight,
    generate_comprehensive_insights
)


def create_sample_events_data():
    """Create sample events data for testing"""
    
    np.random.seed(42)
    
    # Generate sample events
    n_events = 1000
    n_visitors = 200
    n_items = 50
    
    events_data = []
    
    for i in range(n_events):
        visitor_id = np.random.randint(1, n_visitors + 1)
        item_id = np.random.randint(1, n_items + 1)
        
        # Generate realistic event sequence
        event_type = np.random.choice(['view', 'addtocart', 'transaction'], 
                                    p=[0.7, 0.2, 0.1])
        
        timestamp = 1433221332117 + i * 1000  # Sequential timestamps
        
        events_data.append({
            'timestamp': timestamp,
            'visitorid': visitor_id,
            'event': event_type,
            'itemid': item_id,
            'transactionid': f'txn_{i}' if event_type == 'transaction' else ''
        })
    
    return pd.DataFrame(events_data)


def create_sample_model_results():
    """Create sample model results for testing"""
    
    return {
        'model_name': 'Test_Product_Ranking_Model',
        'model_type': 'logistic_regression',
        'performance_report': {
            'evaluation_metrics': {
                'roc_auc': 0.756,
                'precision': 0.723,
                'recall': 0.678,
                'f1_score': 0.700
            },
            'performance_grade': 'Good',
            'validation_passed': True,
            'feature_importance': [
                {'feature': 'conversion_rate', 'importance': 0.345},
                {'feature': 'addtocart_rate', 'importance': 0.289},
                {'feature': 'view_count', 'importance': 0.167},
                {'feature': 'unique_visitors', 'importance': 0.123},
                {'feature': 'transaction_count', 'importance': 0.076}
            ]
        }
    }


def create_sample_ab_test_results():
    """Create sample A/B test results for testing"""
    
    return {
        'test_config': {
            'num_users_per_group': 500,
            'items_per_user': 10,
            'confidence_level': 0.95,
            'simulation_days': 30
        },
        'baseline_metrics': {
            'click_through_rate': 0.0534,
            'conversion_rate': 0.0245,
            'purchase_rate': 0.0131,
            'total_views': 2500,
            'total_clicks': 134,
            'total_purchases': 33
        },
        'ml_ranking_metrics': {
            'click_through_rate': 0.0612,
            'conversion_rate': 0.0289,
            'purchase_rate': 0.0177,
            'total_views': 2500,
            'total_clicks': 153,
            'total_purchases': 44
        },
        'statistical_significance': {
            'purchase_count': {
                'p_value': 0.018,
                'significant': True,
                'relative_improvement': 35.1,
                't_statistic': 2.45
            },
            'click_through_rate': {
                'p_value': 0.032,
                'significant': True,
                'relative_improvement': 14.6
            },
            'conversion_rate': {
                'p_value': 0.041,
                'significant': True,
                'relative_improvement': 18.0
            },
            'summary': {
                'total_tests': 3,
                'significant_tests': 3,
                'primary_metric_significant': True,
                'min_effect_size_met': True
            }
        },
        'business_impact': {
            'revenue_impact': {
                'revenue_lift_per_user': 3.45,
                'revenue_lift_percentage': 15.2,
                'baseline_revenue_per_user': 22.70,
                'ml_revenue_per_user': 26.15,
                'avg_order_value': 50.0
            },
            'metric_improvements': {
                'click_through_rate': {
                    'baseline_value': 0.0534,
                    'ml_value': 0.0612,
                    'relative_improvement': 14.6,
                    'is_significant': True
                },
                'conversion_rate': {
                    'baseline_value': 0.0245,
                    'ml_value': 0.0289,
                    'relative_improvement': 18.0,
                    'is_significant': True
                },
                'purchase_rate': {
                    'baseline_value': 0.0131,
                    'ml_value': 0.0177,
                    'relative_improvement': 35.1,
                    'is_significant': True
                }
            },
            'overall_assessment': {
                'primary_metric_improvement': 35.1,
                'is_statistically_significant': True,
                'meets_minimum_effect_size': True,
                'recommendation': 'RECOMMEND IMPLEMENTATION - ML ranking shows significant positive impact'
            }
        }
    }


def test_event_data_analyzer():
    """Test EventDataAnalyzer functionality"""
    
    print("Testing EventDataAnalyzer...")
    
    # Create sample data
    events_df = create_sample_events_data()
    
    # Create analyzer
    analyzer = EventDataAnalyzer()
    
    # Test analysis
    analysis_results = analyzer.analyze_event_patterns(events_df)
    
    # Validate results
    assert 'event_statistics' in analysis_results
    assert 'conversion_funnel' in analysis_results
    assert 'temporal_patterns' in analysis_results
    assert 'item_performance' in analysis_results
    assert 'user_behavior' in analysis_results
    
    event_stats = analysis_results['event_statistics']
    assert event_stats['total_events'] == len(events_df)
    assert event_stats['unique_visitors'] > 0
    assert event_stats['unique_items'] > 0
    
    print("✓ EventDataAnalyzer tests passed")
    return analysis_results


def test_model_insights_analyzer():
    """Test ModelInsightsAnalyzer functionality"""
    
    print("Testing ModelInsightsAnalyzer...")
    
    # Create sample data
    model_results = create_sample_model_results()
    ab_test_results = create_sample_ab_test_results()
    
    # Create analyzer
    analyzer = ModelInsightsAnalyzer()
    
    # Test analysis
    insights = analyzer.analyze_model_performance_insights(model_results, ab_test_results)
    
    # Validate results
    assert len(insights) > 0
    assert all(isinstance(insight, BusinessInsight) for insight in insights)
    
    # Check insight types
    insight_types = [insight.insight_type for insight in insights]
    expected_types = ['model_performance', 'feature_importance', 'ab_test_performance', 
                     'business_impact', 'statistical_significance']
    
    for expected_type in expected_types:
        assert expected_type in insight_types, f"Missing insight type: {expected_type}"
    
    # Validate insight structure
    for insight in insights:
        assert hasattr(insight, 'title')
        assert hasattr(insight, 'description')
        assert hasattr(insight, 'quantified_impact')
        assert hasattr(insight, 'confidence_level')
        assert hasattr(insight, 'recommendation')
    
    print("✓ ModelInsightsAnalyzer tests passed")
    return insights


def test_insights_report_generator():
    """Test InsightsReportGenerator functionality"""
    
    print("Testing InsightsReportGenerator...")
    
    # Create sample data
    events_df = create_sample_events_data()
    model_results = create_sample_model_results()
    ab_test_results = create_sample_ab_test_results()
    
    # Create generator
    generator = InsightsReportGenerator()
    
    # Test report generation
    report = generator.create_automated_insights_report(
        events_df, model_results, ab_test_results
    )
    
    # Validate report structure
    assert hasattr(report, 'report_id')
    assert hasattr(report, 'generation_timestamp')
    assert hasattr(report, 'executive_summary')
    assert hasattr(report, 'key_findings')
    assert hasattr(report, 'performance_metrics')
    assert hasattr(report, 'business_impact')
    assert hasattr(report, 'recommendations')
    assert hasattr(report, 'technical_summary')
    assert hasattr(report, 'data_quality_assessment')
    
    # Validate content
    assert len(report.key_findings) > 0
    assert len(report.recommendations) > 0
    assert len(report.executive_summary) > 0
    
    # Test export functionality
    json_path = generator.export_insights_report(report, "json", "test_report.json")
    assert os.path.exists(json_path)
    
    txt_path = generator.export_insights_report(report, "txt", "test_report.txt")
    assert os.path.exists(txt_path)
    
    # Clean up test files
    if os.path.exists(json_path):
        os.remove(json_path)
    if os.path.exists(txt_path):
        os.remove(txt_path)
    
    print("✓ InsightsReportGenerator tests passed")
    return report


def test_convenience_functions():
    """Test convenience functions"""
    
    print("Testing convenience functions...")
    
    # Create sample data
    events_df = create_sample_events_data()
    model_results = create_sample_model_results()
    ab_test_results = create_sample_ab_test_results()
    
    # Test generate_comprehensive_insights
    report = generate_comprehensive_insights(events_df, model_results, ab_test_results)
    
    assert hasattr(report, 'report_id')
    assert len(report.key_findings) > 0
    
    print("✓ Convenience functions tests passed")
    return report


def run_all_tests():
    """Run all tests for insights generation functionality"""
    
    print("=" * 60)
    print("RUNNING INSIGHTS GENERATION TESTS")
    print("=" * 60)
    
    try:
        # Test individual components
        event_analysis = test_event_data_analyzer()
        model_insights = test_model_insights_analyzer()
        insights_report = test_insights_report_generator()
        convenience_report = test_convenience_functions()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED SUCCESSFULLY!")
        print("=" * 60)
        
        # Display sample results
        print(f"\nSample Results:")
        print(f"Event Analysis - Total Events: {event_analysis['event_statistics']['total_events']}")
        print(f"Model Insights - Generated: {len(model_insights)} insights")
        print(f"Insights Report - ID: {insights_report.report_id}")
        print(f"Key Findings: {len(insights_report.key_findings)}")
        print(f"Recommendations: {len(insights_report.recommendations)}")
        
        # Show sample insight
        if insights_report.key_findings:
            sample_insight = insights_report.key_findings[0]
            print(f"\nSample Insight:")
            print(f"Title: {sample_insight.title}")
            print(f"Type: {sample_insight.insight_type}")
            print(f"Confidence: {sample_insight.confidence_level}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    
    if success:
        print("\n🎉 All insights generation tests completed successfully!")
        print("The insights generation and reporting functionality is working correctly.")
    else:
        print("\n❌ Some tests failed. Please check the error messages above.")
        sys.exit(1)