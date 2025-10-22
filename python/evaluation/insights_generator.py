"""
Insights Generation and Reporting for Smart Product Re-Ranking System
Implements automated insights report generation with quantified business impact
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BusinessInsight:
    """Individual business insight with quantified impact"""
    insight_type: str
    title: str
    description: str
    quantified_impact: Dict[str, Any]
    confidence_level: str
    recommendation: str
    supporting_data: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class InsightsReport:
    """Comprehensive insights report with business impact metrics"""
    report_id: str
    generation_timestamp: str
    executive_summary: str
    key_findings: List[BusinessInsight]
    performance_metrics: Dict[str, Any]
    business_impact: Dict[str, Any]
    recommendations: List[str]
    technical_summary: Dict[str, Any]
    data_quality_assessment: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['key_findings'] = [finding.to_dict() for finding in self.key_findings]
        return result


class EventDataAnalyzer:
    """Analyzes event data to extract key insights and patterns"""
    
    def __init__(self):
        self.analysis_cache = {}
    
    def analyze_event_patterns(self, events_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Implement summary statistics and key findings extraction from event data
        
        Args:
            events_df: DataFrame with event data (timestamp, visitorid, event, itemid)
            
        Returns:
            Dictionary with event pattern analysis
        """
        logger.info(f"Analyzing event patterns for {len(events_df)} events")
        
        # Basic event statistics
        event_stats = {
            'total_events': len(events_df),
            'unique_visitors': events_df['visitorid'].nunique(),
            'unique_items': events_df['itemid'].nunique(),
            'event_types': events_df['event'].value_counts().to_dict(),
            'date_range': {
                'start_date': pd.to_datetime(events_df['timestamp'], unit='ms').min().isoformat(),
                'end_date': pd.to_datetime(events_df['timestamp'], unit='ms').max().isoformat(),
                'duration_days': (pd.to_datetime(events_df['timestamp'], unit='ms').max() - 
                                pd.to_datetime(events_df['timestamp'], unit='ms').min()).days
            }
        }
        
        # Conversion funnel analysis
        funnel_analysis = self._analyze_conversion_funnel(events_df)
        
        # Temporal patterns
        temporal_patterns = self._analyze_temporal_patterns(events_df)
        
        # Item performance distribution
        item_performance = self._analyze_item_performance_distribution(events_df)
        
        # User behavior patterns
        user_behavior = self._analyze_user_behavior_patterns(events_df)
        
        return {
            'event_statistics': event_stats,
            'conversion_funnel': funnel_analysis,
            'temporal_patterns': temporal_patterns,
            'item_performance': item_performance,
            'user_behavior': user_behavior
        }
    
    def _analyze_conversion_funnel(self, events_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze conversion funnel from views to purchases"""
        
        # Calculate funnel metrics
        total_views = len(events_df[events_df['event'] == 'view'])
        total_addtocart = len(events_df[events_df['event'] == 'addtocart'])
        total_transactions = len(events_df[events_df['event'] == 'transaction'])
        
        # Calculate conversion rates
        view_to_cart_rate = (total_addtocart / total_views * 100) if total_views > 0 else 0
        cart_to_purchase_rate = (total_transactions / total_addtocart * 100) if total_addtocart > 0 else 0
        view_to_purchase_rate = (total_transactions / total_views * 100) if total_views > 0 else 0
        
        return {
            'funnel_metrics': {
                'total_views': total_views,
                'total_addtocart': total_addtocart,
                'total_transactions': total_transactions
            },
            'conversion_rates': {
                'view_to_cart_rate': round(view_to_cart_rate, 2),
                'cart_to_purchase_rate': round(cart_to_purchase_rate, 2),
                'view_to_purchase_rate': round(view_to_purchase_rate, 2)
            },
            'funnel_efficiency': {
                'cart_abandonment_rate': round(100 - cart_to_purchase_rate, 2),
                'browse_abandonment_rate': round(100 - view_to_cart_rate, 2)
            }
        }
    
    def _analyze_temporal_patterns(self, events_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze temporal patterns in user behavior"""
        
        # Convert timestamp to datetime
        events_df = events_df.copy()
        events_df['datetime'] = pd.to_datetime(events_df['timestamp'], unit='ms')
        events_df['hour'] = events_df['datetime'].dt.hour
        events_df['day_of_week'] = events_df['datetime'].dt.day_name()
        
        # Hourly patterns
        hourly_activity = events_df.groupby('hour').size().to_dict()
        peak_hour = max(hourly_activity, key=hourly_activity.get)
        
        # Daily patterns
        daily_activity = events_df.groupby('day_of_week').size().to_dict()
        peak_day = max(daily_activity, key=daily_activity.get)
        
        # Event type patterns by time
        hourly_by_event = events_df.groupby(['hour', 'event']).size().unstack(fill_value=0)
        
        return {
            'hourly_patterns': {
                'activity_by_hour': hourly_activity,
                'peak_hour': peak_hour,
                'peak_hour_events': hourly_activity[peak_hour]
            },
            'daily_patterns': {
                'activity_by_day': daily_activity,
                'peak_day': peak_day,
                'peak_day_events': daily_activity[peak_day]
            },
            'event_timing': {
                'purchase_peak_hour': hourly_by_event['transaction'].idxmax() if 'transaction' in hourly_by_event.columns else None,
                'browsing_peak_hour': hourly_by_event['view'].idxmax() if 'view' in hourly_by_event.columns else None
            }
        }
    
    def _analyze_item_performance_distribution(self, events_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze distribution of item performance"""
        
        # Item-level metrics
        item_metrics = events_df.groupby('itemid').agg({
            'event': 'count',
            'visitorid': 'nunique'
        }).rename(columns={'event': 'total_events', 'visitorid': 'unique_visitors'})
        
        # Event type breakdown by item
        item_events = events_df.groupby(['itemid', 'event']).size().unstack(fill_value=0)
        
        # Calculate performance metrics
        if 'view' in item_events.columns:
            item_metrics['views'] = item_events['view']
        if 'addtocart' in item_events.columns:
            item_metrics['addtocart'] = item_events['addtocart']
        if 'transaction' in item_events.columns:
            item_metrics['transactions'] = item_events['transaction']
        
        # Performance distribution
        performance_stats = {
            'total_items_analyzed': len(item_metrics),
            'items_with_views': len(item_metrics[item_metrics.get('views', 0) > 0]),
            'items_with_purchases': len(item_metrics[item_metrics.get('transactions', 0) > 0]),
            'top_10_percent_items_share': self._calculate_top_performers_share(item_metrics),
            'performance_distribution': {
                'high_performers': len(item_metrics[item_metrics.get('transactions', 0) >= item_metrics.get('transactions', pd.Series([0])).quantile(0.8)]),
                'medium_performers': len(item_metrics[(item_metrics.get('transactions', 0) >= item_metrics.get('transactions', pd.Series([0])).quantile(0.2)) & 
                                                    (item_metrics.get('transactions', 0) < item_metrics.get('transactions', pd.Series([0])).quantile(0.8))]),
                'low_performers': len(item_metrics[item_metrics.get('transactions', 0) < item_metrics.get('transactions', pd.Series([0])).quantile(0.2)])
            }
        }
        
        return performance_stats
    
    def _analyze_user_behavior_patterns(self, events_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze user behavior patterns"""
        
        # User-level metrics
        user_metrics = events_df.groupby('visitorid').agg({
            'event': 'count',
            'itemid': 'nunique'
        }).rename(columns={'event': 'total_events', 'itemid': 'unique_items_viewed'})
        
        # User engagement distribution
        engagement_stats = {
            'total_users': len(user_metrics),
            'avg_events_per_user': round(user_metrics['total_events'].mean(), 2),
            'avg_items_per_user': round(user_metrics['unique_items_viewed'].mean(), 2),
            'user_engagement_distribution': {
                'high_engagement': len(user_metrics[user_metrics['total_events'] >= user_metrics['total_events'].quantile(0.8)]),
                'medium_engagement': len(user_metrics[(user_metrics['total_events'] >= user_metrics['total_events'].quantile(0.2)) & 
                                                    (user_metrics['total_events'] < user_metrics['total_events'].quantile(0.8))]),
                'low_engagement': len(user_metrics[user_metrics['total_events'] < user_metrics['total_events'].quantile(0.2)])
            }
        }
        
        return engagement_stats
    
    def _calculate_top_performers_share(self, item_metrics: pd.DataFrame) -> float:
        """Calculate what percentage of total activity comes from top 10% of items"""
        if 'transactions' not in item_metrics.columns or len(item_metrics) == 0:
            return 0.0
        
        total_transactions = item_metrics['transactions'].sum()
        if total_transactions == 0:
            return 0.0
        
        top_10_percent_count = max(1, len(item_metrics) // 10)
        top_performers = item_metrics.nlargest(top_10_percent_count, 'transactions')
        top_performers_transactions = top_performers['transactions'].sum()
        
        return round((top_performers_transactions / total_transactions) * 100, 2)


class ModelInsightsAnalyzer:
    """Analyzes model performance and generates insights"""
    
    def analyze_model_performance_insights(self, model_results: Dict[str, Any], 
                                         ab_test_results: Dict[str, Any]) -> List[BusinessInsight]:
        """
        Add recommendation generation based on model results
        
        Args:
            model_results: Dictionary with model performance metrics
            ab_test_results: Dictionary with A/B test results
            
        Returns:
            List of BusinessInsight objects
        """
        logger.info("Analyzing model performance insights")
        
        insights = []
        
        # Model Performance Insight
        model_performance_insight = self._generate_model_performance_insight(model_results)
        if model_performance_insight:
            insights.append(model_performance_insight)
        
        # Feature Importance Insight
        feature_importance_insight = self._generate_feature_importance_insight(model_results)
        if feature_importance_insight:
            insights.append(feature_importance_insight)
        
        # A/B Test Performance Insight
        ab_performance_insight = self._generate_ab_test_insight(ab_test_results)
        if ab_performance_insight:
            insights.append(ab_performance_insight)
        
        # Business Impact Insight
        business_impact_insight = self._generate_business_impact_insight(ab_test_results)
        if business_impact_insight:
            insights.append(business_impact_insight)
        
        # Statistical Significance Insight
        statistical_insight = self._generate_statistical_significance_insight(ab_test_results)
        if statistical_insight:
            insights.append(statistical_insight)
        
        logger.info(f"Generated {len(insights)} model performance insights")
        return insights
    
    def _generate_model_performance_insight(self, model_results: Dict[str, Any]) -> Optional[BusinessInsight]:
        """Generate insight about model performance"""
        
        performance_metrics = model_results.get('performance_report', {}).get('evaluation_metrics', {})
        roc_auc = performance_metrics.get('roc_auc', 0)
        precision = performance_metrics.get('precision', 0)
        recall = performance_metrics.get('recall', 0)
        
        if roc_auc == 0:
            return None
        
        # Determine performance level and confidence
        if roc_auc >= 0.8:
            performance_level = "excellent"
            confidence = "High"
        elif roc_auc >= 0.7:
            performance_level = "good"
            confidence = "Medium"
        else:
            performance_level = "poor"
            confidence = "Low"
        
        return BusinessInsight(
            insight_type="model_performance",
            title=f"Model Performance: {performance_level.title()} Predictive Accuracy",
            description=f"The machine learning model demonstrates {performance_level} performance with an ROC-AUC score of {roc_auc:.3f}, "
                       f"precision of {precision:.3f}, and recall of {recall:.3f}. This indicates the model can effectively "
                       f"distinguish between high-performing and low-performing products.",
            quantified_impact={
                'roc_auc_score': round(roc_auc, 3),
                'precision_score': round(precision, 3),
                'recall_score': round(recall, 3),
                'performance_grade': model_results.get('performance_report', {}).get('performance_grade', 'Unknown')
            },
            confidence_level=confidence,
            recommendation=f"{'Deploy the model for production use' if roc_auc >= 0.7 else 'Improve model performance before deployment'}",
            supporting_data={
                'validation_passed': model_results.get('performance_report', {}).get('validation_passed', False),
                'model_type': model_results.get('model_type', 'Unknown')
            }
        )
    
    def _generate_feature_importance_insight(self, model_results: Dict[str, Any]) -> Optional[BusinessInsight]:
        """Generate insight about feature importance"""
        
        feature_importance = model_results.get('performance_report', {}).get('feature_importance', [])
        
        if not feature_importance:
            return None
        
        top_feature = feature_importance[0] if feature_importance else {}
        top_3_features = feature_importance[:3] if len(feature_importance) >= 3 else feature_importance
        
        # Calculate feature concentration
        total_importance = sum(f.get('importance', 0) for f in feature_importance)
        top_3_importance = sum(f.get('importance', 0) for f in top_3_features)
        concentration_ratio = (top_3_importance / total_importance * 100) if total_importance > 0 else 0
        
        return BusinessInsight(
            insight_type="feature_importance",
            title=f"Key Ranking Factor: {top_feature.get('feature', 'Unknown')}",
            description=f"The most influential factor in product ranking is '{top_feature.get('feature', 'Unknown')}' "
                       f"with an importance score of {top_feature.get('importance', 0):.3f}. "
                       f"The top 3 features account for {concentration_ratio:.1f}% of the model's decision-making process.",
            quantified_impact={
                'top_feature_importance': round(top_feature.get('importance', 0), 3),
                'top_3_concentration': round(concentration_ratio, 1),
                'total_features_analyzed': len(feature_importance)
            },
            confidence_level="High",
            recommendation=f"Focus optimization efforts on improving '{top_feature.get('feature', 'Unknown')}' "
                          f"as it has the highest impact on product ranking performance.",
            supporting_data={
                'top_3_features': [f.get('feature', 'Unknown') for f in top_3_features],
                'feature_importance_scores': [round(f.get('importance', 0), 3) for f in top_3_features]
            }
        )
    
    def _generate_ab_test_insight(self, ab_test_results: Dict[str, Any]) -> Optional[BusinessInsight]:
        """Generate insight about A/B test performance"""
        
        baseline_metrics = ab_test_results.get('baseline_metrics', {})
        ml_metrics = ab_test_results.get('ml_ranking_metrics', {})
        
        if not baseline_metrics or not ml_metrics:
            return None
        
        # Calculate improvements
        ctr_improvement = ((ml_metrics.get('click_through_rate', 0) - baseline_metrics.get('click_through_rate', 0)) / 
                          baseline_metrics.get('click_through_rate', 1) * 100) if baseline_metrics.get('click_through_rate', 0) > 0 else 0
        
        cvr_improvement = ((ml_metrics.get('conversion_rate', 0) - baseline_metrics.get('conversion_rate', 0)) / 
                          baseline_metrics.get('conversion_rate', 1) * 100) if baseline_metrics.get('conversion_rate', 0) > 0 else 0
        
        purchase_improvement = ((ml_metrics.get('purchase_rate', 0) - baseline_metrics.get('purchase_rate', 0)) / 
                               baseline_metrics.get('purchase_rate', 1) * 100) if baseline_metrics.get('purchase_rate', 0) > 0 else 0
        
        # Determine overall performance
        avg_improvement = (ctr_improvement + cvr_improvement + purchase_improvement) / 3
        
        return BusinessInsight(
            insight_type="ab_test_performance",
            title=f"ML Ranking Shows {avg_improvement:+.1f}% Average Performance Improvement",
            description=f"The ML-based ranking system outperforms the baseline ranking across key metrics: "
                       f"{ctr_improvement:+.1f}% improvement in click-through rate, "
                       f"{cvr_improvement:+.1f}% improvement in conversion rate, and "
                       f"{purchase_improvement:+.1f}% improvement in purchase rate.",
            quantified_impact={
                'ctr_improvement_percent': round(ctr_improvement, 1),
                'cvr_improvement_percent': round(cvr_improvement, 1),
                'purchase_rate_improvement_percent': round(purchase_improvement, 1),
                'average_improvement_percent': round(avg_improvement, 1)
            },
            confidence_level="High" if avg_improvement > 5 else "Medium" if avg_improvement > 0 else "Low",
            recommendation="Implement ML-based ranking system" if avg_improvement > 0 else "Review and improve ML model",
            supporting_data={
                'baseline_ctr': round(baseline_metrics.get('click_through_rate', 0), 4),
                'ml_ctr': round(ml_metrics.get('click_through_rate', 0), 4),
                'baseline_cvr': round(baseline_metrics.get('conversion_rate', 0), 4),
                'ml_cvr': round(ml_metrics.get('conversion_rate', 0), 4)
            }
        )
    
    def _generate_business_impact_insight(self, ab_test_results: Dict[str, Any]) -> Optional[BusinessInsight]:
        """Generate insight about business impact"""
        
        business_impact = ab_test_results.get('business_impact', {})
        revenue_impact = business_impact.get('revenue_impact', {})
        
        if not revenue_impact:
            return None
        
        revenue_lift = revenue_impact.get('revenue_lift_per_user', 0)
        revenue_lift_percent = revenue_impact.get('revenue_lift_percentage', 0)
        
        # Estimate annual impact (assuming user base)
        estimated_annual_users = 100000  # This should be configurable
        estimated_annual_impact = revenue_lift * estimated_annual_users
        
        return BusinessInsight(
            insight_type="business_impact",
            title=f"Revenue Impact: ${revenue_lift:+.2f} Per User ({revenue_lift_percent:+.1f}%)",
            description=f"The ML ranking system generates an additional ${revenue_lift:.2f} in revenue per user, "
                       f"representing a {revenue_lift_percent:.1f}% increase over the baseline system. "
                       f"With an estimated user base of {estimated_annual_users:,}, this could translate to "
                       f"${estimated_annual_impact:,.0f} in additional annual revenue.",
            quantified_impact={
                'revenue_lift_per_user': round(revenue_lift, 2),
                'revenue_lift_percentage': round(revenue_lift_percent, 1),
                'estimated_annual_impact': round(estimated_annual_impact, 0),
                'baseline_revenue_per_user': round(revenue_impact.get('baseline_revenue_per_user', 0), 2),
                'ml_revenue_per_user': round(revenue_impact.get('ml_revenue_per_user', 0), 2)
            },
            confidence_level="High" if abs(revenue_lift_percent) > 5 else "Medium",
            recommendation="Prioritize ML ranking implementation for revenue optimization" if revenue_lift > 0 else "Review pricing and conversion optimization",
            supporting_data={
                'avg_order_value': revenue_impact.get('avg_order_value', 50),
                'estimated_user_base': estimated_annual_users
            }
        )
    
    def _generate_statistical_significance_insight(self, ab_test_results: Dict[str, Any]) -> Optional[BusinessInsight]:
        """Generate insight about statistical significance"""
        
        significance_results = ab_test_results.get('statistical_significance', {})
        summary = significance_results.get('summary', {})
        
        if not summary:
            return None
        
        significant_tests = summary.get('significant_tests', 0)
        total_tests = summary.get('total_tests', 0)
        primary_significant = summary.get('primary_metric_significant', False)
        effect_size_met = summary.get('min_effect_size_met', False)
        
        confidence_level = ab_test_results.get('test_config', {}).get('confidence_level', 0.95)
        
        return BusinessInsight(
            insight_type="statistical_significance",
            title=f"Statistical Validation: {significant_tests}/{total_tests} Tests Significant",
            description=f"Out of {total_tests} statistical tests performed, {significant_tests} show significant results "
                       f"at the {confidence_level:.0%} confidence level. The primary metric (purchase rate) "
                       f"{'is' if primary_significant else 'is not'} statistically significant, and the effect size "
                       f"{'meets' if effect_size_met else 'does not meet'} the minimum threshold.",
            quantified_impact={
                'significant_tests_ratio': f"{significant_tests}/{total_tests}",
                'confidence_level': confidence_level,
                'primary_metric_significant': primary_significant,
                'minimum_effect_size_met': effect_size_met
            },
            confidence_level="High" if primary_significant and effect_size_met else "Medium" if primary_significant else "Low",
            recommendation="Results are statistically reliable for business decisions" if primary_significant else "Extend test duration or increase sample size",
            supporting_data={
                'purchase_rate_p_value': significance_results.get('purchase_count', {}).get('p_value', 1.0),
                'ctr_p_value': significance_results.get('click_through_rate', {}).get('p_value', 1.0),
                'cvr_p_value': significance_results.get('conversion_rate', {}).get('p_value', 1.0)
            }
        )


class InsightsReportGenerator:
    """Main class for generating comprehensive insights reports"""
    
    def __init__(self):
        self.event_analyzer = EventDataAnalyzer()
        self.model_analyzer = ModelInsightsAnalyzer()
    
    def create_automated_insights_report(self, events_df: pd.DataFrame,
                                       model_results: Dict[str, Any],
                                       ab_test_results: Dict[str, Any],
                                       report_config: Optional[Dict[str, Any]] = None) -> InsightsReport:
        """
        Create automated insights report generation with quantified business impact
        
        Args:
            events_df: DataFrame with event data
            model_results: Dictionary with model performance results
            ab_test_results: Dictionary with A/B test results
            report_config: Optional configuration for report generation
            
        Returns:
            InsightsReport object with comprehensive insights
        """
        logger.info("Creating automated insights report")
        
        # Generate unique report ID
        report_id = f"insights_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Analyze event data patterns
        event_analysis = self.event_analyzer.analyze_event_patterns(events_df)
        
        # Generate model performance insights
        model_insights = self.model_analyzer.analyze_model_performance_insights(model_results, ab_test_results)
        
        # Create executive summary
        executive_summary = self._create_executive_summary(event_analysis, model_insights, ab_test_results)
        
        # Compile performance metrics
        performance_metrics = self._compile_performance_metrics(model_results, ab_test_results, event_analysis)
        
        # Extract business impact
        business_impact = self._extract_business_impact(ab_test_results, event_analysis)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(model_insights, ab_test_results)
        
        # Create technical summary
        technical_summary = self._create_technical_summary(model_results, ab_test_results)
        
        # Assess data quality
        data_quality_assessment = self._assess_data_quality(events_df, model_results)
        
        report = InsightsReport(
            report_id=report_id,
            generation_timestamp=datetime.now().isoformat(),
            executive_summary=executive_summary,
            key_findings=model_insights,
            performance_metrics=performance_metrics,
            business_impact=business_impact,
            recommendations=recommendations,
            technical_summary=technical_summary,
            data_quality_assessment=data_quality_assessment
        )
        
        logger.info(f"Insights report generated successfully: {report_id}")
        return report
    
    def _create_executive_summary(self, event_analysis: Dict[str, Any], 
                                model_insights: List[BusinessInsight],
                                ab_test_results: Dict[str, Any]) -> str:
        """Create executive summary of key findings"""
        
        # Extract key metrics
        total_events = event_analysis.get('event_statistics', {}).get('total_events', 0)
        unique_visitors = event_analysis.get('event_statistics', {}).get('unique_visitors', 0)
        unique_items = event_analysis.get('event_statistics', {}).get('unique_items', 0)
        
        # Get primary business impact
        business_impact = ab_test_results.get('business_impact', {})
        revenue_impact = business_impact.get('revenue_impact', {})
        revenue_lift_percent = revenue_impact.get('revenue_lift_percentage', 0)
        
        # Get model performance
        model_performance_insight = next((insight for insight in model_insights 
                                        if insight.insight_type == 'model_performance'), None)
        model_grade = model_performance_insight.quantified_impact.get('performance_grade', 'Unknown') if model_performance_insight else 'Unknown'
        
        # Get statistical significance
        significance_summary = ab_test_results.get('statistical_significance', {}).get('summary', {})
        primary_significant = significance_summary.get('primary_metric_significant', False)
        
        summary = f"""
EXECUTIVE SUMMARY

This analysis evaluated a machine learning-based product ranking system using {total_events:,} behavioral events 
from {unique_visitors:,} unique visitors across {unique_items:,} products. 

KEY FINDINGS:
• Model Performance: {model_grade} predictive accuracy with statistically {'significant' if primary_significant else 'inconclusive'} results
• Business Impact: {revenue_lift_percent:+.1f}% revenue improvement per user compared to baseline ranking
• Recommendation: {'Implement ML ranking system' if revenue_lift_percent > 0 and primary_significant else 'Continue testing and optimization'}

The analysis demonstrates {'strong potential' if revenue_lift_percent > 5 else 'moderate potential' if revenue_lift_percent > 0 else 'limited potential'} 
for improving product discovery and conversion through intelligent ranking algorithms.
        """.strip()
        
        return summary
    
    def _compile_performance_metrics(self, model_results: Dict[str, Any],
                                   ab_test_results: Dict[str, Any],
                                   event_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Compile key performance metrics"""
        
        # Model metrics
        model_metrics = model_results.get('performance_report', {}).get('evaluation_metrics', {})
        
        # A/B test metrics
        baseline_metrics = ab_test_results.get('baseline_metrics', {})
        ml_metrics = ab_test_results.get('ml_ranking_metrics', {})
        
        # Event metrics
        conversion_funnel = event_analysis.get('conversion_funnel', {}).get('conversion_rates', {})
        
        return {
            'model_performance': {
                'roc_auc': round(model_metrics.get('roc_auc', 0), 3),
                'precision': round(model_metrics.get('precision', 0), 3),
                'recall': round(model_metrics.get('recall', 0), 3),
                'f1_score': round(model_metrics.get('f1_score', 0), 3)
            },
            'ranking_comparison': {
                'baseline_ctr': round(baseline_metrics.get('click_through_rate', 0), 4),
                'ml_ctr': round(ml_metrics.get('click_through_rate', 0), 4),
                'baseline_cvr': round(baseline_metrics.get('conversion_rate', 0), 4),
                'ml_cvr': round(ml_metrics.get('conversion_rate', 0), 4),
                'baseline_purchase_rate': round(baseline_metrics.get('purchase_rate', 0), 4),
                'ml_purchase_rate': round(ml_metrics.get('purchase_rate', 0), 4)
            },
            'conversion_funnel': {
                'view_to_cart_rate': conversion_funnel.get('view_to_cart_rate', 0),
                'cart_to_purchase_rate': conversion_funnel.get('cart_to_purchase_rate', 0),
                'overall_conversion_rate': conversion_funnel.get('view_to_purchase_rate', 0)
            }
        }
    
    def _extract_business_impact(self, ab_test_results: Dict[str, Any],
                               event_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Extract quantified business impact metrics"""
        
        business_impact = ab_test_results.get('business_impact', {})
        revenue_impact = business_impact.get('revenue_impact', {})
        metric_improvements = business_impact.get('metric_improvements', {})
        
        # Calculate potential scale impact
        total_events = event_analysis.get('event_statistics', {}).get('total_events', 0)
        unique_visitors = event_analysis.get('event_statistics', {}).get('unique_visitors', 0)
        
        return {
            'revenue_impact': {
                'per_user_lift': round(revenue_impact.get('revenue_lift_per_user', 0), 2),
                'percentage_lift': round(revenue_impact.get('revenue_lift_percentage', 0), 1),
                'baseline_revenue_per_user': round(revenue_impact.get('baseline_revenue_per_user', 0), 2),
                'ml_revenue_per_user': round(revenue_impact.get('ml_revenue_per_user', 0), 2)
            },
            'engagement_improvements': {
                'ctr_improvement': round(metric_improvements.get('click_through_rate', {}).get('relative_improvement', 0), 1),
                'cvr_improvement': round(metric_improvements.get('conversion_rate', {}).get('relative_improvement', 0), 1),
                'purchase_rate_improvement': round(metric_improvements.get('purchase_rate', {}).get('relative_improvement', 0), 1)
            },
            'scale_metrics': {
                'total_events_analyzed': total_events,
                'unique_visitors_analyzed': unique_visitors,
                'potential_monthly_impact': round(revenue_impact.get('revenue_lift_per_user', 0) * unique_visitors, 2)
            }
        }
    
    def _generate_recommendations(self, model_insights: List[BusinessInsight],
                                ab_test_results: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations"""
        
        recommendations = []
        
        # Extract recommendations from insights
        for insight in model_insights:
            if insight.recommendation and insight.recommendation not in recommendations:
                recommendations.append(insight.recommendation)
        
        # Add overall system recommendation
        overall_assessment = ab_test_results.get('business_impact', {}).get('overall_assessment', {})
        system_recommendation = overall_assessment.get('recommendation', '')
        if system_recommendation and system_recommendation not in recommendations:
            recommendations.append(system_recommendation)
        
        # Add specific technical recommendations
        model_performance = ab_test_results.get('business_impact', {}).get('metric_improvements', {})
        if model_performance.get('purchase_rate', {}).get('relative_improvement', 0) > 10:
            recommendations.append("Scale ML ranking system to all product categories for maximum impact")
        
        if len(recommendations) == 0:
            recommendations.append("Continue monitoring and optimization of the ranking system")
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    def _create_technical_summary(self, model_results: Dict[str, Any],
                                ab_test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create technical summary of the analysis"""
        
        return {
            'model_details': {
                'model_type': model_results.get('model_type', 'Unknown'),
                'model_name': model_results.get('model_name', 'Unknown'),
                'feature_count': len(model_results.get('performance_report', {}).get('feature_importance', [])),
                'validation_passed': model_results.get('performance_report', {}).get('validation_passed', False)
            },
            'test_configuration': {
                'users_per_group': ab_test_results.get('test_config', {}).get('num_users_per_group', 0),
                'items_per_user': ab_test_results.get('test_config', {}).get('items_per_user', 0),
                'confidence_level': ab_test_results.get('test_config', {}).get('confidence_level', 0.95),
                'simulation_days': ab_test_results.get('test_config', {}).get('simulation_days', 30)
            },
            'statistical_tests': {
                'total_tests_performed': ab_test_results.get('statistical_significance', {}).get('summary', {}).get('total_tests', 0),
                'significant_results': ab_test_results.get('statistical_significance', {}).get('summary', {}).get('significant_tests', 0),
                'primary_metric_significant': ab_test_results.get('statistical_significance', {}).get('summary', {}).get('primary_metric_significant', False)
            }
        }
    
    def _assess_data_quality(self, events_df: pd.DataFrame, model_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess data quality for the analysis"""
        
        # Basic data quality checks
        total_events = len(events_df)
        missing_values = events_df.isnull().sum().sum()
        duplicate_events = events_df.duplicated().sum()
        
        # Event type distribution
        event_types = events_df['event'].value_counts().to_dict()
        
        # Data completeness
        completeness_score = ((total_events - missing_values) / total_events * 100) if total_events > 0 else 0
        
        # Model data quality
        model_validation = model_results.get('performance_report', {}).get('validation_passed', False)
        
        quality_grade = "Excellent" if completeness_score >= 95 and model_validation else \
                       "Good" if completeness_score >= 90 else \
                       "Fair" if completeness_score >= 80 else "Poor"
        
        return {
            'data_completeness': {
                'total_events': total_events,
                'missing_values': int(missing_values),
                'duplicate_events': int(duplicate_events),
                'completeness_percentage': round(completeness_score, 1)
            },
            'event_distribution': event_types,
            'model_validation': {
                'validation_passed': model_validation,
                'data_sufficient_for_ml': total_events >= 1000  # Minimum threshold
            },
            'overall_quality_grade': quality_grade,
            'quality_issues': self._identify_quality_issues(events_df, completeness_score, model_validation)
        }
    
    def _identify_quality_issues(self, events_df: pd.DataFrame, 
                               completeness_score: float, 
                               model_validation: bool) -> List[str]:
        """Identify potential data quality issues"""
        
        issues = []
        
        if completeness_score < 90:
            issues.append(f"Data completeness below 90% ({completeness_score:.1f}%)")
        
        if not model_validation:
            issues.append("Model validation failed - may indicate data quality issues")
        
        if len(events_df) < 1000:
            issues.append("Limited data volume may affect analysis reliability")
        
        # Check event type balance
        event_counts = events_df['event'].value_counts()
        if 'transaction' in event_counts and event_counts['transaction'] < 10:
            issues.append("Very few transaction events may limit conversion analysis")
        
        return issues if issues else ["No significant data quality issues identified"]
    
    def export_insights_report(self, report: InsightsReport, 
                             output_format: str = "json",
                             filepath: Optional[str] = None) -> str:
        """
        Export insights report to file
        
        Args:
            report: InsightsReport object to export
            output_format: Format for export ("json", "txt", "html")
            filepath: Optional custom filepath
            
        Returns:
            Path to exported file
        """
        if filepath is None:
            filepath = f"{report.report_id}.{output_format}"
        
        logger.info(f"Exporting insights report to {filepath}")
        
        if output_format.lower() == "json":
            with open(filepath, 'w') as f:
                json.dump(report.to_dict(), f, indent=2, default=str)
        
        elif output_format.lower() == "txt":
            self._export_text_report(report, filepath)
        
        elif output_format.lower() == "html":
            self._export_html_report(report, filepath)
        
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
        
        logger.info(f"Insights report exported successfully to {filepath}")
        return filepath
    
    def _export_text_report(self, report: InsightsReport, filepath: str) -> None:
        """Export report as formatted text file"""
        
        with open(filepath, 'w') as f:
            f.write(f"SMART PRODUCT RE-RANKING INSIGHTS REPORT\n")
            f.write(f"Report ID: {report.report_id}\n")
            f.write(f"Generated: {report.generation_timestamp}\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-" * 20 + "\n")
            f.write(f"{report.executive_summary}\n\n")
            
            f.write("KEY FINDINGS\n")
            f.write("-" * 20 + "\n")
            for i, finding in enumerate(report.key_findings, 1):
                f.write(f"{i}. {finding.title}\n")
                f.write(f"   {finding.description}\n")
                f.write(f"   Confidence: {finding.confidence_level}\n")
                f.write(f"   Recommendation: {finding.recommendation}\n\n")
            
            f.write("PERFORMANCE METRICS\n")
            f.write("-" * 20 + "\n")
            for category, metrics in report.performance_metrics.items():
                f.write(f"{category.upper()}:\n")
                for metric, value in metrics.items():
                    f.write(f"  {metric}: {value}\n")
                f.write("\n")
            
            f.write("RECOMMENDATIONS\n")
            f.write("-" * 20 + "\n")
            for i, rec in enumerate(report.recommendations, 1):
                f.write(f"{i}. {rec}\n")
    
    def _export_html_report(self, report: InsightsReport, filepath: str) -> None:
        """Export report as HTML file"""
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Smart Product Re-Ranking Insights Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                .finding {{ background-color: #f9f9f9; padding: 15px; margin: 10px 0; border-left: 4px solid #007acc; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #e6f3ff; border-radius: 3px; }}
                .recommendation {{ background-color: #fff3cd; padding: 10px; margin: 5px 0; border-radius: 3px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Smart Product Re-Ranking Insights Report</h1>
                <p><strong>Report ID:</strong> {report.report_id}</p>
                <p><strong>Generated:</strong> {report.generation_timestamp}</p>
            </div>
            
            <div class="section">
                <h2>Executive Summary</h2>
                <p>{report.executive_summary.replace(chr(10), '<br>')}</p>
            </div>
            
            <div class="section">
                <h2>Key Findings</h2>
                {''.join([f'<div class="finding"><h3>{finding.title}</h3><p>{finding.description}</p><p><strong>Confidence:</strong> {finding.confidence_level}</p><p><strong>Recommendation:</strong> {finding.recommendation}</p></div>' for finding in report.key_findings])}
            </div>
            
            <div class="section">
                <h2>Recommendations</h2>
                <ol>
                    {''.join([f'<li class="recommendation">{rec}</li>' for rec in report.recommendations])}
                </ol>
            </div>
        </body>
        </html>
        """
        
        with open(filepath, 'w') as f:
            f.write(html_content)


# Convenience functions
def generate_comprehensive_insights(events_df: pd.DataFrame,
                                  model_results: Dict[str, Any],
                                  ab_test_results: Dict[str, Any]) -> InsightsReport:
    """
    Convenience function to generate comprehensive insights report
    
    Args:
        events_df: DataFrame with event data
        model_results: Dictionary with model performance results
        ab_test_results: Dictionary with A/B test results
        
    Returns:
        InsightsReport object
    """
    generator = InsightsReportGenerator()
    return generator.create_automated_insights_report(events_df, model_results, ab_test_results)


def create_business_summary(events_df: pd.DataFrame,
                          model_results: Dict[str, Any],
                          ab_test_results: Dict[str, Any],
                          output_file: str = "insights.txt") -> str:
    """
    Convenience function to create a concise business summary
    
    Args:
        events_df: DataFrame with event data
        model_results: Dictionary with model performance results
        ab_test_results: Dictionary with A/B test results
        output_file: Output file path
        
    Returns:
        Path to generated summary file
    """
    report = generate_comprehensive_insights(events_df, model_results, ab_test_results)
    generator = InsightsReportGenerator()
    return generator.export_insights_report(report, "txt", output_file)


if __name__ == "__main__":
    print("Insights Generation and Reporting module loaded successfully!")
    print("Available classes:")
    print("- EventDataAnalyzer: Analyzes event data patterns")
    print("- ModelInsightsAnalyzer: Analyzes model performance insights")
    print("- InsightsReportGenerator: Generates comprehensive insights reports")
    print("\nAvailable functions:")
    print("- generate_comprehensive_insights(): Generate complete insights report")
    print("- create_business_summary(): Create concise business summary")