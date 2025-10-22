"""
A/B Testing simulation for Smart Product Re-Ranking System
Implements baseline ranking system, ML ranking system, and comprehensive A/B simulation
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List, Optional, Any
import logging
from dataclasses import dataclass
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ABTestConfig:
    """Configuration for A/B testing simulation"""
    num_users_per_group: int = 1000
    items_per_user: int = 10
    simulation_days: int = 30
    confidence_level: float = 0.95
    minimum_effect_size: float = 0.05  # 5% minimum improvement
    random_state: int = 42


@dataclass
class ABTestResults:
    """Results from A/B test simulation"""
    test_config: ABTestConfig
    baseline_metrics: Dict[str, float]
    ml_ranking_metrics: Dict[str, float]
    statistical_significance: Dict[str, Any]
    business_impact: Dict[str, Any]
    simulation_data: pd.DataFrame
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'test_config': self.test_config.__dict__,
            'baseline_metrics': self.baseline_metrics,
            'ml_ranking_metrics': self.ml_ranking_metrics,
            'statistical_significance': self.statistical_significance,
            'business_impact': self.business_impact,
            'simulation_summary': {
                'total_users': len(self.simulation_data['user_id'].unique()),
                'total_interactions': len(self.simulation_data),
                'test_duration_days': self.test_config.simulation_days
            }
        }


class BaselineRankingSystem:
    """Implements baseline ranking system (by view count) for comparison"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        np.random.seed(random_state)
    
    def rank_by_view_count(self, products_df: pd.DataFrame) -> pd.DataFrame:
        """
        Implement baseline ranking system (by view count) for comparison
        
        Args:
            products_df: DataFrame with product data including view counts
            
        Returns:
            DataFrame with products ranked by view count
        """
        logger.info("Ranking products by view count (baseline method)")
        
        # Ensure required columns exist
        if 'view_count' not in products_df.columns:
            if 'total_impressions' in products_df.columns:
                products_df = products_df.copy()
                products_df['view_count'] = products_df['total_impressions']
            else:
                raise ValueError("Products DataFrame must contain 'view_count' or 'total_impressions' column")
        
        # Sort by view count (descending)
        ranked_products = products_df.sort_values('view_count', ascending=False).copy()
        
        # Add ranking information
        ranked_products['baseline_rank'] = range(1, len(ranked_products) + 1)
        ranked_products['baseline_score'] = ranked_products['view_count'] / ranked_products['view_count'].max()
        
        logger.info(f"Ranked {len(ranked_products)} products by view count")
        return ranked_products
    
    def rank_by_popularity_metrics(self, products_df: pd.DataFrame, 
                                 metric: str = 'view_count') -> pd.DataFrame:
        """
        Rank products by various popularity metrics
        
        Args:
            products_df: DataFrame with product data
            metric: Metric to use for ranking ('view_count', 'addtocart_count', 'transaction_count')
            
        Returns:
            DataFrame with products ranked by specified metric
        """
        logger.info(f"Ranking products by {metric}")
        
        if metric not in products_df.columns:
            raise ValueError(f"Metric '{metric}' not found in products DataFrame")
        
        # Sort by specified metric (descending)
        ranked_products = products_df.sort_values(metric, ascending=False).copy()
        
        # Add ranking information
        ranked_products[f'{metric}_rank'] = range(1, len(ranked_products) + 1)
        ranked_products[f'{metric}_score'] = ranked_products[metric] / ranked_products[metric].max()
        
        return ranked_products


class MLRankingSystem:
    """Implements new ranking system (by predicted relevance score)"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        np.random.seed(random_state)
    
    def rank_by_ml_predictions(self, products_df: pd.DataFrame, 
                             prediction_column: str = 'relevance_score') -> pd.DataFrame:
        """
        Implement new ranking system (by predicted relevance score)
        
        Args:
            products_df: DataFrame with product data including ML predictions
            prediction_column: Column name containing ML relevance scores
            
        Returns:
            DataFrame with products ranked by ML predictions
        """
        logger.info("Ranking products by ML predictions (new method)")
        
        if prediction_column not in products_df.columns:
            raise ValueError(f"Prediction column '{prediction_column}' not found in products DataFrame")
        
        # Sort by ML predictions (descending)
        ranked_products = products_df.sort_values(prediction_column, ascending=False).copy()
        
        # Add ranking information
        ranked_products['ml_rank'] = range(1, len(ranked_products) + 1)
        ranked_products['ml_score'] = ranked_products[prediction_column]
        
        logger.info(f"Ranked {len(ranked_products)} products by ML predictions")
        return ranked_products
    
    def combine_ml_and_baseline_features(self, products_df: pd.DataFrame,
                                       ml_weight: float = 0.7,
                                       baseline_weight: float = 0.3) -> pd.DataFrame:
        """
        Combine ML predictions with baseline features for hybrid ranking
        
        Args:
            products_df: DataFrame with both ML scores and baseline metrics
            ml_weight: Weight for ML predictions
            baseline_weight: Weight for baseline metrics
            
        Returns:
            DataFrame with hybrid ranking scores
        """
        logger.info(f"Creating hybrid ranking (ML: {ml_weight}, Baseline: {baseline_weight})")
        
        products_df = products_df.copy()
        
        # Normalize scores to 0-1 range
        if 'relevance_score' in products_df.columns:
            ml_normalized = products_df['relevance_score']
        else:
            raise ValueError("ML relevance_score column not found")
        
        if 'view_count' in products_df.columns:
            baseline_normalized = products_df['view_count'] / products_df['view_count'].max()
        else:
            baseline_normalized = 0
        
        # Calculate hybrid score
        products_df['hybrid_score'] = (ml_weight * ml_normalized + 
                                     baseline_weight * baseline_normalized)
        
        # Rank by hybrid score
        products_df = products_df.sort_values('hybrid_score', ascending=False)
        products_df['hybrid_rank'] = range(1, len(products_df) + 1)
        
        return products_df


class RankingABTest:
    """Enhanced A/B test simulation for comparing ranking methods with statistical significance testing"""
    
    def __init__(self, config: Optional[ABTestConfig] = None):
        self.config = config or ABTestConfig()
        self.random_state = self.config.random_state
        np.random.seed(self.random_state)
        self.baseline_system = BaselineRankingSystem(self.random_state)
        self.ml_system = MLRankingSystem(self.random_state)
    
    def simulate_user_behavior(self, ranked_products: pd.DataFrame, 
                             ranking_method: str,
                             num_users: Optional[int] = None,
                             items_per_user: Optional[int] = None) -> pd.DataFrame:
        """Simulate user behavior based on product ranking"""
        
        # Use config defaults if not provided
        if num_users is None:
            num_users = self.config.num_users_per_group
        if items_per_user is None:
            items_per_user = self.config.items_per_user
        
        logger.info(f"Simulating user behavior for {ranking_method} with {num_users} users")
        
        results = []
        
        for user_id in range(num_users):
            # Simulate user seeing top N items based on ranking
            if ranking_method == 'baseline':
                # Baseline: rank by view count or total impressions
                if 'view_count' in ranked_products.columns:
                    user_items = ranked_products.nlargest(items_per_user, 'view_count')
                elif 'total_impressions' in ranked_products.columns:
                    user_items = ranked_products.nlargest(items_per_user, 'total_impressions')
                else:
                    # Fallback to first N items if no ranking column available
                    user_items = ranked_products.head(items_per_user)
            elif ranking_method == 'ml_ranking':
                # ML ranking: use model predictions
                if 'relevance_score' in ranked_products.columns:
                    user_items = ranked_products.nlargest(items_per_user, 'relevance_score')
                elif 'ml_score' in ranked_products.columns:
                    user_items = ranked_products.nlargest(items_per_user, 'ml_score')
                else:
                    raise ValueError("ML ranking requires 'relevance_score' or 'ml_score' column")
            else:
                raise ValueError(f"Unknown ranking method: {ranking_method}")
            
            # Simulate user interactions based on historical performance
            position = 1
            for _, item in user_items.iterrows():
                # Calculate interaction probabilities based on historical data
                if 'view_count' in item:
                    view_prob = min(1.0, item['view_count'] / ranked_products['view_count'].max())
                elif 'total_impressions' in item:
                    view_prob = min(1.0, item['total_impressions'] / ranked_products['total_impressions'].max())
                else:
                    view_prob = 0.8  # Default view probability
                
                # Position bias - items higher in ranking are more likely to be viewed
                position_bias = max(0.1, 1.0 - (position - 1) * 0.1)
                view_prob *= position_bias
                
                # CTR and CVR from historical data (with some noise)
                base_ctr = item.get('addtocart_rate', item.get('ctr', 0.05))  # Default 5% CTR
                base_cvr = item.get('conversion_rate', item.get('cvr', 0.02))  # Default 2% CVR
                
                # ML ranking should improve performance
                if ranking_method == 'ml_ranking':
                    ml_boost = item.get('relevance_score', item.get('ml_score', 0.5))
                    base_ctr *= (1 + ml_boost * 0.3)  # Up to 30% improvement
                    base_cvr *= (1 + ml_boost * 0.2)  # Up to 20% improvement
                
                # Add some randomness to simulate real user behavior
                ctr_noise = np.random.normal(0, 0.05)  # 5% noise
                cvr_noise = np.random.normal(0, 0.03)  # 3% noise
                
                actual_ctr = max(0, min(1, base_ctr + ctr_noise))
                actual_cvr = max(0, min(1, base_cvr + cvr_noise))
                
                # Simulate events
                viewed = np.random.random() < view_prob
                if viewed:
                    clicked = np.random.random() < actual_ctr
                    if clicked:
                        purchased = np.random.random() < actual_cvr
                    else:
                        purchased = False
                else:
                    clicked = False
                    purchased = False
                
                # Get item ID
                item_id = item.get('itemid', item.get('product_id', f'item_{position}'))
                
                results.append({
                    'user_id': user_id,
                    'product_id': item_id,
                    'ranking_method': ranking_method,
                    'position': position,
                    'viewed': viewed,
                    'clicked': clicked,
                    'purchased': purchased,
                    'historical_ctr': base_ctr,
                    'historical_cvr': base_cvr,
                    'ml_score': item.get('relevance_score', item.get('ml_score', 0)),
                    'view_prob': view_prob,
                    'actual_ctr': actual_ctr,
                    'actual_cvr': actual_cvr
                })
                
                position += 1
        
        return pd.DataFrame(results)
    
    def build_ab_simulation_system(self, products_df: pd.DataFrame) -> ABTestResults:
        """
        Build A/B simulation system with baseline and ML ranking comparison
        
        Args:
            products_df: DataFrame with product data and ML predictions
            
        Returns:
            ABTestResults object with comprehensive test results
        """
        logger.info("Building comprehensive A/B simulation system")
        logger.info(f"Test configuration: {self.config.num_users_per_group} users per group, "
                   f"{self.config.items_per_user} items per user")
        
        # Prepare baseline ranking
        baseline_products = self.baseline_system.rank_by_view_count(products_df)
        
        # Prepare ML ranking
        ml_products = self.ml_system.rank_by_ml_predictions(products_df)
        
        # Group A: Baseline ranking (by view count)
        logger.info("Simulating Group A (Baseline Ranking)...")
        group_a_results = self.simulate_user_behavior(baseline_products, 'baseline')
        
        # Group B: ML-based ranking
        logger.info("Simulating Group B (ML Ranking)...")
        group_b_results = self.simulate_user_behavior(ml_products, 'ml_ranking')
        
        # Combine results
        all_results = pd.concat([group_a_results, group_b_results], ignore_index=True)
        
        # Calculate metrics by group
        baseline_metrics, ml_metrics = self.create_conversion_rate_comparison(all_results)
        
        # Statistical significance testing
        significance_results = self.create_statistical_significance_testing(all_results)
        
        # Business impact analysis
        business_impact = self.generate_business_impact_reports(
            baseline_metrics, ml_metrics, significance_results
        )
        
        # Create comprehensive results
        ab_results = ABTestResults(
            test_config=self.config,
            baseline_metrics=baseline_metrics,
            ml_ranking_metrics=ml_metrics,
            statistical_significance=significance_results,
            business_impact=business_impact,
            simulation_data=all_results
        )
        
        logger.info("A/B simulation system completed successfully")
        return ab_results
    
    def create_conversion_rate_comparison(self, results_df: pd.DataFrame) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Create conversion rate comparison and statistical significance testing
        
        Args:
            results_df: DataFrame with simulation results
            
        Returns:
            Tuple of (baseline_metrics, ml_ranking_metrics)
        """
        logger.info("Creating conversion rate comparison between ranking methods")
        
        # Calculate metrics by group
        metrics_by_group = results_df.groupby('ranking_method').agg({
            'viewed': ['count', 'sum'],
            'clicked': 'sum',
            'purchased': 'sum',
            'user_id': 'nunique'
        }).round(6)
        
        # Flatten column names
        metrics_by_group.columns = ['total_interactions', 'total_views', 'total_clicks', 'total_purchases', 'unique_users']
        
        # Calculate rates
        metrics_by_group['view_rate'] = metrics_by_group['total_views'] / metrics_by_group['total_interactions']
        metrics_by_group['click_through_rate'] = metrics_by_group['total_clicks'] / metrics_by_group['total_views']
        metrics_by_group['conversion_rate'] = metrics_by_group['total_purchases'] / metrics_by_group['total_clicks']
        metrics_by_group['purchase_rate'] = metrics_by_group['total_purchases'] / metrics_by_group['total_views']
        
        # Per-user metrics
        metrics_by_group['avg_views_per_user'] = metrics_by_group['total_views'] / metrics_by_group['unique_users']
        metrics_by_group['avg_clicks_per_user'] = metrics_by_group['total_clicks'] / metrics_by_group['unique_users']
        metrics_by_group['avg_purchases_per_user'] = metrics_by_group['total_purchases'] / metrics_by_group['unique_users']
        
        # Handle division by zero
        metrics_by_group = metrics_by_group.fillna(0)
        
        # Extract metrics for each group
        baseline_metrics = metrics_by_group.loc['baseline'].to_dict()
        ml_metrics = metrics_by_group.loc['ml_ranking'].to_dict()
        
        logger.info(f"Baseline CTR: {baseline_metrics['click_through_rate']:.4f}, "
                   f"ML CTR: {ml_metrics['click_through_rate']:.4f}")
        logger.info(f"Baseline CVR: {baseline_metrics['conversion_rate']:.4f}, "
                   f"ML CVR: {ml_metrics['conversion_rate']:.4f}")
        
        return baseline_metrics, ml_metrics
    
    def create_statistical_significance_testing(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Create statistical significance testing for A/B test results
        
        Args:
            results_df: DataFrame with simulation results
            
        Returns:
            Dictionary with statistical test results
        """
        logger.info("Performing statistical significance testing")
        
        # Separate groups
        baseline_group = results_df[results_df['ranking_method'] == 'baseline']
        ml_group = results_df[results_df['ranking_method'] == 'ml_ranking']
        
        # Aggregate by user for proper statistical testing
        baseline_user_metrics = baseline_group.groupby('user_id').agg({
            'viewed': 'sum',
            'clicked': 'sum',
            'purchased': 'sum'
        })
        
        ml_user_metrics = ml_group.groupby('user_id').agg({
            'viewed': 'sum',
            'clicked': 'sum',
            'purchased': 'sum'
        })
        
        # Calculate user-level rates
        baseline_user_metrics['ctr'] = baseline_user_metrics['clicked'] / baseline_user_metrics['viewed']
        baseline_user_metrics['cvr'] = baseline_user_metrics['purchased'] / baseline_user_metrics['clicked']
        baseline_user_metrics['purchase_rate'] = baseline_user_metrics['purchased'] / baseline_user_metrics['viewed']
        
        ml_user_metrics['ctr'] = ml_user_metrics['clicked'] / ml_user_metrics['viewed']
        ml_user_metrics['cvr'] = ml_user_metrics['purchased'] / ml_user_metrics['clicked']
        ml_user_metrics['purchase_rate'] = ml_user_metrics['purchased'] / ml_user_metrics['viewed']
        
        # Handle division by zero
        baseline_user_metrics = baseline_user_metrics.fillna(0)
        ml_user_metrics = ml_user_metrics.fillna(0)
        
        # Statistical tests
        tests = {}
        alpha = 1 - self.config.confidence_level
        
        # Test purchase rates (primary metric)
        baseline_purchases = baseline_user_metrics['purchased']
        ml_purchases = ml_user_metrics['purchased']
        
        t_stat, p_value = stats.ttest_ind(baseline_purchases, ml_purchases)
        effect_size = (ml_purchases.mean() - baseline_purchases.mean()) / baseline_purchases.std()
        
        tests['purchase_count'] = {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant': p_value < alpha,
            'baseline_mean': float(baseline_purchases.mean()),
            'ml_mean': float(ml_purchases.mean()),
            'relative_improvement': float((ml_purchases.mean() - baseline_purchases.mean()) / baseline_purchases.mean() * 100) if baseline_purchases.mean() > 0 else 0,
            'effect_size': float(effect_size),
            'confidence_level': self.config.confidence_level
        }
        
        # Test click-through rates
        baseline_ctr = baseline_user_metrics['ctr']
        ml_ctr = ml_user_metrics['ctr']
        
        t_stat, p_value = stats.ttest_ind(baseline_ctr, ml_ctr)
        effect_size = (ml_ctr.mean() - baseline_ctr.mean()) / baseline_ctr.std() if baseline_ctr.std() > 0 else 0
        
        tests['click_through_rate'] = {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant': p_value < alpha,
            'baseline_mean': float(baseline_ctr.mean()),
            'ml_mean': float(ml_ctr.mean()),
            'relative_improvement': float((ml_ctr.mean() - baseline_ctr.mean()) / baseline_ctr.mean() * 100) if baseline_ctr.mean() > 0 else 0,
            'effect_size': float(effect_size),
            'confidence_level': self.config.confidence_level
        }
        
        # Test conversion rates
        baseline_cvr = baseline_user_metrics['cvr']
        ml_cvr = ml_user_metrics['cvr']
        
        t_stat, p_value = stats.ttest_ind(baseline_cvr, ml_cvr)
        effect_size = (ml_cvr.mean() - baseline_cvr.mean()) / baseline_cvr.std() if baseline_cvr.std() > 0 else 0
        
        tests['conversion_rate'] = {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant': p_value < alpha,
            'baseline_mean': float(baseline_cvr.mean()),
            'ml_mean': float(ml_cvr.mean()),
            'relative_improvement': float((ml_cvr.mean() - baseline_cvr.mean()) / baseline_cvr.mean() * 100) if baseline_cvr.mean() > 0 else 0,
            'effect_size': float(effect_size),
            'confidence_level': self.config.confidence_level
        }
        
        # Overall test summary
        significant_tests = sum(1 for test in tests.values() if test['significant'])
        
        tests['summary'] = {
            'total_tests': len(tests) - 1,  # Exclude summary itself
            'significant_tests': significant_tests,
            'overall_significant': significant_tests > 0,
            'primary_metric_significant': tests['purchase_count']['significant'],
            'min_effect_size_met': abs(tests['purchase_count']['relative_improvement']) >= (self.config.minimum_effect_size * 100)
        }
        
        logger.info(f"Statistical testing completed: {significant_tests}/{len(tests)-1} tests significant")
        return tests
    
    def generate_business_impact_reports(self, baseline_metrics: Dict[str, float],
                                       ml_metrics: Dict[str, float],
                                       significance_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate quantified improvement metrics and business impact reports
        
        Args:
            baseline_metrics: Baseline ranking metrics
            ml_metrics: ML ranking metrics
            significance_results: Statistical significance test results
            
        Returns:
            Dictionary with business impact analysis
        """
        logger.info("Generating business impact reports")
        
        # Calculate improvements
        improvements = {}
        
        # Key metrics improvements
        for metric in ['click_through_rate', 'conversion_rate', 'purchase_rate']:
            baseline_value = baseline_metrics.get(metric, 0)
            ml_value = ml_metrics.get(metric, 0)
            
            if baseline_value > 0:
                absolute_improvement = ml_value - baseline_value
                relative_improvement = (absolute_improvement / baseline_value) * 100
            else:
                absolute_improvement = ml_value
                relative_improvement = 0
            
            improvements[metric] = {
                'baseline_value': baseline_value,
                'ml_value': ml_value,
                'absolute_improvement': absolute_improvement,
                'relative_improvement': relative_improvement,
                'is_significant': significance_results.get(metric, {}).get('significant', False)
            }
        
        # Revenue impact estimation (assuming average order value)
        avg_order_value = 50.0  # Placeholder - should be configurable
        baseline_revenue_per_user = baseline_metrics.get('avg_purchases_per_user', 0) * avg_order_value
        ml_revenue_per_user = ml_metrics.get('avg_purchases_per_user', 0) * avg_order_value
        
        revenue_impact = {
            'avg_order_value': avg_order_value,
            'baseline_revenue_per_user': baseline_revenue_per_user,
            'ml_revenue_per_user': ml_revenue_per_user,
            'revenue_lift_per_user': ml_revenue_per_user - baseline_revenue_per_user,
            'revenue_lift_percentage': ((ml_revenue_per_user - baseline_revenue_per_user) / baseline_revenue_per_user * 100) if baseline_revenue_per_user > 0 else 0
        }
        
        # Engagement improvements
        engagement_impact = {
            'baseline_engagement': {
                'avg_views_per_user': baseline_metrics.get('avg_views_per_user', 0),
                'avg_clicks_per_user': baseline_metrics.get('avg_clicks_per_user', 0),
                'avg_purchases_per_user': baseline_metrics.get('avg_purchases_per_user', 0)
            },
            'ml_engagement': {
                'avg_views_per_user': ml_metrics.get('avg_views_per_user', 0),
                'avg_clicks_per_user': ml_metrics.get('avg_clicks_per_user', 0),
                'avg_purchases_per_user': ml_metrics.get('avg_purchases_per_user', 0)
            }
        }
        
        # Overall business impact assessment
        primary_metric_improvement = improvements.get('purchase_rate', {}).get('relative_improvement', 0)
        is_statistically_significant = significance_results.get('summary', {}).get('primary_metric_significant', False)
        meets_minimum_effect = significance_results.get('summary', {}).get('min_effect_size_met', False)
        
        business_recommendation = self._generate_business_recommendation(
            primary_metric_improvement, is_statistically_significant, meets_minimum_effect
        )
        
        business_impact = {
            'metric_improvements': improvements,
            'revenue_impact': revenue_impact,
            'engagement_impact': engagement_impact,
            'overall_assessment': {
                'primary_metric_improvement': primary_metric_improvement,
                'is_statistically_significant': is_statistically_significant,
                'meets_minimum_effect_size': meets_minimum_effect,
                'recommendation': business_recommendation,
                'confidence_level': self.config.confidence_level
            },
            'test_summary': {
                'test_duration_days': self.config.simulation_days,
                'users_per_group': self.config.num_users_per_group,
                'total_users_tested': self.config.num_users_per_group * 2,
                'items_per_user': self.config.items_per_user
            }
        }
        
        logger.info(f"Business impact analysis completed - Primary improvement: {primary_metric_improvement:.2f}%")
        return business_impact
    
    def _generate_business_recommendation(self, improvement: float, 
                                        is_significant: bool, 
                                        meets_minimum_effect: bool) -> str:
        """Generate business recommendation based on test results"""
        if is_significant and meets_minimum_effect and improvement > 0:
            return "RECOMMEND IMPLEMENTATION - ML ranking shows significant positive impact"
        elif is_significant and improvement > 0:
            return "CONSIDER IMPLEMENTATION - Significant improvement but below minimum effect size"
        elif improvement > 0 and meets_minimum_effect:
            return "EXTEND TEST - Good improvement but not statistically significant yet"
        elif improvement > 0:
            return "CONTINUE MONITORING - Positive trend but needs more data"
        elif improvement < 0 and is_significant:
            return "DO NOT IMPLEMENT - ML ranking shows significant negative impact"
        else:
            return "INCONCLUSIVE - No clear benefit demonstrated"
    
    def plot_ab_results(self, ab_results: ABTestResults) -> None:
        """Plot comprehensive A/B test results"""
        
        logger.info("Generating A/B test result visualizations")
        
        # Prepare data for plotting
        baseline_metrics = ab_results.baseline_metrics
        ml_metrics = ab_results.ml_ranking_metrics
        
        metrics_comparison = pd.DataFrame({
            'Baseline': [baseline_metrics['click_through_rate'], baseline_metrics['conversion_rate'], 
                        baseline_metrics['purchase_rate'], baseline_metrics['avg_purchases_per_user']],
            'ML Ranking': [ml_metrics['click_through_rate'], ml_metrics['conversion_rate'], 
                          ml_metrics['purchase_rate'], ml_metrics['avg_purchases_per_user']],
            'Metric': ['Click-Through Rate', 'Conversion Rate', 'Purchase Rate', 'Purchases per User']
        })
        
        # Create comprehensive visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('A/B Test Results: Baseline vs ML Ranking', fontsize=16, fontweight='bold')
        
        # Plot 1: Metric Comparison
        ax1 = axes[0, 0]
        x_pos = np.arange(len(metrics_comparison))
        width = 0.35
        
        ax1.bar(x_pos - width/2, metrics_comparison['Baseline'], width, label='Baseline', alpha=0.8, color='steelblue')
        ax1.bar(x_pos + width/2, metrics_comparison['ML Ranking'], width, label='ML Ranking', alpha=0.8, color='orange')
        
        ax1.set_xlabel('Metrics')
        ax1.set_ylabel('Rate')
        ax1.set_title('Key Metrics Comparison')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(metrics_comparison['Metric'], rotation=45, ha='right')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # Plot 2: Statistical Significance
        ax2 = axes[0, 1]
        significance_data = []
        for metric, result in ab_results.statistical_significance.items():
            if metric != 'summary':
                significance_data.append({
                    'Metric': metric.replace('_', ' ').title(),
                    'P-Value': result['p_value'],
                    'Significant': result['significant']
                })
        
        sig_df = pd.DataFrame(significance_data)
        colors = ['green' if sig else 'red' for sig in sig_df['Significant']]
        
        bars = ax2.bar(sig_df['Metric'], sig_df['P-Value'], color=colors, alpha=0.7)
        ax2.axhline(y=0.05, color='red', linestyle='--', label='Significance Threshold (0.05)')
        ax2.set_ylabel('P-Value')
        ax2.set_title('Statistical Significance Tests')
        ax2.set_xticklabels(sig_df['Metric'], rotation=45, ha='right')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        # Plot 3: Improvement Percentages
        ax3 = axes[0, 2]
        improvements = []
        for metric, data in ab_results.business_impact['metric_improvements'].items():
            improvements.append({
                'Metric': metric.replace('_', ' ').title(),
                'Improvement': data['relative_improvement']
            })
        
        imp_df = pd.DataFrame(improvements)
        colors = ['green' if imp > 0 else 'red' for imp in imp_df['Improvement']]
        
        ax3.bar(imp_df['Metric'], imp_df['Improvement'], color=colors, alpha=0.7)
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax3.set_ylabel('Improvement (%)')
        ax3.set_title('Relative Improvements')
        ax3.set_xticklabels(imp_df['Metric'], rotation=45, ha='right')
        ax3.grid(axis='y', alpha=0.3)
        
        # Plot 4: Position Analysis
        ax4 = axes[1, 0]
        position_metrics = ab_results.simulation_data.groupby(['ranking_method', 'position']).agg({
            'clicked': 'mean',
            'purchased': 'mean'
        }).reset_index()
        
        for method in ['baseline', 'ml_ranking']:
            method_data = position_metrics[position_metrics['ranking_method'] == method]
            ax4.plot(method_data['position'], method_data['clicked'], 
                    marker='o', label=f'{method.replace("_", " ").title()} - Clicks')
        
        ax4.set_xlabel('Position in Ranking')
        ax4.set_ylabel('Click Rate')
        ax4.set_title('Click Rate by Position')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Revenue Impact
        ax5 = axes[1, 1]
        revenue_data = ab_results.business_impact['revenue_impact']
        
        revenue_comparison = ['Baseline', 'ML Ranking']
        revenue_values = [revenue_data['baseline_revenue_per_user'], revenue_data['ml_revenue_per_user']]
        
        bars = ax5.bar(revenue_comparison, revenue_values, color=['steelblue', 'orange'], alpha=0.8)
        ax5.set_ylabel('Revenue per User ($)')
        ax5.set_title('Revenue Impact per User')
        ax5.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, revenue_values):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    f'${value:.2f}', ha='center', va='bottom')
        
        # Plot 6: Test Summary
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        # Create summary text
        summary_text = f"""
Test Summary:
• Users per group: {ab_results.test_config.num_users_per_group:,}
• Items per user: {ab_results.test_config.items_per_user}
• Test duration: {ab_results.test_config.simulation_days} days
• Confidence level: {ab_results.test_config.confidence_level:.0%}

Primary Results:
• Purchase rate improvement: {ab_results.business_impact['metric_improvements']['purchase_rate']['relative_improvement']:.2f}%
• Statistically significant: {ab_results.statistical_significance['summary']['primary_metric_significant']}
• Revenue lift per user: ${ab_results.business_impact['revenue_impact']['revenue_lift_per_user']:.2f}

Recommendation:
{ab_results.business_impact['overall_assessment']['recommendation']}
        """
        
        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
        
        logger.info("A/B test visualizations generated successfully")
    
    def plot_position_analysis(self, results_df: pd.DataFrame) -> plt.Figure:
        """
        Analyze performance by position in ranking
        
        Args:
            results_df: DataFrame with simulation results
            
        Returns:
            Matplotlib figure object
        """
        logger.info("Creating position analysis visualization")
        
        position_metrics = results_df.groupby(['ranking_method', 'position']).agg({
            'viewed': 'mean',
            'clicked': 'mean',
            'purchased': 'mean'
        }).reset_index()
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Performance Analysis by Position in Ranking', fontsize=16, fontweight='bold')
        
        # View rate by position
        sns.lineplot(data=position_metrics, x='position', y='viewed', 
                    hue='ranking_method', marker='o', ax=axes[0], linewidth=2, markersize=8)
        axes[0].set_title('View Rate by Position')
        axes[0].set_ylabel('View Rate')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(title='Ranking Method')
        
        # Click rate by position
        sns.lineplot(data=position_metrics, x='position', y='clicked', 
                    hue='ranking_method', marker='o', ax=axes[1], linewidth=2, markersize=8)
        axes[1].set_title('Click Rate by Position')
        axes[1].set_ylabel('Click Rate')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(title='Ranking Method')
        
        # Purchase rate by position
        sns.lineplot(data=position_metrics, x='position', y='purchased', 
                    hue='ranking_method', marker='o', ax=axes[2], linewidth=2, markersize=8)
        axes[2].set_title('Purchase Rate by Position')
        axes[2].set_ylabel('Purchase Rate')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend(title='Ranking Method')
        
        plt.tight_layout()
        return fig
    
    def export_ab_results(self, ab_results: ABTestResults, filepath: str) -> None:
        """
        Export A/B test results to JSON file
        
        Args:
            ab_results: ABTestResults object
            filepath: Path to save the results
        """
        logger.info(f"Exporting A/B test results to {filepath}")
        
        import json
        
        # Convert results to dictionary
        results_dict = ab_results.to_dict()
        
        # Save to JSON file
        with open(filepath, 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)
        
        logger.info(f"A/B test results exported successfully to {filepath}")


# Convenience functions
def run_ab_test_simulation(products_df: pd.DataFrame, 
                          num_users_per_group: int = 1000,
                          items_per_user: int = 10,
                          confidence_level: float = 0.95) -> ABTestResults:
    """
    Convenience function to run complete A/B test simulation
    
    Args:
        products_df: DataFrame with product data and ML predictions
        num_users_per_group: Number of users per test group
        items_per_user: Number of items shown to each user
        confidence_level: Statistical confidence level
        
    Returns:
        ABTestResults object with complete test results
    """
    config = ABTestConfig(
        num_users_per_group=num_users_per_group,
        items_per_user=items_per_user,
        confidence_level=confidence_level
    )
    
    ab_test = RankingABTest(config)
    return ab_test.build_ab_simulation_system(products_df)


def compare_ranking_methods(products_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Convenience function to compare baseline and ML ranking methods
    
    Args:
        products_df: DataFrame with product data and ML predictions
        
    Returns:
        Dictionary with comparison results
    """
    ab_results = run_ab_test_simulation(products_df)
    
    return {
        'baseline_performance': ab_results.baseline_metrics,
        'ml_performance': ab_results.ml_ranking_metrics,
        'statistical_significance': ab_results.statistical_significance,
        'business_impact': ab_results.business_impact,
        'recommendation': ab_results.business_impact['overall_assessment']['recommendation']
    }


def create_ranking_comparison_report(products_df: pd.DataFrame, 
                                   save_plots: bool = False,
                                   output_dir: str = "ab_test_results") -> Dict[str, Any]:
    """
    Create comprehensive ranking comparison report
    
    Args:
        products_df: DataFrame with product data and ML predictions
        save_plots: Whether to save plots to files
        output_dir: Directory to save results
        
    Returns:
        Dictionary with complete analysis results
    """
    import os
    
    # Run A/B test
    ab_results = run_ab_test_simulation(products_df)
    
    # Create visualizations
    ab_test = RankingABTest()
    ab_test.plot_ab_results(ab_results)
    
    position_fig = ab_test.plot_position_analysis(ab_results.simulation_data)
    
    # Save results if requested
    if save_plots:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save plots
        position_fig.savefig(f"{output_dir}/position_analysis.png", dpi=300, bbox_inches='tight')
        
        # Save results to JSON
        ab_test.export_ab_results(ab_results, f"{output_dir}/ab_test_results.json")
        
        # Save simulation data to CSV
        ab_results.simulation_data.to_csv(f"{output_dir}/simulation_data.csv", index=False)
    
    return ab_results.to_dict()


if __name__ == "__main__":
    print("Enhanced A/B Testing module for Smart Product Re-Ranking System loaded successfully!")
    print("Available classes:")
    print("- BaselineRankingSystem: Implements baseline ranking by view count")
    print("- MLRankingSystem: Implements ML-based ranking")
    print("- RankingABTest: Comprehensive A/B testing with statistical significance")
    print("\nAvailable functions:")
    print("- run_ab_test_simulation(): Run complete A/B test")
    print("- compare_ranking_methods(): Compare baseline vs ML ranking")
    print("- create_ranking_comparison_report(): Generate comprehensive report")