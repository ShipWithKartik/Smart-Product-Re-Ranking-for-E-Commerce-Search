"""
Configuration file for A/B testing simulation
Provides configurable parameters for A/B testing, statistical analysis, and business impact evaluation
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
import json
from pathlib import Path


@dataclass
class SimulationConfig:
    """Configuration for A/B test simulation"""
    
    # Test setup
    num_users_per_group: int = 1000
    items_per_user: int = 10
    test_duration_days: int = 14
    
    # User behavior simulation
    baseline_ctr: float = 0.05  # Click-through rate for baseline
    baseline_cvr: float = 0.02  # Conversion rate for baseline
    position_decay_factor: float = 0.8  # How much CTR decreases by position
    
    # Randomization
    random_seed: int = 42
    enable_user_randomization: bool = True
    
    # Sample size calculation
    auto_calculate_sample_size: bool = False
    minimum_detectable_effect: float = 0.05  # 5% relative improvement
    statistical_power: float = 0.8
    
    # Simulation realism
    add_noise: bool = True
    noise_level: float = 0.1
    simulate_seasonality: bool = False
    weekend_effect: float = 0.9  # Weekend traffic multiplier


@dataclass
class StatisticalConfig:
    """Configuration for statistical analysis"""
    
    # Significance testing
    confidence_level: float = 0.95
    alpha: float = 0.05
    two_tailed_test: bool = True
    
    # Multiple testing correction
    apply_bonferroni_correction: bool = True
    family_wise_error_rate: float = 0.05
    
    # Effect size calculation
    calculate_effect_size: bool = True
    effect_size_method: str = "cohen_d"  # Options: "cohen_d", "glass_delta", "hedges_g"
    
    # Bootstrap analysis
    enable_bootstrap: bool = True
    bootstrap_samples: int = 1000
    bootstrap_confidence_level: float = 0.95
    
    # Bayesian analysis (optional)
    enable_bayesian_analysis: bool = False
    prior_alpha: float = 1.0
    prior_beta: float = 1.0


@dataclass
class MetricsConfig:
    """Configuration for metrics calculation"""
    
    # Primary metrics
    primary_metric: str = "conversion_rate"  # Options: "ctr", "conversion_rate", "revenue_per_user"
    
    # Secondary metrics
    secondary_metrics: List[str] = field(default_factory=lambda: [
        "click_through_rate",
        "purchase_rate",
        "revenue_per_user",
        "items_per_session"
    ])
    
    # Metric calculation settings
    exclude_outliers: bool = True
    outlier_method: str = "iqr"  # Options: "iqr", "zscore", "percentile"
    outlier_threshold: float = 3.0
    
    # Revenue calculation
    average_order_value: float = 50.0
    revenue_attribution_window_hours: int = 24
    
    # Engagement metrics
    calculate_engagement_metrics: bool = True
    session_timeout_minutes: int = 30
    
    # Funnel analysis
    enable_funnel_analysis: bool = True
    funnel_steps: List[str] = field(default_factory=lambda: [
        "impression", "click", "add_to_cart", "purchase"
    ])


@dataclass
class BusinessImpactConfig:
    """Configuration for business impact analysis"""
    
    # Business context
    monthly_active_users: int = 100000
    annual_revenue_target: float = 10000000.0
    
    # Cost considerations
    implementation_cost: float = 50000.0
    maintenance_cost_annual: float = 20000.0
    opportunity_cost_factor: float = 0.1
    
    # ROI calculation
    calculate_roi: bool = True
    roi_time_horizon_months: int = 12
    discount_rate: float = 0.1
    
    # Risk assessment
    enable_risk_analysis: bool = True
    confidence_intervals: List[float] = field(default_factory=lambda: [0.8, 0.9, 0.95])
    worst_case_scenario_percentile: float = 0.1
    
    # Sensitivity analysis
    perform_sensitivity_analysis: bool = True
    sensitivity_parameters: List[str] = field(default_factory=lambda: [
        "conversion_rate", "average_order_value", "user_base_size"
    ])
    sensitivity_range: float = 0.2  # ±20% variation


@dataclass
class ReportingConfig:
    """Configuration for reporting and visualization"""
    
    # Report generation
    generate_summary_report: bool = True
    generate_detailed_report: bool = True
    include_visualizations: bool = True
    
    # Visualization settings
    plot_confidence_intervals: bool = True
    plot_distribution_comparisons: bool = True
    plot_time_series: bool = True
    plot_funnel_analysis: bool = True
    
    # Export settings
    export_raw_data: bool = False
    export_format: str = "json"  # Options: "json", "csv", "excel"
    include_metadata: bool = True
    
    # Dashboard creation
    create_dashboard: bool = False
    dashboard_refresh_interval_hours: int = 24
    
    # Alerting
    enable_alerts: bool = False
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "significant_degradation": -0.05,
        "significant_improvement": 0.05
    })


@dataclass
class ValidationConfig:
    """Configuration for test validation"""
    
    # Pre-test validation
    validate_sample_size: bool = True
    validate_randomization: bool = True
    validate_data_quality: bool = True
    
    # During-test monitoring
    enable_sequential_testing: bool = False
    interim_analysis_frequency_days: int = 3
    early_stopping_threshold: float = 0.001
    
    # Post-test validation
    validate_assumptions: bool = True
    check_novelty_effect: bool = True
    validate_external_validity: bool = True
    
    # Quality checks
    minimum_sample_size_per_group: int = 100
    maximum_imbalance_ratio: float = 1.1
    minimum_test_duration_days: int = 7


@dataclass
class ABTestingConfig:
    """Complete configuration for A/B testing pipeline"""
    
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    statistical: StatisticalConfig = field(default_factory=StatisticalConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    business_impact: BusinessImpactConfig = field(default_factory=BusinessImpactConfig)
    reporting: ReportingConfig = field(default_factory=ReportingConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'simulation': self.simulation.__dict__,
            'statistical': self.statistical.__dict__,
            'metrics': self.metrics.__dict__,
            'business_impact': self.business_impact.__dict__,
            'reporting': self.reporting.__dict__,
            'validation': self.validation.__dict__
        }
    
    def save_to_file(self, file_path: str) -> None:
        """Save configuration to JSON file"""
        config_dict = self.to_dict()
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load_from_file(cls, file_path: str) -> 'ABTestingConfig':
        """Load configuration from JSON file"""
        with open(file_path, 'r') as f:
            config_dict = json.load(f)
        
        return cls(
            simulation=SimulationConfig(**config_dict.get('simulation', {})),
            statistical=StatisticalConfig(**config_dict.get('statistical', {})),
            metrics=MetricsConfig(**config_dict.get('metrics', {})),
            business_impact=BusinessImpactConfig(**config_dict.get('business_impact', {})),
            reporting=ReportingConfig(**config_dict.get('reporting', {})),
            validation=ValidationConfig(**config_dict.get('validation', {}))
        )


# Predefined configuration profiles
class ABTestConfigProfiles:
    """Predefined configuration profiles for different testing scenarios"""
    
    @staticmethod
    def quick_test() -> ABTestingConfig:
        """Configuration for quick testing and validation"""
        config = ABTestingConfig()
        config.simulation.num_users_per_group = 500
        config.simulation.test_duration_days = 7
        config.statistical.bootstrap_samples = 500
        config.business_impact.calculate_roi = False
        config.reporting.generate_detailed_report = False
        return config
    
    @staticmethod
    def production_test() -> ABTestingConfig:
        """Configuration for production A/B testing"""
        config = ABTestingConfig()
        config.simulation.num_users_per_group = 5000
        config.simulation.test_duration_days = 14
        config.statistical.apply_bonferroni_correction = True
        config.business_impact.calculate_roi = True
        config.validation.enable_sequential_testing = True
        config.reporting.create_dashboard = True
        return config
    
    @staticmethod
    def research_study() -> ABTestingConfig:
        """Configuration for research and deep analysis"""
        config = ABTestingConfig()
        config.simulation.num_users_per_group = 2000
        config.statistical.enable_bayesian_analysis = True
        config.business_impact.perform_sensitivity_analysis = True
        config.business_impact.enable_risk_analysis = True
        config.reporting.generate_detailed_report = True
        config.reporting.export_raw_data = True
        return config
    
    @staticmethod
    def minimal_test() -> ABTestingConfig:
        """Configuration for minimal testing with basic metrics"""
        config = ABTestingConfig()
        config.simulation.num_users_per_group = 200
        config.simulation.test_duration_days = 3
        config.metrics.secondary_metrics = ["click_through_rate"]
        config.business_impact.calculate_roi = False
        config.reporting.include_visualizations = False
        config.validation.validate_assumptions = False
        return config


# Default configuration
DEFAULT_AB_CONFIG = ABTestingConfig()


def create_ab_config_file(profile: str = "default", output_path: str = "ab_testing_config.json") -> None:
    """
    Create an A/B testing configuration file
    
    Args:
        profile: Configuration profile name
        output_path: Path to save configuration file
    """
    if profile == "quick_test":
        config = ABTestConfigProfiles.quick_test()
    elif profile == "production_test":
        config = ABTestConfigProfiles.production_test()
    elif profile == "research_study":
        config = ABTestConfigProfiles.research_study()
    elif profile == "minimal_test":
        config = ABTestConfigProfiles.minimal_test()
    else:
        config = DEFAULT_AB_CONFIG
    
    config.save_to_file(output_path)
    print(f"A/B testing configuration file created: {output_path}")


def validate_ab_config(config: ABTestingConfig) -> List[str]:
    """
    Validate A/B testing configuration
    
    Args:
        config: Configuration to validate
        
    Returns:
        List of validation error messages
    """
    errors = []
    
    # Validate sample sizes
    if config.simulation.num_users_per_group < 50:
        errors.append("Sample size per group should be at least 50 for meaningful results")
    
    # Validate confidence level
    if not (0.5 <= config.statistical.confidence_level <= 0.99):
        errors.append("Confidence level should be between 0.5 and 0.99")
    
    # Validate test duration
    if config.simulation.test_duration_days < 1:
        errors.append("Test duration should be at least 1 day")
    
    # Validate effect size
    if config.simulation.minimum_detectable_effect <= 0:
        errors.append("Minimum detectable effect should be positive")
    
    # Validate business metrics
    if config.business_impact.monthly_active_users <= 0:
        errors.append("Monthly active users should be positive")
    
    return errors


if __name__ == "__main__":
    # Create configuration files for different profiles
    profiles = ["default", "quick_test", "production_test", "research_study", "minimal_test"]
    
    for profile in profiles:
        output_file = f"ab_testing_config_{profile}.json"
        create_ab_config_file(profile, output_file)
        print(f"Created {output_file}")
    
    # Validate default configuration
    errors = validate_ab_config(DEFAULT_AB_CONFIG)
    if errors:
        print(f"Configuration errors: {errors}")
    else:
        print("Default A/B testing configuration is valid")