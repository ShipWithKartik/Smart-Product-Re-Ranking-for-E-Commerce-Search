# Data processing package for Smart Product Re-Ranking System

from .csv_loader import (
    CSVDataLoader,
    EventType,
    ValidationResult,
    EventData,
    load_events_csv,
    validate_events_csv
)

from .behavioral_metrics import (
    BehavioralMetricsCalculator,
    ItemBehavioralMetrics,
    calculate_behavioral_metrics,
    get_top_performing_items
)

from .behavioral_pipeline import (
    BehavioralMetricsPipeline,
    run_behavioral_metrics_pipeline
)

# Import feature engineering components
try:
    from .feature_engineering import (
        TemporalFeatureExtractor,
        EngagementFeatureExtractor,
        FeatureEngineeringPipeline,
        extract_temporal_and_engagement_features
    )
except ImportError:
    # Feature engineering requires pandas/numpy
    pass

# Optional imports that require external dependencies
# These are imported only when specifically needed to avoid dependency issues
# from .data_extractor import *  # Requires sqlalchemy
# from .retailrocket_extractor import RetailrocketDataExtractor, RetailrocketFeatureEngineer  # May require additional deps

__all__ = [
    'CSVDataLoader',
    'EventType', 
    'ValidationResult',
    'EventData',
    'load_events_csv',
    'validate_events_csv',
    'BehavioralMetricsCalculator',
    'ItemBehavioralMetrics',
    'calculate_behavioral_metrics',
    'get_top_performing_items',
    'BehavioralMetricsPipeline',
    'run_behavioral_metrics_pipeline'
]

# Add feature engineering to __all__ if successfully imported
try:
    from .feature_engineering import TemporalFeatureExtractor
    __all__.extend([
        'TemporalFeatureExtractor',
        'EngagementFeatureExtractor', 
        'FeatureEngineeringPipeline',
        'extract_temporal_and_engagement_features'
    ])
except ImportError:
    pass