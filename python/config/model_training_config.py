"""
Configuration file for machine learning model training
Provides configurable parameters for model training, evaluation, and deployment
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
import json
from pathlib import Path
from enum import Enum


class ModelType(Enum):
    """Available model types"""
    LOGISTIC_REGRESSION = "logistic_regression"
    GRADIENT_BOOSTING = "gradient_boosting"
    RANDOM_FOREST = "random_forest"
    SVM = "svm"


@dataclass
class ModelHyperparameters:
    """Hyperparameters for different model types"""
    
    # Logistic Regression parameters
    logistic_regression: Dict[str, Any] = field(default_factory=lambda: {
        'C': 1.0,
        'penalty': 'l2',
        'solver': 'liblinear',
        'max_iter': 1000,
        'random_state': 42,
        'class_weight': 'balanced'
    })
    
    # Gradient Boosting parameters
    gradient_boosting: Dict[str, Any] = field(default_factory=lambda: {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 6,
        'min_samples_split': 20,
        'min_samples_leaf': 10,
        'subsample': 0.8,
        'random_state': 42
    })
    
    # Random Forest parameters
    random_forest: Dict[str, Any] = field(default_factory=lambda: {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 10,
        'min_samples_leaf': 5,
        'max_features': 'sqrt',
        'random_state': 42,
        'class_weight': 'balanced'
    })
    
    # SVM parameters
    svm: Dict[str, Any] = field(default_factory=lambda: {
        'C': 1.0,
        'kernel': 'rbf',
        'gamma': 'scale',
        'probability': True,
        'random_state': 42,
        'class_weight': 'balanced'
    })


@dataclass
class TrainingConfig:
    """Configuration for model training"""
    
    # Data splitting
    test_size: float = 0.2
    validation_size: float = 0.2
    random_state: int = 42
    stratify: bool = True
    
    # Cross-validation
    cv_folds: int = 5
    cv_scoring: str = 'roc_auc'
    cv_random_state: int = 42
    
    # Feature preprocessing
    scale_features: bool = True
    scaler_type: str = 'standard'  # Options: 'standard', 'minmax', 'robust'
    handle_missing: str = 'median'  # Options: 'mean', 'median', 'mode', 'drop'
    
    # Feature selection
    enable_feature_selection: bool = False
    feature_selection_method: str = 'rfe'  # Options: 'rfe', 'univariate', 'lasso'
    max_features: Optional[int] = None
    
    # Model validation
    min_roc_auc: float = 0.7
    min_precision: float = 0.6
    min_recall: float = 0.6
    
    # Training optimization
    enable_hyperparameter_tuning: bool = False
    tuning_method: str = 'grid_search'  # Options: 'grid_search', 'random_search', 'bayesian'
    tuning_cv_folds: int = 3
    tuning_n_iter: int = 50  # For random search


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation"""
    
    # Metrics to calculate
    calculate_roc_auc: bool = True
    calculate_precision_recall: bool = True
    calculate_f1_score: bool = True
    calculate_confusion_matrix: bool = True
    
    # Visualization settings
    generate_plots: bool = True
    plot_roc_curve: bool = True
    plot_precision_recall_curve: bool = True
    plot_feature_importance: bool = True
    plot_confusion_matrix: bool = True
    
    # Feature importance analysis
    analyze_feature_importance: bool = True
    top_n_features: int = 10
    importance_threshold: float = 0.01
    
    # Performance thresholds
    excellent_auc_threshold: float = 0.9
    good_auc_threshold: float = 0.8
    acceptable_auc_threshold: float = 0.7
    
    # Report generation
    generate_detailed_report: bool = True
    include_model_interpretation: bool = True
    save_evaluation_plots: bool = True


@dataclass
class DeploymentConfig:
    """Configuration for model deployment"""
    
    # Model persistence
    save_model: bool = True
    model_save_path: str = "models"
    model_filename_template: str = "{model_name}_{timestamp}"
    save_scaler: bool = True
    save_metadata: bool = True
    
    # Model versioning
    enable_versioning: bool = True
    version_format: str = "v{major}.{minor}.{patch}"
    auto_increment_version: bool = True
    
    # Performance monitoring
    enable_monitoring: bool = True
    performance_degradation_threshold: float = 0.05
    retrain_threshold: float = 0.1
    
    # Prediction settings
    default_prediction_threshold: float = 0.5
    batch_prediction_size: int = 1000
    enable_prediction_caching: bool = False


@dataclass
class LoggingConfig:
    """Configuration for logging"""
    
    # Logging levels
    log_level: str = "INFO"  # Options: "DEBUG", "INFO", "WARNING", "ERROR"
    enable_file_logging: bool = True
    log_file_path: str = "logs/model_training.log"
    
    # Log formatting
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    
    # Performance logging
    log_training_progress: bool = True
    log_evaluation_metrics: bool = True
    log_feature_importance: bool = True
    
    # Error handling
    log_errors_to_file: bool = True
    error_log_file: str = "logs/model_errors.log"


@dataclass
class ModelTrainingConfig:
    """Complete configuration for model training pipeline"""
    
    # Model selection
    primary_model: ModelType = ModelType.LOGISTIC_REGRESSION
    models_to_compare: List[ModelType] = field(default_factory=lambda: [
        ModelType.LOGISTIC_REGRESSION,
        ModelType.GRADIENT_BOOSTING
    ])
    
    # Configuration sections
    hyperparameters: ModelHyperparameters = field(default_factory=ModelHyperparameters)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    deployment: DeploymentConfig = field(default_factory=DeploymentConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'primary_model': self.primary_model.value,
            'models_to_compare': [model.value for model in self.models_to_compare],
            'hyperparameters': {
                'logistic_regression': self.hyperparameters.logistic_regression,
                'gradient_boosting': self.hyperparameters.gradient_boosting,
                'random_forest': self.hyperparameters.random_forest,
                'svm': self.hyperparameters.svm
            },
            'training': self.training.__dict__,
            'evaluation': self.evaluation.__dict__,
            'deployment': self.deployment.__dict__,
            'logging': self.logging.__dict__
        }
    
    def save_to_file(self, file_path: str) -> None:
        """Save configuration to JSON file"""
        config_dict = self.to_dict()
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load_from_file(cls, file_path: str) -> 'ModelTrainingConfig':
        """Load configuration from JSON file"""
        with open(file_path, 'r') as f:
            config_dict = json.load(f)
        
        # Convert model types back from strings
        primary_model = ModelType(config_dict.get('primary_model', 'logistic_regression'))
        models_to_compare = [ModelType(model) for model in config_dict.get('models_to_compare', [])]
        
        # Create hyperparameters object
        hyperparams_dict = config_dict.get('hyperparameters', {})
        hyperparameters = ModelHyperparameters(
            logistic_regression=hyperparams_dict.get('logistic_regression', {}),
            gradient_boosting=hyperparams_dict.get('gradient_boosting', {}),
            random_forest=hyperparams_dict.get('random_forest', {}),
            svm=hyperparams_dict.get('svm', {})
        )
        
        return cls(
            primary_model=primary_model,
            models_to_compare=models_to_compare,
            hyperparameters=hyperparameters,
            training=TrainingConfig(**config_dict.get('training', {})),
            evaluation=EvaluationConfig(**config_dict.get('evaluation', {})),
            deployment=DeploymentConfig(**config_dict.get('deployment', {})),
            logging=LoggingConfig(**config_dict.get('logging', {}))
        )


# Predefined configuration profiles
class ModelConfigProfiles:
    """Predefined configuration profiles for different use cases"""
    
    @staticmethod
    def development() -> ModelTrainingConfig:
        """Configuration for development/testing"""
        config = ModelTrainingConfig()
        config.training.cv_folds = 3
        config.training.enable_hyperparameter_tuning = False
        config.evaluation.generate_plots = True
        config.deployment.save_model = False
        config.logging.log_level = "DEBUG"
        return config
    
    @staticmethod
    def production() -> ModelTrainingConfig:
        """Configuration for production deployment"""
        config = ModelTrainingConfig()
        config.training.cv_folds = 5
        config.training.enable_hyperparameter_tuning = True
        config.evaluation.generate_plots = False
        config.deployment.save_model = True
        config.deployment.enable_monitoring = True
        config.logging.log_level = "INFO"
        return config
    
    @staticmethod
    def research() -> ModelTrainingConfig:
        """Configuration for research and experimentation"""
        config = ModelTrainingConfig()
        config.models_to_compare = [
            ModelType.LOGISTIC_REGRESSION,
            ModelType.GRADIENT_BOOSTING,
            ModelType.RANDOM_FOREST,
            ModelType.SVM
        ]
        config.training.enable_hyperparameter_tuning = True
        config.training.tuning_method = 'random_search'
        config.evaluation.generate_plots = True
        config.evaluation.analyze_feature_importance = True
        config.logging.log_level = "DEBUG"
        return config
    
    @staticmethod
    def fast_prototype() -> ModelTrainingConfig:
        """Configuration for fast prototyping"""
        config = ModelTrainingConfig()
        config.training.cv_folds = 3
        config.training.test_size = 0.3
        config.training.enable_hyperparameter_tuning = False
        config.evaluation.generate_plots = False
        config.deployment.save_model = False
        config.logging.log_level = "WARNING"
        return config


# Default configuration
DEFAULT_MODEL_CONFIG = ModelTrainingConfig()


def create_model_config_file(profile: str = "default", output_path: str = "model_training_config.json") -> None:
    """
    Create a model training configuration file
    
    Args:
        profile: Configuration profile name
        output_path: Path to save configuration file
    """
    if profile == "development":
        config = ModelConfigProfiles.development()
    elif profile == "production":
        config = ModelConfigProfiles.production()
    elif profile == "research":
        config = ModelConfigProfiles.research()
    elif profile == "fast_prototype":
        config = ModelConfigProfiles.fast_prototype()
    else:
        config = DEFAULT_MODEL_CONFIG
    
    config.save_to_file(output_path)
    print(f"Model training configuration file created: {output_path}")


if __name__ == "__main__":
    # Create configuration files for different profiles
    profiles = ["default", "development", "production", "research", "fast_prototype"]
    
    for profile in profiles:
        output_file = f"model_training_config_{profile}.json"
        create_model_config_file(profile, output_file)
        print(f"Created {output_file}")