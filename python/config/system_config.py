"""
System-wide configuration for the Smart Product Re-Ranking system
Provides centralized configuration management and environment-specific settings
"""

import json
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from datetime import datetime

# Import configuration classes (using absolute imports to avoid relative import issues)
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.behavioral_metrics_config import BehavioralMetricsConfig, ConfigProfiles
from config.model_training_config import ModelTrainingConfig, ModelConfigProfiles
from config.ab_testing_config import ABTestingConfig, ABTestConfigProfiles


@dataclass
class DatabaseConfig:
    """Database configuration settings"""
    host: str = "localhost"
    port: int = 5432
    database: str = "smart_ranking"
    username: str = "ranking_user"
    password: str = ""  # Should be set via environment variable
    connection_pool_size: int = 10
    connection_timeout: int = 30
    enable_ssl: bool = True


@dataclass
class CacheConfig:
    """Cache configuration settings"""
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str = ""  # Should be set via environment variable
    cache_ttl_seconds: int = 3600
    enable_caching: bool = True
    max_cache_size_mb: int = 100


@dataclass
class APIConfig:
    """API configuration settings"""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    max_request_size_mb: int = 10
    rate_limit_per_minute: int = 1000
    enable_cors: bool = True
    api_key_required: bool = True
    jwt_secret_key: str = ""  # Should be set via environment variable
    jwt_expiration_hours: int = 24


@dataclass
class MonitoringConfig:
    """Monitoring and alerting configuration"""
    enable_metrics: bool = True
    metrics_port: int = 9090
    log_level: str = "INFO"
    log_retention_days: int = 30
    
    # Performance thresholds
    max_prediction_latency_ms: int = 100
    max_memory_usage_mb: int = 1000
    min_model_accuracy: float = 0.7
    
    # Alerting
    enable_alerts: bool = True
    alert_email: str = ""
    slack_webhook_url: str = ""
    
    # Health checks
    health_check_interval_seconds: int = 60
    health_check_timeout_seconds: int = 10


@dataclass
class SecurityConfig:
    """Security configuration settings"""
    enable_encryption: bool = True
    encryption_key: str = ""  # Should be set via environment variable
    enable_audit_logging: bool = True
    max_login_attempts: int = 5
    session_timeout_minutes: int = 30
    
    # Data privacy
    enable_data_anonymization: bool = True
    pii_retention_days: int = 90
    enable_gdpr_compliance: bool = True


@dataclass
class DeploymentConfig:
    """Deployment configuration settings"""
    environment: str = "development"  # development, staging, production
    version: str = "1.0.0"
    deployment_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Scaling
    min_replicas: int = 1
    max_replicas: int = 10
    cpu_limit: str = "1000m"
    memory_limit: str = "2Gi"
    
    # Feature flags
    enable_new_features: bool = False
    enable_experimental_models: bool = False
    enable_a_b_testing: bool = True


@dataclass
class SystemConfig:
    """Complete system configuration"""
    
    # Core configurations
    behavioral_metrics: BehavioralMetricsConfig = field(default_factory=BehavioralMetricsConfig)
    model_training: ModelTrainingConfig = field(default_factory=ModelTrainingConfig)
    ab_testing: ABTestingConfig = field(default_factory=ABTestingConfig)
    
    # Infrastructure configurations
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    api: APIConfig = field(default_factory=APIConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    deployment: DeploymentConfig = field(default_factory=DeploymentConfig)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'behavioral_metrics': self.behavioral_metrics.to_dict(),
            'model_training': self.model_training.to_dict(),
            'ab_testing': self.ab_testing.to_dict(),
            'database': self.database.__dict__,
            'cache': self.cache.__dict__,
            'api': self.api.__dict__,
            'monitoring': self.monitoring.__dict__,
            'security': self.security.__dict__,
            'deployment': self.deployment.__dict__
        }
    
    def save_to_file(self, file_path: str) -> None:
        """Save configuration to JSON file"""
        config_dict = self.to_dict()
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load_from_file(cls, file_path: str) -> 'SystemConfig':
        """Load configuration from JSON file"""
        with open(file_path, 'r') as f:
            config_dict = json.load(f)
        
        # Load sub-configurations
        behavioral_config = BehavioralMetricsConfig()
        if 'behavioral_metrics' in config_dict:
            behavioral_config.update_from_dict(config_dict['behavioral_metrics'])
        
        model_config = ModelTrainingConfig.load_from_file(file_path) if 'model_training' in config_dict else ModelTrainingConfig()
        ab_config = ABTestingConfig.load_from_file(file_path) if 'ab_testing' in config_dict else ABTestingConfig()
        
        return cls(
            behavioral_metrics=behavioral_config,
            model_training=model_config,
            ab_testing=ab_config,
            database=DatabaseConfig(**config_dict.get('database', {})),
            cache=CacheConfig(**config_dict.get('cache', {})),
            api=APIConfig(**config_dict.get('api', {})),
            monitoring=MonitoringConfig(**config_dict.get('monitoring', {})),
            security=SecurityConfig(**config_dict.get('security', {})),
            deployment=DeploymentConfig(**config_dict.get('deployment', {}))
        )
    
    def load_environment_variables(self) -> None:
        """Load sensitive configuration from environment variables"""
        # Database
        self.database.password = os.getenv('DB_PASSWORD', self.database.password)
        self.database.host = os.getenv('DB_HOST', self.database.host)
        self.database.port = int(os.getenv('DB_PORT', str(self.database.port)))
        
        # Cache
        self.cache.redis_password = os.getenv('REDIS_PASSWORD', self.cache.redis_password)
        self.cache.redis_host = os.getenv('REDIS_HOST', self.cache.redis_host)
        
        # API
        self.api.jwt_secret_key = os.getenv('JWT_SECRET_KEY', self.api.jwt_secret_key)
        
        # Security
        self.security.encryption_key = os.getenv('ENCRYPTION_KEY', self.security.encryption_key)
        
        # Monitoring
        self.monitoring.alert_email = os.getenv('ALERT_EMAIL', self.monitoring.alert_email)
        self.monitoring.slack_webhook_url = os.getenv('SLACK_WEBHOOK_URL', self.monitoring.slack_webhook_url)
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of issues"""
        errors = []
        
        # Validate database configuration
        if not self.database.host:
            errors.append("Database host is required")
        if not self.database.database:
            errors.append("Database name is required")
        
        # Validate API configuration
        if self.api.port < 1 or self.api.port > 65535:
            errors.append("API port must be between 1 and 65535")
        
        # Validate monitoring configuration
        if self.monitoring.max_prediction_latency_ms <= 0:
            errors.append("Max prediction latency must be positive")
        
        # Validate security configuration
        if self.security.max_login_attempts <= 0:
            errors.append("Max login attempts must be positive")
        
        return errors


class SystemConfigProfiles:
    """Predefined system configuration profiles"""
    
    @staticmethod
    def development() -> SystemConfig:
        """Configuration for development environment"""
        config = SystemConfig()
        
        # Use development profiles for sub-configurations
        config.behavioral_metrics = ConfigProfiles.development()
        config.model_training = ModelConfigProfiles.development()
        config.ab_testing = ABTestConfigProfiles.quick_test()
        
        # Development-specific settings
        config.api.debug = True
        config.api.api_key_required = False
        config.monitoring.log_level = "DEBUG"
        config.security.enable_encryption = False
        config.deployment.environment = "development"
        config.deployment.enable_experimental_models = True
        
        return config
    
    @staticmethod
    def staging() -> SystemConfig:
        """Configuration for staging environment"""
        config = SystemConfig()
        
        # Use production-like profiles
        config.behavioral_metrics = ConfigProfiles.production()
        config.model_training = ModelConfigProfiles.production()
        config.ab_testing = ABTestConfigProfiles.production_test()
        
        # Staging-specific settings
        config.api.debug = False
        config.api.api_key_required = True
        config.monitoring.log_level = "INFO"
        config.security.enable_encryption = True
        config.deployment.environment = "staging"
        config.deployment.enable_a_b_testing = True
        
        return config
    
    @staticmethod
    def production() -> SystemConfig:
        """Configuration for production environment"""
        config = SystemConfig()
        
        # Use production profiles
        config.behavioral_metrics = ConfigProfiles.production()
        config.model_training = ModelConfigProfiles.production()
        config.ab_testing = ABTestConfigProfiles.production_test()
        
        # Production-specific settings
        config.api.debug = False
        config.api.api_key_required = True
        config.monitoring.log_level = "WARNING"
        config.monitoring.enable_alerts = True
        config.security.enable_encryption = True
        config.security.enable_audit_logging = True
        config.deployment.environment = "production"
        config.deployment.min_replicas = 3
        config.deployment.max_replicas = 20
        
        return config
    
    @staticmethod
    def research() -> SystemConfig:
        """Configuration for research environment"""
        config = SystemConfig()
        
        # Use research profiles
        config.behavioral_metrics = ConfigProfiles.development()
        config.model_training = ModelConfigProfiles.research()
        config.ab_testing = ABTestConfigProfiles.research_study()
        
        # Research-specific settings
        config.api.debug = True
        config.monitoring.log_level = "DEBUG"
        config.deployment.environment = "research"
        config.deployment.enable_experimental_models = True
        
        return config


def create_system_config_file(profile: str = "development", output_path: str = "system_config.json") -> None:
    """
    Create a system configuration file
    
    Args:
        profile: Configuration profile name
        output_path: Path to save configuration file
    """
    if profile == "development":
        config = SystemConfigProfiles.development()
    elif profile == "staging":
        config = SystemConfigProfiles.staging()
    elif profile == "production":
        config = SystemConfigProfiles.production()
    elif profile == "research":
        config = SystemConfigProfiles.research()
    else:
        config = SystemConfig()
    
    # Load environment variables
    config.load_environment_variables()
    
    # Validate configuration
    errors = config.validate()
    if errors:
        print(f"Configuration validation warnings: {errors}")
    
    config.save_to_file(output_path)
    print(f"System configuration file created: {output_path}")


def load_system_config(config_path: str = "system_config.json") -> SystemConfig:
    """
    Load system configuration from file with environment variable override
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Loaded and validated system configuration
    """
    if not Path(config_path).exists():
        print(f"Configuration file {config_path} not found, using default configuration")
        config = SystemConfig()
    else:
        config = SystemConfig.load_from_file(config_path)
    
    # Load environment variables
    config.load_environment_variables()
    
    # Validate configuration
    errors = config.validate()
    if errors:
        raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
    
    return config


# Global configuration instance
_global_config: Optional[SystemConfig] = None


def get_config() -> SystemConfig:
    """Get global configuration instance"""
    global _global_config
    
    if _global_config is None:
        _global_config = load_system_config()
    
    return _global_config


def set_config(config: SystemConfig) -> None:
    """Set global configuration instance"""
    global _global_config
    _global_config = config


if __name__ == "__main__":
    # Create configuration files for different environments
    environments = ["development", "staging", "production", "research"]
    
    for env in environments:
        output_file = f"system_config_{env}.json"
        create_system_config_file(env, output_file)
        print(f"Created {output_file}")
    
    # Test configuration loading and validation
    try:
        config = load_system_config("system_config_development.json")
        print("Configuration loaded and validated successfully")
    except Exception as e:
        print(f"Configuration error: {e}")