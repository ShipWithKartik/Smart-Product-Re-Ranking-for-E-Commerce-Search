"""
Configuration settings for Smart Product Re-Ranking System
"""

import os
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class DatabaseConfig:
    """Database connection configuration"""
    # SQLite configuration for Retailrocket dataset
    db_type: str = "sqlite"
    sqlite_path: str = "retailrocket.db"
    
    # MySQL configuration (for reference)
    host: str = "localhost"
    port: int = 3306
    database: str = "smart_product_ranking"
    user: str = "root"
    password: str = ""
    
    def get_connection_string(self) -> str:
        """Generate database connection string"""
        if self.db_type == "sqlite":
            return f"sqlite:///{self.sqlite_path}"
        else:
            return f"mysql+pymysql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"

@dataclass
class FeatureConfig:
    """Feature engineering configuration"""
    price_buckets: list = None
    rating_buckets: list = None
    discount_thresholds: list = None
    
    def __post_init__(self):
        if self.price_buckets is None:
            self.price_buckets = [0, 50, 100, 200, 500, float('inf')]
        if self.rating_buckets is None:
            self.rating_buckets = [0, 3.0, 4.0, 4.5, 5.0]
        if self.discount_thresholds is None:
            self.discount_thresholds = [0, 10, 25, 50, 100]

@dataclass
class ModelConfig:
    """Machine learning model configuration"""
    performance_quantile: float = 0.7
    min_roc_auc: float = 0.7
    random_state: int = 42
    test_size: float = 0.2
    
    # Logistic Regression parameters
    lr_max_iter: int = 1000
    lr_solver: str = "liblinear"
    
    # Gradient Boosting parameters
    gb_n_estimators: int = 100
    gb_learning_rate: float = 0.1
    gb_max_depth: int = 3

@dataclass
class SystemConfig:
    """Overall system configuration"""
    database: DatabaseConfig = None
    features: FeatureConfig = None
    model: ModelConfig = None
    
    def __post_init__(self):
        if self.database is None:
            self.database = DatabaseConfig()
        if self.features is None:
            self.features = FeatureConfig()
        if self.model is None:
            self.model = ModelConfig()

# Global configuration instance
CONFIG = SystemConfig()