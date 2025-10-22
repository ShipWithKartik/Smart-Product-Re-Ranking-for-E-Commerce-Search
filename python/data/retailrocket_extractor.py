"""
Data extraction and processing for Retailrocket dataset
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from typing import Tuple, Optional
import sqlite3
import os

class RetailrocketDataExtractor:
    """Extract and process data from Retailrocket events.csv"""
    
    def __init__(self, events_file_path: str = "events.csv"):
        """Initialize with path to events.csv file"""
        self.events_file_path = events_file_path
        self.db_path = "retailrocket.db"
        
    def load_events_to_sqlite(self, chunk_size: int = 10000) -> None:
        """Load events.csv into SQLite database for efficient querying"""
        print(f"Loading {self.events_file_path} into SQLite database...")
        
        # Create SQLite connection
        conn = sqlite3.connect(self.db_path)
        
        # Read CSV in chunks to handle large files
        chunk_count = 0
        for chunk in pd.read_csv(self.events_file_path, chunksize=chunk_size):
            chunk_count += 1
            print(f"Processing chunk {chunk_count} ({len(chunk)} rows)")
            
            # Clean the data
            chunk = chunk.dropna(subset=['visitorid', 'event', 'itemid'])
            chunk['visitorid'] = chunk['visitorid'].astype(int)
            chunk['itemid'] = chunk['itemid'].astype(int)
            
            # Write to SQLite
            chunk.to_sql('events', conn, if_exists='append', index=False)
        
        # Create indexes for better performance
        cursor = conn.cursor()
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_itemid ON events(itemid)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_event ON events(event)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_visitor_item ON events(visitorid, itemid)")
        
        conn.commit()
        conn.close()
        print(f"Data loaded successfully into {self.db_path}")
    
    def extract_behavioral_features(self) -> pd.DataFrame:
        """Extract aggregated behavioral features from events data"""
        
        # Check if SQLite database exists, create if not
        if not os.path.exists(self.db_path):
            self.load_events_to_sqlite()
        
        # Connect to SQLite database
        conn = sqlite3.connect(self.db_path)
        
        # Main aggregation query
        query = """
        SELECT 
            itemid as product_id,
            SUM(CASE WHEN event = 'view' THEN 1 ELSE 0 END) as total_impressions,
            SUM(CASE WHEN event = 'addtocart' THEN 1 ELSE 0 END) as total_carts,
            SUM(CASE WHEN event = 'transaction' THEN 1 ELSE 0 END) as total_purchases,
            COUNT(DISTINCT visitorid) as unique_visitors,
            COUNT(*) as total_events
        FROM events 
        GROUP BY itemid
        HAVING total_impressions > 0
        ORDER BY total_purchases DESC, total_impressions DESC
        """
        
        print("Extracting behavioral features...")
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        print(f"Extracted features for {len(df)} products")
        return df
    
    def get_data_summary(self) -> dict:
        """Get summary statistics of the dataset"""
        if not os.path.exists(self.db_path):
            self.load_events_to_sqlite()
        
        conn = sqlite3.connect(self.db_path)
        
        # Event distribution
        event_dist = pd.read_sql_query("""
            SELECT event, COUNT(*) as count,
                   ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM events), 2) as percentage
            FROM events GROUP BY event
        """, conn)
        
        # Basic stats
        stats = pd.read_sql_query("""
            SELECT 
                COUNT(*) as total_events,
                COUNT(DISTINCT visitorid) as unique_visitors,
                COUNT(DISTINCT itemid) as unique_items
            FROM events
        """, conn)
        
        conn.close()
        
        return {
            'event_distribution': event_dist,
            'basic_stats': stats,
            'database_path': self.db_path
        }

class RetailrocketFeatureEngineer:
    """Feature engineering for Retailrocket behavioral data"""
    
    @staticmethod
    def calculate_behavioral_metrics(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate CTR, CVR and other behavioral metrics"""
        df = df.copy()
        
        # Calculate Click-Through Rate (views to carts)
        df['ctr'] = np.where(df['total_impressions'] > 0, 
                            df['total_carts'] / df['total_impressions'], 0)
        
        # Calculate Conversion Rate (carts to purchases)  
        df['cvr'] = np.where(df['total_carts'] > 0,
                            df['total_purchases'] / df['total_carts'], 0)
        
        # Calculate Purchase Rate (views to purchases)
        df['purchase_rate'] = np.where(df['total_impressions'] > 0,
                                     df['total_purchases'] / df['total_impressions'], 0)
        
        # Calculate engagement score (normalized)
        df['engagement_score'] = (
            df['total_carts'] * 2 + df['total_purchases'] * 5
        ) / df['total_impressions']
        
        # Handle any remaining NaN values
        df = df.fillna(0)
        
        return df
    
    @staticmethod
    def create_performance_labels(df: pd.DataFrame, 
                                quantile_threshold: float = 0.75) -> pd.DataFrame:
        """Create binary performance labels based on purchase behavior"""
        df = df.copy()
        
        # Calculate performance threshold (75th percentile by default)
        purchase_threshold = df['total_purchases'].quantile(quantile_threshold)
        
        # Create binary label
        df['is_high_performing'] = (df['total_purchases'] >= purchase_threshold).astype(int)
        
        print(f"Performance threshold (purchases): {purchase_threshold}")
        print(f"High performing products: {df['is_high_performing'].sum()} / {len(df)} "
              f"({df['is_high_performing'].mean()*100:.1f}%)")
        
        return df
    
    @staticmethod
    def prepare_features_for_modeling(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare feature matrix and target variable for modeling"""
        
        # Feature columns for modeling
        feature_cols = [
            'total_impressions', 'total_carts', 'unique_visitors',
            'ctr', 'cvr', 'purchase_rate', 'engagement_score'
        ]
        
        # Ensure all feature columns exist
        missing_cols = [col for col in feature_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing feature columns: {missing_cols}")
        
        X = df[feature_cols].copy()
        y = df['is_high_performing'].copy()
        
        # Log transform skewed features
        log_features = ['total_impressions', 'total_carts', 'unique_visitors']
        for feature in log_features:
            X[f'{feature}_log'] = np.log1p(X[feature])  # log1p handles zeros
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"Target distribution: {y.value_counts().to_dict()}")
        
        return X, y


if __name__ == "__main__":
    # Example usage
    extractor = RetailrocketDataExtractor()
    
    # Get data summary
    summary = extractor.get_data_summary()
    print("Dataset Summary:")
    print(summary['basic_stats'])
    print("\nEvent Distribution:")
    print(summary['event_distribution'])
    
    # Extract behavioral features
    behavioral_df = extractor.extract_behavioral_features()
    print(f"\nBehavioral features extracted: {behavioral_df.shape}")
    print(behavioral_df.head())
    
    # Feature engineering
    engineer = RetailrocketFeatureEngineer()
    behavioral_df = engineer.calculate_behavioral_metrics(behavioral_df)
    behavioral_df = engineer.create_performance_labels(behavioral_df)
    
    X, y = engineer.prepare_features_for_modeling(behavioral_df)
    print(f"\nReady for modeling: X{X.shape}, y{y.shape}")