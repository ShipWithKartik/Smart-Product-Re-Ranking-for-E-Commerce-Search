"""
Sample data generation utilities for Smart Product Re-Ranking System
"""

import random
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import numpy as np

class SampleDataGenerator:
    """Generates realistic sample data for testing and development"""
    
    def __init__(self, seed: int = 42):
        """Initialize with random seed for reproducibility"""
        random.seed(seed)
        np.random.seed(seed)
        
        # Product categories and their typical characteristics
        self.categories = {
            'Electronics': {'price_range': (50, 1000), 'rating_bias': 4.2, 'discount_range': (5, 25)},
            'Clothing': {'price_range': (20, 200), 'rating_bias': 4.0, 'discount_range': (15, 40)},
            'Home & Garden': {'price_range': (30, 300), 'rating_bias': 4.1, 'discount_range': (10, 30)},
            'Books': {'price_range': (10, 50), 'rating_bias': 4.4, 'discount_range': (0, 20)},
            'Sports': {'price_range': (40, 400), 'rating_bias': 4.2, 'discount_range': (15, 35)}
        }
        
        # Common search queries by category
        self.search_queries = {
            'Electronics': ['laptop', 'smartphone', 'headphones', 'tablet', 'camera', 'speaker'],
            'Clothing': ['jacket', 'shoes', 'dress', 'jeans', 'shirt', 'sweater'],
            'Home & Garden': ['furniture', 'lamp', 'plant', 'tool', 'decor', 'kitchen'],
            'Books': ['novel', 'textbook', 'cookbook', 'biography', 'fiction', 'guide'],
            'Sports': ['equipment', 'shoes', 'clothing', 'gear', 'fitness', 'outdoor']
        }
    
    def generate_products(self, num_products: int = 100) -> pd.DataFrame:
        """Generate sample product data"""
        products = []
        
        for i in range(1, num_products + 1):
            category = random.choice(list(self.categories.keys()))
            cat_info = self.categories[category]
            
            # Generate price with some randomness
            price = round(random.uniform(*cat_info['price_range']), 2)
            
            # Generate rating with bias towards category average
            rating = max(1.0, min(5.0, np.random.normal(cat_info['rating_bias'], 0.5)))
            rating = round(rating, 1)
            
            # Generate discount
            discount = round(random.uniform(*cat_info['discount_range']), 1)
            
            products.append({
                'id': i,
                'category': category,
                'price': price,
                'rating': rating,
                'discount': discount
            })
        
        return pd.DataFrame(products)
    
    def generate_inventory(self, product_ids: List[int]) -> pd.DataFrame:
        """Generate sample inventory data"""
        inventory = []
        
        for product_id in product_ids:
            # Stock levels with realistic distribution
            stock = random.choices(
                [0, random.randint(1, 20), random.randint(21, 100), random.randint(101, 500)],
                weights=[0.05, 0.25, 0.50, 0.20]
            )[0]
            
            # Delivery time based on stock levels
            if stock == 0:
                delivery_time = random.randint(7, 14)  # Out of stock = longer delivery
            elif stock < 20:
                delivery_time = random.randint(3, 7)   # Low stock = medium delivery
            else:
                delivery_time = random.randint(1, 4)   # Good stock = fast delivery
            
            # Seller rating
            seller_rating = round(max(2.0, min(5.0, np.random.normal(4.2, 0.6))), 1)
            
            inventory.append({
                'product_id': product_id,
                'stock': stock,
                'delivery_time': delivery_time,
                'seller_rating': seller_rating
            })
        
        return pd.DataFrame(inventory)
    
    def generate_user_searches(self, product_ids: List[int], products_df: pd.DataFrame, 
                             num_searches: int = 1000) -> pd.DataFrame:
        """Generate realistic user search behavior data"""
        searches = []
        
        # Create product category mapping
        product_categories = dict(zip(products_df['id'], products_df['category']))
        
        for i in range(num_searches):
            user_id = random.randint(1, 200)  # 200 unique users
            product_id = random.choice(product_ids)
            category = product_categories[product_id]
            
            # Select query based on product category
            query = random.choice(self.search_queries[category])
            
            # Generate behavioral metrics with realistic patterns
            impressions = random.randint(1, 50)
            
            # Click probability based on product rating and position
            product_rating = products_df[products_df['id'] == product_id]['rating'].iloc[0]
            click_prob = min(0.8, (product_rating / 5.0) * 0.6 + random.uniform(0.1, 0.3))
            clicks = int(impressions * click_prob * random.uniform(0.5, 1.5))
            clicks = min(clicks, impressions)  # Can't have more clicks than impressions
            
            # Add to cart probability (based on clicks)
            add_to_cart = clicks > 0 and random.random() < 0.3
            
            # Purchase probability (based on add to cart and product characteristics)
            if add_to_cart:
                purchase_prob = 0.4 + (product_rating - 3.0) * 0.1  # Higher rating = higher purchase prob
                purchase_flag = random.random() < purchase_prob
            else:
                purchase_flag = False
            
            # Generate timestamp within last 30 days
            days_ago = random.randint(0, 30)
            search_timestamp = datetime.now() - timedelta(days=days_ago)
            
            searches.append({
                'user_id': user_id,
                'query': query,
                'product_id': product_id,
                'impressions': impressions,
                'clicks': clicks,
                'add_to_cart': add_to_cart,
                'purchase_flag': purchase_flag,
                'search_timestamp': search_timestamp
            })
        
        return pd.DataFrame(searches)
    
    def generate_complete_dataset(self, num_products: int = 100, 
                                num_searches: int = 1000) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Generate complete dataset with all tables"""
        print(f"Generating {num_products} products...")
        products_df = self.generate_products(num_products)
        
        print(f"Generating inventory for {num_products} products...")
        inventory_df = self.generate_inventory(products_df['id'].tolist())
        
        print(f"Generating {num_searches} user searches...")
        searches_df = self.generate_user_searches(products_df['id'].tolist(), products_df, num_searches)
        
        return products_df, inventory_df, searches_df
    
    def save_to_sql_files(self, products_df: pd.DataFrame, inventory_df: pd.DataFrame, 
                         searches_df: pd.DataFrame, output_dir: str = "sql/"):
        """Save generated data as SQL INSERT statements"""
        
        # Products INSERT statements
        with open(f"{output_dir}generated_products.sql", "w") as f:
            f.write("-- Generated product data\n")
            f.write("INSERT INTO products (id, category, price, rating, discount) VALUES\n")
            
            values = []
            for _, row in products_df.iterrows():
                values.append(f"({row['id']}, '{row['category']}', {row['price']}, {row['rating']}, {row['discount']})")
            
            f.write(",\n".join(values) + ";\n")
        
        # Inventory INSERT statements
        with open(f"{output_dir}generated_inventory.sql", "w") as f:
            f.write("-- Generated inventory data\n")
            f.write("INSERT INTO inventory (product_id, stock, delivery_time, seller_rating) VALUES\n")
            
            values = []
            for _, row in inventory_df.iterrows():
                values.append(f"({row['product_id']}, {row['stock']}, {row['delivery_time']}, {row['seller_rating']})")
            
            f.write(",\n".join(values) + ";\n")
        
        # User searches INSERT statements
        with open(f"{output_dir}generated_searches.sql", "w") as f:
            f.write("-- Generated user search data\n")
            f.write("INSERT INTO user_searches (user_id, query, product_id, impressions, clicks, add_to_cart, purchase_flag, search_timestamp) VALUES\n")
            
            values = []
            for _, row in searches_df.iterrows():
                timestamp_str = row['search_timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                values.append(f"({row['user_id']}, '{row['query']}', {row['product_id']}, {row['impressions']}, {row['clicks']}, {row['add_to_cart']}, {row['purchase_flag']}, '{timestamp_str}')")
            
            f.write(",\n".join(values) + ";\n")
        
        print(f"Generated SQL files saved to {output_dir}")


if __name__ == "__main__":
    # Example usage
    generator = SampleDataGenerator()
    products, inventory, searches = generator.generate_complete_dataset(num_products=50, num_searches=500)
    
    print("Sample data generation complete!")
    print(f"Products: {len(products)} records")
    print(f"Inventory: {len(inventory)} records") 
    print(f"Searches: {len(searches)} records")
    
    # Save to SQL files
    generator.save_to_sql_files(products, inventory, searches)