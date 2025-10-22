-- Sample data generation for Smart Product Re-Ranking System
-- This script populates the database with realistic test data

-- Insert sample products across different categories
INSERT INTO products (category, price, rating, discount) VALUES
-- Electronics
('Electronics', 299.99, 4.2, 10.0),
('Electronics', 599.99, 4.5, 15.0),
('Electronics', 149.99, 3.8, 5.0),
('Electronics', 899.99, 4.7, 20.0),
('Electronics', 199.99, 4.1, 8.0),

-- Clothing
('Clothing', 49.99, 4.0, 25.0),
('Clothing', 79.99, 4.3, 30.0),
('Clothing', 29.99, 3.5, 15.0),
('Clothing', 129.99, 4.6, 35.0),
('Clothing', 89.99, 4.2, 20.0),

-- Home & Garden
('Home & Garden', 159.99, 4.4, 12.0),
('Home & Garden', 89.99, 3.9, 18.0),
('Home & Garden', 249.99, 4.1, 22.0),
('Home & Garden', 39.99, 3.7, 10.0),
('Home & Garden', 199.99, 4.5, 25.0),

-- Books
('Books', 19.99, 4.3, 0.0),
('Books', 24.99, 4.6, 5.0),
('Books', 14.99, 4.0, 0.0),
('Books', 29.99, 4.4, 10.0),
('Books', 34.99, 4.7, 15.0),

-- Sports
('Sports', 79.99, 4.1, 20.0),
('Sports', 149.99, 4.4, 25.0),
('Sports', 199.99, 4.2, 30.0),
('Sports', 59.99, 3.8, 15.0),
('Sports', 299.99, 4.6, 35.0);

-- Insert inventory data for all products
INSERT INTO inventory (product_id, stock, delivery_time, seller_rating) VALUES
(1, 50, 2, 4.3),
(2, 25, 1, 4.7),
(3, 100, 3, 4.1),
(4, 15, 1, 4.8),
(5, 75, 2, 4.2),
(6, 200, 4, 4.0),
(7, 150, 3, 4.4),
(8, 300, 5, 3.9),
(9, 80, 2, 4.6),
(10, 120, 3, 4.3),
(11, 60, 2, 4.2),
(12, 90, 4, 4.0),
(13, 40, 3, 4.5),
(14, 180, 5, 3.8),
(15, 70, 2, 4.4),
(16, 500, 7, 4.1),
(17, 300, 5, 4.5),
(18, 400, 6, 4.2),
(19, 250, 4, 4.3),
(20, 150, 3, 4.6),
(21, 80, 2, 4.0),
(22, 60, 3, 4.4),
(23, 45, 2, 4.2),
(24, 120, 4, 3.9),
(25, 35, 1, 4.7);

-- Insert sample user search behavior data
-- This creates realistic behavioral patterns for analysis
INSERT INTO user_searches (user_id, query, product_id, impressions, clicks, add_to_cart, purchase_flag) VALUES
-- High-performing electronics
(1, 'laptop', 2, 10, 8, TRUE, TRUE),
(2, 'laptop', 2, 5, 4, TRUE, TRUE),
(3, 'smartphone', 4, 8, 6, TRUE, TRUE),
(4, 'headphones', 1, 12, 9, TRUE, FALSE),
(5, 'tablet', 5, 6, 5, TRUE, TRUE),

-- Medium-performing products
(6, 'jacket', 7, 15, 6, TRUE, FALSE),
(7, 'shoes', 9, 20, 8, FALSE, FALSE),
(8, 'furniture', 11, 8, 3, TRUE, TRUE),
(9, 'book', 17, 25, 15, TRUE, TRUE),
(10, 'sports equipment', 22, 12, 5, TRUE, FALSE),

-- Low-performing products
(11, 'cheap electronics', 3, 30, 3, FALSE, FALSE),
(12, 'basic clothing', 8, 40, 4, FALSE, FALSE),
(13, 'garden tools', 14, 25, 2, FALSE, FALSE),
(14, 'old books', 18, 35, 3, TRUE, FALSE),
(15, 'basic sports', 24, 20, 2, FALSE, FALSE),

-- Additional behavioral data for better analytics
(16, 'premium laptop', 4, 5, 4, TRUE, TRUE),
(17, 'designer clothes', 9, 8, 6, TRUE, TRUE),
(18, 'home decor', 15, 12, 8, TRUE, TRUE),
(19, 'bestseller book', 20, 15, 12, TRUE, TRUE),
(20, 'professional sports', 25, 6, 5, TRUE, TRUE),

-- More diverse search patterns
(21, 'electronics sale', 1, 20, 5, TRUE, FALSE),
(22, 'fashion deals', 6, 25, 7, TRUE, TRUE),
(23, 'home improvement', 13, 18, 4, FALSE, FALSE),
(24, 'educational books', 16, 30, 18, TRUE, TRUE),
(25, 'fitness equipment', 21, 14, 6, TRUE, FALSE);