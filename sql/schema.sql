-- Smart Product Re-Ranking System Database Schema
-- MySQL database schema for products, user searches, and inventory

-- Drop tables if they exist (for clean setup)
DROP TABLE IF EXISTS user_searches;
DROP TABLE IF EXISTS inventory;
DROP TABLE IF EXISTS products;

-- Products table: Core product information
CREATE TABLE products (
    id INT PRIMARY KEY AUTO_INCREMENT,
    category VARCHAR(100) NOT NULL,
    price DECIMAL(10,2) NOT NULL,
    rating DECIMAL(3,2) NOT NULL CHECK (rating >= 0 AND rating <= 5),
    discount DECIMAL(5,2) DEFAULT 0 CHECK (discount >= 0 AND discount <= 100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_category (category),
    INDEX idx_price (price),
    INDEX idx_rating (rating)
);

-- User searches table: Behavioral data for analytics
CREATE TABLE user_searches (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    query VARCHAR(255) NOT NULL,
    product_id INT NOT NULL,
    impressions INT DEFAULT 1,
    clicks INT DEFAULT 0,
    add_to_cart BOOLEAN DEFAULT FALSE,
    purchase_flag BOOLEAN DEFAULT FALSE,
    search_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (product_id) REFERENCES products(id) ON DELETE CASCADE,
    INDEX idx_product_behavior (product_id, clicks, purchase_flag),
    INDEX idx_user_query (user_id, query),
    INDEX idx_timestamp (search_timestamp)
);

-- Inventory table: Stock and seller information
CREATE TABLE inventory (
    product_id INT PRIMARY KEY,
    stock INT NOT NULL CHECK (stock >= 0),
    delivery_time INT NOT NULL CHECK (delivery_time > 0),
    seller_rating DECIMAL(3,2) NOT NULL CHECK (seller_rating >= 0 AND seller_rating <= 5),
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (product_id) REFERENCES products(id) ON DELETE CASCADE,
    INDEX idx_stock_delivery (stock, delivery_time),
    INDEX idx_seller_rating (seller_rating)
);