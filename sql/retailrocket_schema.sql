-- Retailrocket Dataset Schema for SQLite/MySQL
-- Schema for events.csv data

-- Create events table for Retailrocket dataset
CREATE TABLE IF NOT EXISTS events (
    timestamp BIGINT NOT NULL,
    visitorid INT NOT NULL,
    event VARCHAR(20) NOT NULL,
    itemid INT NOT NULL,
    transactionid VARCHAR(50),
    INDEX idx_itemid (itemid),
    INDEX idx_event (event),
    INDEX idx_visitor_item (visitorid, itemid)
);

-- Data aggregation query to pivot events by itemid
-- This query creates the behavioral features needed for the ML model
CREATE VIEW product_behavior_summary AS
SELECT 
    itemid as product_id,
    SUM(CASE WHEN event = 'view' THEN 1 ELSE 0 END) as total_impressions,
    SUM(CASE WHEN event = 'addtocart' THEN 1 ELSE 0 END) as total_carts,
    SUM(CASE WHEN event = 'transaction' THEN 1 ELSE 0 END) as total_purchases,
    COUNT(DISTINCT visitorid) as unique_visitors,
    COUNT(*) as total_events
FROM events 
GROUP BY itemid
HAVING total_impressions > 0  -- Only include items that have been viewed
ORDER BY total_purchases DESC, total_impressions DESC;