-- Key queries for Retailrocket dataset analysis

-- 1. Main aggregation query for behavioral features
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
ORDER BY total_purchases DESC, total_impressions DESC;

-- 2. Data quality check queries
-- Check event distribution
SELECT event, COUNT(*) as count, 
       ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM events), 2) as percentage
FROM events 
GROUP BY event;

-- Check for items with high engagement
SELECT itemid, 
       SUM(CASE WHEN event = 'view' THEN 1 ELSE 0 END) as views,
       SUM(CASE WHEN event = 'addtocart' THEN 1 ELSE 0 END) as carts,
       SUM(CASE WHEN event = 'transaction' THEN 1 ELSE 0 END) as purchases
FROM events 
GROUP BY itemid
HAVING views > 100
ORDER BY purchases DESC, carts DESC
LIMIT 20;

-- 3. User behavior analysis
SELECT visitorid,
       COUNT(*) as total_events,
       SUM(CASE WHEN event = 'view' THEN 1 ELSE 0 END) as views,
       SUM(CASE WHEN event = 'addtocart' THEN 1 ELSE 0 END) as carts,
       SUM(CASE WHEN event = 'transaction' THEN 1 ELSE 0 END) as purchases
FROM events
GROUP BY visitorid
HAVING total_events > 10
ORDER BY purchases DESC, carts DESC
LIMIT 20;