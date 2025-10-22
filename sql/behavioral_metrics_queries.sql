-- Behavioral Metrics Aggregation Queries
-- Smart Product Re-Ranking System
-- These queries calculate CTR, CVR, and cart-to-purchase ratios per product

-- 1. Calculate CTR (Click-Through Rate) per product
-- CTR = clicks / impressions
SELECT 
    product_id,
    SUM(impressions) as total_impressions,
    SUM(clicks) as total_clicks,
    CASE 
        WHEN SUM(impressions) > 0 THEN ROUND(SUM(clicks) * 1.0 / SUM(impressions), 4)
        ELSE 0 
    END as ctr
FROM user_searches
GROUP BY product_id
HAVING SUM(impressions) > 0
ORDER BY ctr DESC;

-- 2. Calculate CVR (Conversion Rate) per product  
-- CVR = purchases / clicks
SELECT 
    product_id,
    SUM(clicks) as total_clicks,
    SUM(CASE WHEN purchase_flag = TRUE THEN 1 ELSE 0 END) as total_purchases,
    CASE 
        WHEN SUM(clicks) > 0 THEN ROUND(SUM(CASE WHEN purchase_flag = TRUE THEN 1 ELSE 0 END) * 1.0 / SUM(clicks), 4)
        ELSE 0 
    END as cvr
FROM user_searches
GROUP BY product_id
HAVING SUM(clicks) > 0
ORDER BY cvr DESC;

-- 3. Calculate Cart-to-Purchase Ratio per product
-- Cart-to-Purchase = purchases / cart_additions
SELECT 
    product_id,
    SUM(CASE WHEN add_to_cart = TRUE THEN 1 ELSE 0 END) as total_cart_additions,
    SUM(CASE WHEN purchase_flag = TRUE THEN 1 ELSE 0 END) as total_purchases,
    CASE 
        WHEN SUM(CASE WHEN add_to_cart = TRUE THEN 1 ELSE 0 END) > 0 
        THEN ROUND(SUM(CASE WHEN purchase_flag = TRUE THEN 1 ELSE 0 END) * 1.0 / SUM(CASE WHEN add_to_cart = TRUE THEN 1 ELSE 0 END), 4)
        ELSE 0 
    END as cart_to_purchase_ratio
FROM user_searches
GROUP BY product_id
HAVING SUM(CASE WHEN add_to_cart = TRUE THEN 1 ELSE 0 END) > 0
ORDER BY cart_to_purchase_ratio DESC;

-- 4. Combined behavioral metrics query
-- All metrics in one query for efficient data extraction
SELECT 
    us.product_id,
    p.category,
    p.price,
    p.rating,
    p.discount,
    SUM(us.impressions) as total_impressions,
    SUM(us.clicks) as total_clicks,
    SUM(CASE WHEN us.add_to_cart = TRUE THEN 1 ELSE 0 END) as total_cart_additions,
    SUM(CASE WHEN us.purchase_flag = TRUE THEN 1 ELSE 0 END) as total_purchases,
    
    -- CTR calculation
    CASE 
        WHEN SUM(us.impressions) > 0 THEN ROUND(SUM(us.clicks) * 1.0 / SUM(us.impressions), 4)
        ELSE 0 
    END as ctr,
    
    -- CVR calculation
    CASE 
        WHEN SUM(us.clicks) > 0 THEN ROUND(SUM(CASE WHEN us.purchase_flag = TRUE THEN 1 ELSE 0 END) * 1.0 / SUM(us.clicks), 4)
        ELSE 0 
    END as cvr,
    
    -- Cart-to-Purchase ratio calculation
    CASE 
        WHEN SUM(CASE WHEN us.add_to_cart = TRUE THEN 1 ELSE 0 END) > 0 
        THEN ROUND(SUM(CASE WHEN us.purchase_flag = TRUE THEN 1 ELSE 0 END) * 1.0 / SUM(CASE WHEN us.add_to_cart = TRUE THEN 1 ELSE 0 END), 4)
        ELSE 0 
    END as cart_to_purchase_ratio,
    
    COUNT(DISTINCT us.user_id) as unique_users
FROM user_searches us
JOIN products p ON us.product_id = p.id
GROUP BY us.product_id, p.category, p.price, p.rating, p.discount
HAVING SUM(us.impressions) > 0
ORDER BY total_purchases DESC, ctr DESC;

-- 5. Product performance summary with inventory data
-- Includes seller and stock information for comprehensive analysis
SELECT 
    us.product_id,
    p.category,
    p.price,
    p.rating,
    p.discount,
    i.stock,
    i.delivery_time,
    i.seller_rating,
    
    -- Behavioral metrics
    SUM(us.impressions) as total_impressions,
    SUM(us.clicks) as total_clicks,
    SUM(CASE WHEN us.add_to_cart = TRUE THEN 1 ELSE 0 END) as total_cart_additions,
    SUM(CASE WHEN us.purchase_flag = TRUE THEN 1 ELSE 0 END) as total_purchases,
    
    -- Calculated ratios
    CASE 
        WHEN SUM(us.impressions) > 0 THEN ROUND(SUM(us.clicks) * 1.0 / SUM(us.impressions), 4)
        ELSE 0 
    END as ctr,
    
    CASE 
        WHEN SUM(us.clicks) > 0 THEN ROUND(SUM(CASE WHEN us.purchase_flag = TRUE THEN 1 ELSE 0 END) * 1.0 / SUM(us.clicks), 4)
        ELSE 0 
    END as cvr,
    
    CASE 
        WHEN SUM(CASE WHEN us.add_to_cart = TRUE THEN 1 ELSE 0 END) > 0 
        THEN ROUND(SUM(CASE WHEN us.purchase_flag = TRUE THEN 1 ELSE 0 END) * 1.0 / SUM(CASE WHEN us.add_to_cart = TRUE THEN 1 ELSE 0 END), 4)
        ELSE 0 
    END as cart_to_purchase_ratio,
    
    COUNT(DISTINCT us.user_id) as unique_users
FROM user_searches us
JOIN products p ON us.product_id = p.id
JOIN inventory i ON us.product_id = i.product_id
GROUP BY us.product_id, p.category, p.price, p.rating, p.discount, i.stock, i.delivery_time, i.seller_rating
HAVING SUM(us.impressions) > 0
ORDER BY total_purchases DESC, ctr DESC;

-- 6. Data quality validation queries
-- Check for products with behavioral data
SELECT 
    'Products with behavioral data' as metric,
    COUNT(DISTINCT product_id) as count
FROM user_searches
WHERE impressions > 0

UNION ALL

SELECT 
    'Products with clicks' as metric,
    COUNT(DISTINCT product_id) as count
FROM user_searches
WHERE clicks > 0

UNION ALL

SELECT 
    'Products with purchases' as metric,
    COUNT(DISTINCT product_id) as count
FROM user_searches
WHERE purchase_flag = TRUE

UNION ALL

SELECT 
    'Products with cart additions' as metric,
    COUNT(DISTINCT product_id) as count
FROM user_searches
WHERE add_to_cart = TRUE;