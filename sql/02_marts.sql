
DROP TABLE IF EXISTS marts.dim_customer CASCADE;

CREATE TABLE marts.dim_customer AS
SELECT
    "CustomerID"                                AS customer_id,
    MIN("Country")                              AS country,
    COUNT(DISTINCT "InvoiceNo")                 AS n_orders,
    SUM("Quantity")                             AS total_items,
    SUM("TotalPrice")                           AS total_revenue,
    MIN("InvoiceDate")                          AS first_order_at,
    MAX("InvoiceDate")                          AS last_order_at,
    EXTRACT(DAY FROM MAX("InvoiceDate") - MIN("InvoiceDate")) AS lifetime_days
FROM staging.retail_clean
GROUP BY "CustomerID";

CREATE INDEX idx_dim_customer_id ON marts.dim_customer(customer_id);


-- ---------- dim_product ----------
DROP TABLE IF EXISTS marts.dim_product CASCADE;

CREATE TABLE marts.dim_product AS
SELECT
    "StockCode"                     AS stock_code,
    MIN("Description")              AS description,
    AVG("UnitPrice")                AS avg_unit_price,
    SUM("Quantity")                 AS total_sold,
    COUNT(DISTINCT "CustomerID")    AS n_customers,
    COUNT(DISTINCT "InvoiceNo")     AS n_orders
FROM staging.retail_clean
GROUP BY "StockCode";

CREATE INDEX idx_dim_product_code ON marts.dim_product(stock_code);


-- ---------- fact_sales ----------
DROP TABLE IF EXISTS marts.fact_sales CASCADE;

CREATE TABLE marts.fact_sales AS
SELECT
    "InvoiceNo"       AS invoice_no,
    "StockCode"       AS stock_code,
    "CustomerID"      AS customer_id,
    "InvoiceDate"     AS invoice_at,
    "Quantity"        AS quantity,
    "UnitPrice"       AS unit_price,
    "TotalPrice"      AS total_price,
    "Country"         AS country
FROM staging.retail_clean;

CREATE INDEX idx_fact_sales_customer ON marts.fact_sales(customer_id);
CREATE INDEX idx_fact_sales_date     ON marts.fact_sales(invoice_at);


-- ---------- customer_sequences ----------
-- Здесь самая важная витрина для рекомендаций:
-- для каждого клиента — упорядоченный по времени массив StockCode.
-- Оставляем только клиентов с достаточной историей (>= 10 покупок),
-- чтобы у DL-модели было на чём учиться.

DROP TABLE IF EXISTS marts.customer_sequences CASCADE;

CREATE TABLE marts.customer_sequences AS
WITH ordered AS (
    SELECT
        "CustomerID"  AS customer_id,
        "StockCode"   AS stock_code,
        "InvoiceDate" AS invoice_at,
        ROW_NUMBER() OVER (
            PARTITION BY "CustomerID"
            ORDER BY "InvoiceDate", "InvoiceNo"
        ) AS seq_pos
    FROM staging.retail_clean
),
seq AS (
    SELECT
        customer_id,
        ARRAY_AGG(stock_code ORDER BY seq_pos) AS item_sequence,
        COUNT(*)                               AS seq_len,
        MIN(invoice_at)                        AS first_at,
        MAX(invoice_at)                        AS last_at
    FROM ordered
    GROUP BY customer_id
)
SELECT *
FROM seq
WHERE seq_len >= 10;

CREATE INDEX idx_customer_seq_id ON marts.customer_sequences(customer_id);