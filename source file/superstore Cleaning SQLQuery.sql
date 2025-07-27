✅ Complete EDA & Cleaning Steps in SQL

✅ 1. Data Profiling

-- Total records
SELECT COUNT(*) AS TotalRows
FROM SalesDB.dbo.superstore_data;

-- Distinct counts for categorical fields
SELECT 
  COUNT(DISTINCT Education)          AS DistinctEducation,
  COUNT(DISTINCT Marital_Status)     AS DistinctMaritalStatuses
FROM SalesDB.dbo.superstore_data;

-- Null checks in key fields
SELECT 
  SUM(CASE WHEN Income IS NULL THEN 1 END)       AS NullIncome,
  SUM(CASE WHEN Dt_Customer IS NULL THEN 1 END)  AS NullCustomerDate,
  SUM(CASE WHEN Kidhome IS NULL THEN 1 END)      AS NullKidhome,
  SUM(CASE WHEN Teenhome IS NULL THEN 1 END)     AS NullTeenhome
FROM SalesDB.dbo.superstore_data;

✅ 2. Date Conversions & Derived Fields

SELECT
  *,
  TRY_CAST(Dt_Customer AS date)                               AS CustomerDate,
  FORMAT(TRY_CAST(Dt_Customer AS date), 'yyyy-MM')            AS CustYearMonth,
  DATEDIFF(day, TRY_CAST(Dt_Customer AS date), GETDATE())     AS DaysSinceCustomerDate
FROM SalesDB.dbo.superstore_data;

✅ 3. Missing Data Handling

UPDATE SalesDB.dbo.superstore_data
SET
  Income     = COALESCE(Income, 0),
  Kidhome    = COALESCE(Kidhome, 0),
  Teenhome   = COALESCE(Teenhome, 0);

✅ 4. Data Type Consistency & Validation

-- Find income values that can't be converted to numeric (if originally varchar)
SELECT *
FROM SalesDB.dbo.superstore_data
WHERE TRY_CAST(Income AS decimal(18,2)) IS NULL
  AND Income IS NOT NULL;

-- Identify invalid dates
SELECT *
FROM SalesDB.dbo.superstore_data
WHERE TRY_CAST(Dt_Customer AS date) IS NULL
  AND Dt_Customer IS NOT NULL;

✅ 5. Outlier Detection
SELECT *
FROM (
  SELECT
    *,
    DATEDIFF(day, TRY_CAST(Dt_Customer AS date), GETDATE()) AS DaysSinceCustomerDate
  FROM SalesDB.dbo.superstore_data
) AS t
WHERE Income > 200000
   OR DaysSinceCustomerDate < 0
   OR Kidhome < 0
   OR Teenhome < 0;

✅ 6. Duplicate Removal
WITH CustomerCTE AS (
  SELECT *,
    ROW_NUMBER() OVER (
      PARTITION BY Id 
      ORDER BY TRY_CAST(Dt_Customer AS date) DESC
    ) AS rn
  FROM SalesDB.dbo.superstore_data
)
DELETE FROM CustomerCTE 
WHERE rn > 1;

✅ 7. Normalize Categorical Fields

UPDATE SalesDB.dbo.superstore_data
SET 
  Education      = UPPER(LTRIM(RTRIM(Education))),
  Marital_Status = UPPER(LTRIM(RTRIM(Marital_Status)));

✅ 8. Derived Columns

SELECT COLUMN_NAME
FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_NAME = 'superstore_data'
  AND COLUMN_NAME = 'TotalSpend';

ALTER TABLE SalesDB.dbo.superstore_data
DROP COLUMN TotalSpend;

ALTER TABLE SalesDB.dbo.superstore_data
ADD TotalSpend AS (
  MntWines + MntFruits + MntMeatProducts + 
  MntFishProducts + MntSweetProducts + MntGoldProds
);

ALTER TABLE SalesDB.dbo.superstore_data
ALTER COLUMN ProfitMargin DECIMAL(12,4);

UPDATE SalesDB.dbo.superstore_data
SET
  ProfitMargin = CASE 
                   WHEN TotalSpend = 0 THEN NULL
                   ELSE CAST((TotalSpend - Income) AS decimal(18,2)) / TotalSpend * 100 
                 END,
  CustomerYear  = YEAR(TRY_CAST(Dt_Customer AS date)),
  CustomerMonth = MONTH(TRY_CAST(Dt_Customer AS date));

  ✅ 9. Segmentation & Aggregation

-- By Education
SELECT 
  Education,
  AVG(ProfitMargin)  AS AvgProfitPct,
  AVG(Income)        AS AvgIncome,
  SUM(TotalSpend)    AS SumSpend
FROM SalesDB.dbo.superstore_data
GROUP BY Education;

-- By Marital Status
SELECT
  Marital_Status,
  COUNT(*)           AS CountCust,
  AVG(Income)        AS AvgIncome,
  SUM(TotalSpend)    AS TotalSpent
FROM SalesDB.dbo.superstore_data
GROUP BY Marital_Status;

-- Monthly customer counts
SELECT 
  CustYearMonth,
  COUNT(*)           AS CountInMonth
FROM (
  SELECT FORMAT(TRY_CAST(Dt_Customer AS date), 'yyyy-MM') AS CustYearMonth
  FROM SalesDB.dbo.superstore_data
) AS x
GROUP BY CustYearMonth
ORDER BY CustYearMonth;

SELECT * FROM SalesDB.dbo.superstore_data_cleaned;
