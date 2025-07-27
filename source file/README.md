
📊 Customer Sales & Marketing Analytics Using SQL and Python
________________________________________
🧾 1. Summary
This project aims to analyze customer purchase behavior, sales performance, and profitability using a blend of SQL for data preparation and exploratory analysis, Python for predictive modeling and segmentation, and (optionally) Power BI for dashboarding. It extracts actionable insights to optimize sales strategies, marketing efforts, and product offerings.
________________________________________
📦 2. Dataset Overview
•	Source: Exported from SQL Server (SSMS) using SELECT TOP 1000
•	File: superstore_cleaned.csv
•	Size: 1,000 rows, 26 columns
•	Type: Tabular, transactional
•	Date Range: Customer acquisition data from 2021–2022
•	Features: Customer demographics, sales amounts by product category, income, recency, response to campaigns, and visit behavior
________________________________________
🎯 3. Project Goal
•	Clean and transform raw transactional sales data using SQL
•	Perform exploratory data analysis (EDA) to uncover hidden patterns
•	Apply machine learning models in Python for:
o	Predicting income levels
o	Segmenting customers using clustering
o	Forecasting sales trends over time
•	Support data-driven decisions for marketing and operational strategy
________________________________________
🔍 4. Problem Statement
How can a business use historical customer and sales data to:
•	Improve profitability?
•	Identify high-value customers and regions?
•	Optimize discount and campaign strategies?
•	Predict future sales patterns?
________________________________________
📊 5. KPIs (Key Performance Indicators)
•	✅ Total Sales
•	✅ Total Profit
•	✅ Profit Margin %
•	✅ Average Discount %
•	✅ Customer Segmentation (by education, marital status, etc.)
•	✅ Monthly Sales Trends
•	✅ Top Products by Sales and Profit
________________________________________
📈 6. Chart Requirements (Optional: Power BI)
•	Line Chart: Monthly Wine Sales and Profit
•	Bar Chart: Spend by Category
•	Pie Chart: Orders by Segment
•	Heatmap: Discount vs Profitability
•	Map: Regional Spend/Profit (if regional data exists)
•	Table: Top 10 Customers or Products by Profit
________________________________________
🧠 7. Exploratory Data Analysis (EDA) in SQL
✅ EDA Steps Performed:
1.	Data Profiling: Count rows, distinct values, NULL checks
2.	Date Parsing: TRY_CAST() for Dt_Customer, extract Year-Month
3.	Missing Values: Used COALESCE() or removed invalid records
4.	Outlier Detection: Handled extreme income, future dates, invalid children count
5.	Duplicates: Removed using ROW_NUMBER() partitioning
6.	Derived Fields:
o	TotalSpend: Sum of all category-wise spending
o	ProfitMargin: (TotalSpend - Income)/TotalSpend
o	CustomerYear and CustomerMonth from Dt_Customer
7.	Categorical Normalization: Trimmed and uppercased education/marital status
8.	Segmentation: Grouped by customer attributes for insights
________________________________________
🔢 8. Modeling (Python - Step 14 Implementation)
📌 A. Linear Regression: Income Prediction
•	Goal: Predict customer income using features like spending, family size, and education
•	Tools: LinearRegression from scikit-learn
•	Preprocessing: Label-encoding of categorical features
•	Model Score: R² = 0.127 (Baseline, low due to high income variability)
________________________________________
📌 B. KMeans Clustering: Customer Segmentation
•	Goal: Cluster customers based on product purchase behavior
•	Input Features: MntWines, MntFruits, MntMeatProducts, etc.
•	Elbow Method: Chose k=3
•	Output: Cluster labels added to dataset for segmentation
________________________________________
📌 C. ARIMA Time Series Forecasting: Wine Sales
•	Goal: Forecast next 6 months of wine sales
•	Method: ARIMA(1,1,1)
•	Data: Monthly sum of MntWines, resampled by customer join date
•	Result: Successfully forecasted wine spending; deprecated frequency warning fixed by switching from 'M' to 'ME'
________________________________________
🔍 9. Key Findings
•	📉 High discounts often correlated with lower profits
•	🏆 Certain products (e.g., wines) dominate customer spending
•	👨‍👩‍👧 Customers with higher education and married status tend to spend more
•	📆 New customer acquisition has seasonal spikes
•	🧊 Outliers include customers with unrealistically high income or recent/future acquisition dates
________________________________________
📋 10. Inference & Business Decisions
Based on analysis:
•	🔒 Limit discounts on high-performing products to preserve margins
•	🎯 Target high-value segments (e.g., married, educated)
•	📈 Use sales trends to stock products in advance of high seasons
•	📣 Personalize marketing based on clusters discovered via KMeans
________________________________________
📌 11. Conclusion
This project demonstrates how structured sales data can be cleaned, analyzed, and modeled to inform strategic decisions. SQL served as the backbone for EDA and preparation, while Python extended the analysis with predictive modeling and clustering.
________________________________________
🧭 12. Future Work
•	🌐 Integrate real-time data with Power BI dashboards
•	🤖 Enhance regression models with feature engineering or nonlinear algorithms (e.g., XGBoost)
•	⏱ Automate SQL-to-Python pipeline for monthly reporting
•	📤 Use cluster labels in marketing campaigns and A/B tests
________________________________________
💼 Final Deliverables
Component	Tool	Status
Data Cleaning & EDA	SQL	✅ Completed
Income Prediction	Python (Linear Regression)	✅ Completed
Customer Segmentation	Python (KMeans)	✅ Completed
Time Series Forecasting	Python (ARIMA)	✅ Completed
Data Export from SQL	SSMS to CSV	✅ Completed
Visualization	Power BI (optional)	❌ Skipped (as per scope)
SQL & Python Steps Below : 

✅ Exploratory Data Analysis & Cleaning Steps in SQL

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

SELECT TOP (1000) [Id]
      ,[Year_Birth]
      ,[Education]
      ,[Marital_Status]
      ,[Income]
      ,[Kidhome]
      ,[Teenhome]
      ,[Dt_Customer]
      ,[Recency]
      ,[MntWines]
      ,[MntFruits]
      ,[MntMeatProducts]
      ,[MntFishProducts]
      ,[MntSweetProducts]
      ,[MntGoldProds]
      ,[NumDealsPurchases]
      ,[NumWebPurchases]
      ,[NumCatalogPurchases]
      ,[NumStorePurchases]
      ,[NumWebVisitsMonth]
      ,[Response]
      ,[Complain]
      ,[ProfitMargin]
      ,[CustomerYear]
      ,[CustomerMonth]
      ,[TotalSpend]
  FROM [SalesDB].[dbo].[superstore_data]
Screenshot of Column Names
 



#🔧 Step-by-Step: Clean & Prep for ML in Jupyter

✅ Step 1: Load the Data (No Headers)
I exported from SSMS 21 via "Select Top 1000 Rows", the CSV probably doesn't include column headers.

import pandas as pd

# Load without headers
df = pd.read_csv(r"D:\Software\For Other\AI\Data Science\All_Notes\Projects Notes\Projects\SQL project Market analysis\superstore_cleaned.csv", header=None)

# Optional: check shape
print(df.shape)
df.head()
Output:
(1000, 26)

✅ Step 2: Add Meaningful Column Names¶
assign Column Names manually. Based on Screenshot of Column Names image :
df.columns = [
    'Customer_ID', 'Birth_Year', 'Education', 'Marital_Status', 'Income',
    'Kidhome', 'Teenhome', 'Dt_Customer', 'Recency', 'MntWines', 'MntFruits',
    'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds',
    'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 
    'NumStorePurchases', 'NumWebVisitsMonth', 'AcceptedCmp3', 'AcceptedCmp4',
    'AcceptedCmp5', 'AcceptedCmp1', 'AcceptedCmp2', 'Response'
]
df.head()
✅ Step 3: Convert Date Column to datetime
df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'])
✅ Step 4: Handle Missing or Negative Income
# Replace negative income with NaN and drop them

df['Income'] = df['Income'].apply(lambda x: x if x > 0 else None)
df.dropna(subset=['Income'], inplace=True)
🔢 PART 1: Linear Regression
🎯 Predict income using other features

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Encode categorical columns
for col in ['Education', 'Marital_Status']:
    df[col] = LabelEncoder().fit_transform(df[col])

# Features and target
X = df[['Kidhome', 'Teenhome', 'Recency', 'MntWines', 'MntFruits',
        'MntMeatProducts', 'MntFishProducts', 'MntGoldProds',
        'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases',
        'NumStorePurchases', 'NumWebVisitsMonth', 'Education', 'Marital_Status']]
y = df['Income']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
lr = LinearRegression()
lr.fit(X_train, y_train)

print("R^2 Score:", lr.score(X_test, y_test))
Output:
R^2 Score: 0.127186987101495

📊 PART 2: KMeans Clustering
🎯 Group customers based on spending behavior
import os
os.environ["OMP_NUM_THREADS"] = "4"

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Cluster only on spend amounts
X_kmeans = df[['MntWines', 'MntFruits', 'MntMeatProducts', 
               'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']]

# Elbow method to find best k
inertia = []
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_kmeans)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 10), inertia, marker='o')
plt.title('Elbow Method')
plt.xlabel('k')
plt.ylabel('Inertia')
plt.show()

# Fit with chosen k
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_kmeans)

df[['Cluster'] + X_kmeans.columns.tolist()].head()

📈 PART 3: ARIMA (Time Series)
🎯 Forecast overall wine spending over time
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Group by date and sum wine spending, resample monthly with 'ME'
df_ts = df.groupby('Dt_Customer')['MntWines'].sum().resample('ME').sum()

# Plot
df_ts.plot(title='Monthly Wine Spending')
plt.ylabel("Wine Spend ($)")
plt.show()

# ARIMA Modeling
model = ARIMA(df_ts, order=(1,1,1))  # (p,d,q)
model_fit = model.fit()
forecast = model_fit.forecast(steps=6)

print("Next 6 months forecast:")
print(forecast)


------------------------------------------------------------------------------------


## 📄 License
This project is licensed under the MIT License 

## 🙋‍♂️ Author

📧 [jayeshpardeshi161@gmail.com]
📌 LinkedIn: [ Profile URL]