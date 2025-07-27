# 📊 Customer Sales & Marketing Analytics Using SQL and Python
________________________________________
## 🧾 1. Summary
This project aims to analyze customer purchase behavior, sales performance, and profitability using a blend of SQL for data preparation and exploratory analysis, Python for predictive modeling and segmentation, and (optionally) Power BI for dashboarding. It extracts actionable insights to optimize sales strategies, marketing efforts, and product offerings.
________________________________________
## 📦 2. Dataset Overview
1.	Source: Exported from SQL Server (SSMS) using SELECT TOP 1000
2.	File: superstore_cleaned.csv
3.	Size: 1,000 rows, 26 columns
4.	Type: Tabular, transactional
5.	Date Range: Customer acquisition data from 2021–2022
6.	Features: Customer demographics, sales amounts by product category, income, recency, response to campaigns, and visit behavior
________________________________________
## 🎯 3. Project Goal
1.	Clean and transform raw transactional sales data using SQL
2.	Perform exploratory data analysis (EDA) to uncover hidden patterns
3.	Apply machine learning models in Python for:
4.	Predicting income levels
5.	Segmenting customers using clustering
6.	Forecasting sales trends over time
7.	Support data-driven decisions for marketing and operational strategy
________________________________________
## 🔍 4. Problem Statement
How can a business use historical customer and sales data to:
1.	Improve profitability?
2.	Identify high-value customers and regions?
3.	Optimize discount and campaign strategies?
4.	Predict future sales patterns?
________________________________________
## 📊 5. KPIs (Key Performance Indicators)
1.	✅ Total Sales
2.	✅ Total Profit
3.	✅ Profit Margin %
4.	✅ Average Discount %
5.	✅ Customer Segmentation (by education, marital status, etc.)
6.	✅ Monthly Sales Trends
7.	✅ Top Products by Sales and Profit
________________________________________
## 📈 6. Chart Requirements (Optional: Power BI)
1.	Line Chart: Monthly Wine Sales and Profit
2.	Bar Chart: Spend by Category
3.	Pie Chart: Orders by Segment
4.	Heatmap: Discount vs Profitability
5.	Map: Regional Spend/Profit (if regional data exists)
6.	Table: Top 10 Customers or Products by Profit
________________________________________
## 🧠 7. Exploratory Data Analysis (EDA) in SQL
### ✅ EDA Steps Performed:
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
## 🔢 8. Modeling (Python - Step 14 Implementation)
#### 📌 A. Linear Regression: Income Prediction
1.	Goal: Predict customer income using features like spending, family size, and education
2.	Tools: LinearRegression from scikit-learn
3.	Preprocessing: Label-encoding of categorical features
4.	Model Score: R² = 0.127 (Baseline, low due to high income variability)
________________________________________
#### 📌 B. KMeans Clustering: Customer Segmentation
1.	Goal: Cluster customers based on product purchase behavior
2.	Input Features: MntWines, MntFruits, MntMeatProducts, etc.
3.	Elbow Method: Chose k=3
4.	Output: Cluster labels added to dataset for segmentation
________________________________________
#### 📌 C. ARIMA Time Series Forecasting: Wine Sales
1.	Goal: Forecast next 6 months of wine sales
2.	Method: ARIMA(1,1,1)
3.	Data: Monthly sum of MntWines, resampled by customer join date
4.	Result: Successfully forecasted wine spending; deprecated frequency warning fixed by switching from 'M' to 'ME'
________________________________________
### 🔍 9. Key Findings
1.	📉 High discounts often correlated with lower profits
2.	🏆 Certain products (e.g., wines) dominate customer spending
3.	👨‍👩‍👧 Customers with higher education and married status tend to spend more
4.	📆 New customer acquisition has seasonal spikes
5.	🧊 Outliers include customers with unrealistically high income or recent/future acquisition dates
________________________________________
### 📋 10. Inference & Business Decisions
Based on analysis:
1.	🔒 Limit discounts on high-performing products to preserve margins
2.	🎯 Target high-value segments (e.g., married, educated)
3.	📈 Use sales trends to stock products in advance of high seasons
4.	📣 Personalize marketing based on clusters discovered via KMeans
________________________________________
### 📌 11. Conclusion
This project demonstrates how structured sales data can be cleaned, analyzed, and modeled to inform strategic decisions. SQL served as the backbone for EDA and preparation, while Python extended the analysis with predictive modeling and clustering.
________________________________________
### 🧭 12. Future Work
1.	🌐 Integrate real-time data with Power BI dashboards
2.	🤖 Enhance regression models with feature engineering or nonlinear algorithms (e.g., XGBoost)
3.	⏱ Automate SQL-to-Python pipeline for monthly reporting
4.	📤 Use cluster labels in marketing campaigns and A/B tests
________________________________________
### 💼 Final Deliverables
1. Component	Tool	Status
2. Data Cleaning & EDA	SQL	✅ Completed
3. Income Prediction	Python (Linear Regression)	✅ Completed
4. Customer Segmentation	Python (KMeans)	✅ Completed
5. Time Series Forecasting	Python (ARIMA)	✅ Completed
6. Data Export from SQL	SSMS to CSV	✅ Completed
7. Visualization	Power BI (optional)	❌ Skipped (as per scope)
8. SQL & Python Steps Below : 

### ✅ Exploratory Data Analysis & Cleaning Steps in SQL

#### ✅ 1. Data Profiling

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

#### ✅ 2. Date Conversions & Derived Fields

SELECT
  *,
  TRY_CAST(Dt_Customer AS date)                               AS CustomerDate,
  FORMAT(TRY_CAST(Dt_Customer AS date), 'yyyy-MM')            AS CustYearMonth,
  DATEDIFF(day, TRY_CAST(Dt_Customer AS date), GETDATE())     AS DaysSinceCustomerDate
FROM SalesDB.dbo.superstore_data;

#### ✅ 3. Missing Data Handling

UPDATE SalesDB.dbo.superstore_data
SET
  Income     = COALESCE(Income, 0),
  Kidhome    = COALESCE(Kidhome, 0),
  Teenhome   = COALESCE(Teenhome, 0);

#### ✅ 4. Data Type Consistency & Validation

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

#### ✅ 5. Outlier Detection
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

#### ✅ 6. Duplicate Removal
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

#### ✅ 7. Normalize Categorical Fields

UPDATE SalesDB.dbo.superstore_data
SET 
  Education      = UPPER(LTRIM(RTRIM(Education))),
  Marital_Status = UPPER(LTRIM(RTRIM(Marital_Status)));

#### ✅ 8. Derived Columns

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

 #### ✅ 9. Segmentation & Aggregation

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
 
### SQL Query :

###### ✅ Complete EDA & Cleaning Steps in SQL

####### ✅ 1. Data Profiling

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

###### ✅ 2. Date Conversions & Derived Fields

SELECT
  *,
  TRY_CAST(Dt_Customer AS date)                               AS CustomerDate,
  FORMAT(TRY_CAST(Dt_Customer AS date), 'yyyy-MM')            AS CustYearMonth,
  DATEDIFF(day, TRY_CAST(Dt_Customer AS date), GETDATE())     AS DaysSinceCustomerDate
FROM SalesDB.dbo.superstore_data;

###### ✅ 3. Missing Data Handling

UPDATE SalesDB.dbo.superstore_data
SET
  Income     = COALESCE(Income, 0),
  Kidhome    = COALESCE(Kidhome, 0),
  Teenhome   = COALESCE(Teenhome, 0);

###### ✅ 4. Data Type Consistency & Validation

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

###### ✅ 5. Outlier Detection
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

###### ✅ 6. Duplicate Removal
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

#### ✅ 7. Normalize Categorical Fields

UPDATE SalesDB.dbo.superstore_data
SET 
  Education      = UPPER(LTRIM(RTRIM(Education))),
  Marital_Status = UPPER(LTRIM(RTRIM(Marital_Status)));

###### ✅ 8. Derived Columns

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

 ###### ✅ 9. Segmentation & Aggregation

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



<img width="1920" height="1080" alt="Screenshot of Column Names" src="https://github.com/user-attachments/assets/914c4592-1db5-4227-90b1-0fe34ca1fc9a" />


## 🔧 Step-by-Step: Clean & Prep for ML in Jupyter

#### ✅ Step 1: Load the Data (No Headers)
I exported from SSMS 21 via "Select Top 1000 Rows", the CSV probably doesn't include column headers.

import pandas as pd

#### Load without headers
df = pd.read_csv(r"D:\Software\For Other\AI\Data Science\All_Notes\Projects Notes\Projects\SQL project Market analysis\superstore_cleaned.csv", header=None)

#### Optional: check shape
print(df.shape)
df.head()
Output:
(1000, 26)

#### ✅ Step 2: Add Meaningful Column Names¶
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

<img width="1205" height="744" alt="Assign Column Names manually  Based on Screenshot of Column Names image" src="https://github.com/user-attachments/assets/b563f07a-5f8c-46bf-b6aa-4a915bc908b0" />


#### ✅ Step 3: Convert Date Column to datetime
df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'])

#### ✅ Step 4: Handle Missing or Negative Income
# Replace negative income with NaN and drop them

df['Income'] = df['Income'].apply(lambda x: x if x > 0 else None)
df.dropna(subset=['Income'], inplace=True)

### 🔢 PART 1: Linear Regression

#### 🎯 Predict income using other features

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

#### Encode categorical columns
for col in ['Education', 'Marital_Status']:
    df[col] = LabelEncoder().fit_transform(df[col])

#### Features and target
X = df[['Kidhome', 'Teenhome', 'Recency', 'MntWines', 'MntFruits',
        'MntMeatProducts', 'MntFishProducts', 'MntGoldProds',
        'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases',
        'NumStorePurchases', 'NumWebVisitsMonth', 'Education', 'Marital_Status']]
y = df['Income']

#### Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#### Model
lr = LinearRegression()
lr.fit(X_train, y_train)

print("R^2 Score:", lr.score(X_test, y_test))
Output:
R^2 Score: 0.127186987101495

<img width="1170" height="572" alt="Linear Regression Score" src="https://github.com/user-attachments/assets/a691198b-981c-459a-ae76-b7db1b708528" />


### 📊 PART 2: KMeans Clustering
### 🎯 Group customers based on spending behavior
import os
os.environ["OMP_NUM_THREADS"] = "4"

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

#### Cluster only on spend amounts
X_kmeans = df[['MntWines', 'MntFruits', 'MntMeatProducts', 
               'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']]

#### Elbow method to find best k
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

#### Fit with chosen k
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_kmeans)

df[['Cluster'] + X_kmeans.columns.tolist()].head()

<img width="1013" height="671" alt="KMeans Clustering - Group customers based on spending behavior" src="https://github.com/user-attachments/assets/fd359050-287c-417f-9f6b-7644c45a759b" />


#### 📈 PART 3: ARIMA (Time Series)
#### 🎯 Forecast overall wine spending over time
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

#### Group by date and sum wine spending, resample monthly with 'ME'
df_ts = df.groupby('Dt_Customer')['MntWines'].sum().resample('ME').sum()

#### Plot
df_ts.plot(title='Monthly Wine Spending')
plt.ylabel("Wine Spend ($)")
plt.show()

#### ARIMA Modeling
model = ARIMA(df_ts, order=(1,1,1))  # (p,d,q)
model_fit = model.fit()
forecast = model_fit.forecast(steps=6)

print("Next 6 months forecast:")
print(forecast)
<img width="815" height="619" alt="ARIMA (Time Series) Forecast overall wine spending over time" src="https://github.com/user-attachments/assets/dd7cc034-9fb9-4e2a-a68f-c49020bf9dbd" />

--

## 📊 Project Step: Data Loading and Initial Inspection

✅ Step 1: Load the Data (No Headers)
I exported from SSMS 21 via "Select Top 1000 Rows", the CSV probably doesn't include column headers.

| **Step No.** | **Objective**                    | **My Action (in Python)**                                                                                                             | **What I Did**                                                                                         |
| ------------ | -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| 1            | Import necessary libraries       | `import pandas as pd`                                                                                                                 | I imported the `pandas` library, which is essential for handling and analyzing structured data.                         |
| 2            | Load dataset without headers     | `df = pd.read_csv(r"D:\Projects\Customer Sales & Marketing Analytics - SQL and Python\datasets\superstore_cleaned.csv", header=None)` | I loaded the dataset while specifying `header=None` because the CSV file does not contain predefined column headers.    |
| 3            | Inspect the shape of the dataset | `print(df.shape)`                                                                                                                     | I checked the shape of the dataset to understand the number of rows and columns, which helps in planning further steps. |
| 4            | Preview the data                 | `df.head()`                                                                                                                           | I viewed the first few records to get an idea of the data structure and contents, helping me identify potential issues. |


<img width="1164" height="655" alt="Step 1 Load the Data No Headers" src="https://github.com/user-attachments/assets/9f5f47e1-5774-4de8-b2a6-d4259fbcf84a" />


<img width="1920" height="1080" alt="Screenshot of Column Names" src="https://github.com/user-attachments/assets/c0d7cc35-f0c1-48f0-aa71-ab867e8c0f13" />

✅ Step 2: Add Meaningful Column Names
| **Step No.** | **Objective**                            | **My Action (in Python)**                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | **What I Did**                                                                                                                                       |
| ------------ | ---------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 5            | Assign descriptive column names manually | `python<br>df.columns = [<br>    'Customer_ID', 'Birth_Year', 'Education', 'Marital_Status', 'Income',<br>    'Kidhome', 'Teenhome', 'Dt_Customer', 'Recency', 'MntWines', 'MntFruits',<br>    'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds',<br>    'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases',<br>    'NumStorePurchases', 'NumWebVisitsMonth', 'AcceptedCmp3', 'AcceptedCmp4',<br>    'AcceptedCmp5', 'AcceptedCmp1', 'AcceptedCmp2', 'Response'<br>]<br>df.head()` | I manually assigned meaningful column names based on a provided screenshot. This improves readability and ensures consistency when analyzing or visualizing the data. |
<img width="1173" height="491" alt="Step 2- Add Meaningful Column Names" src="https://github.com/user-attachments/assets/39f64b00-ba06-4f94-85de-431dd1947ea1" />

✅ Step 3: Convert Date Column to datetime
| **Step No.** | **Objective**                       | **My Action (in Python)**                               | **What I Did**                                                                                                                           |
| ------------ | ----------------------------------- | ------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 6            | Convert `Dt_Customer` to `datetime` | `df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'])` | I converted the `'Dt_Customer'` column to `datetime` format to enable accurate time-based analysis and manipulation, such as calculating customer tenure. |

✅ Step 4: Handle Missing or Negative Income
| **Step No.** | **Objective**                             | **My Action (in Python)**                                           | **What I Did**                                                                                                                                      |
| ------------ | ----------------------------------------- | ------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 7            | Replace negative income values with `NaN` | `df['Income'] = df['Income'].apply(lambda x: x if x > 0 else None)` | I replaced negative values in the `Income` column with `None` (which is interpreted as `NaN`) since negative income is invalid in this context.                      |
| 8            | Remove records with missing income        | `df.dropna(subset=['Income'], inplace=True)`                        | I dropped rows where `Income` is missing or invalid. This ensures the integrity of any income-based analysis or segmentation I plan to perform later in the project. |

✅ Step 5: Linear Regression – 🎯 Predict Income Using Other Features
| **Step No.** | **Objective**                                | **My Action (in Python)**                                                                                                                                                                                                                                                                                                                                                          | **What I Did**                                                                                                        |
| ------------ | -------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| 9            | Import linear regression tools and utilities | `from sklearn.linear_model import LinearRegression`  <br> `from sklearn.model_selection import train_test_split`  <br> `from sklearn.preprocessing import LabelEncoder`                                                                                                                                                                                                            | I imported the required classes and functions from `sklearn` to build and evaluate a linear regression model.         |
| 10           | Encode categorical features                  | `python<br>for col in ['Education', 'Marital_Status']:<br>&nbsp;&nbsp;&nbsp;&nbsp;df[col] = LabelEncoder().fit_transform(df[col])`                                                                                                                                                                                                                                                 | I used `LabelEncoder` to convert categorical columns into numeric format so they can be used in the regression model. |
| 11           | Select input features and target variable    | `python<br>X = df[['Kidhome', 'Teenhome', 'Recency', 'MntWines', 'MntFruits',<br>&nbsp;&nbsp;&nbsp;&nbsp;'MntMeatProducts', 'MntFishProducts', 'MntGoldProds',<br>&nbsp;&nbsp;&nbsp;&nbsp;'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases',<br>&nbsp;&nbsp;&nbsp;&nbsp;'NumStorePurchases', 'NumWebVisitsMonth', 'Education', 'Marital_Status']]<br>y = df['Income']` | I defined the independent variables (`X`) and the dependent variable (`y`) for predicting income.                     |
| 12           | Split data into training and testing sets    | `X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)`                                                                                                                                                                                                                                                                                        | I split the dataset into 80% training and 20% testing to validate model performance.                                  |
| 13           | Train the linear regression model            | `lr = LinearRegression()`  <br> `lr.fit(X_train, y_train)`                                                                                                                                                                                                                                                                                                                         | I instantiated and trained a `LinearRegression` model using the training data.                                        |
| 14           | Evaluate model performance using R² score    | `print("R^2 Score:", lr.score(X_test, y_test))`                                                                                                                                                                                                                                                                                                                                    | I printed the R² score to evaluate how well the model explains the variance in income based on the selected features. |
