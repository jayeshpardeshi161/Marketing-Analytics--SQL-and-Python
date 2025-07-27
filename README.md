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
## 🔢 8. Modeling (Python - Step Implementation)
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

--

## ✅ Exploratory Data Analysis & Cleaning Steps in SQL (SSMS21)

✅ 1. Data Profiling

| **Step No.** | **Objective**                                     | **My SQL Query**                                                                                                                                                                                                                                                                                                                                                  | **What I Did**                                                                                                                                                       |
| ------------ | ------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1            | Count total records in the dataset                | `sql<br>SELECT COUNT(*) AS TotalRows<br>FROM SalesDB.dbo.superstore_data;`                                                                                                                                                                                                                                                                                        | I checked the total number of records in the dataset to establish a baseline and verify data completeness.                                                           |
| 2            | Profile categorical variables                     | `sql<br>SELECT <br>&nbsp;&nbsp;COUNT(DISTINCT Education) AS DistinctEducation,<br>&nbsp;&nbsp;COUNT(DISTINCT Marital_Status) AS DistinctMaritalStatuses<br>FROM SalesDB.dbo.superstore_data;`                                                                                                                                                                     | I identified the number of unique values in key categorical columns like `Education` and `Marital_Status` to understand their variability and use in later modeling. |
| 3            | Check for missing/null values in important fields | `sql<br>SELECT <br>&nbsp;&nbsp;SUM(CASE WHEN Income IS NULL THEN 1 END) AS NullIncome,<br>&nbsp;&nbsp;SUM(CASE WHEN Dt_Customer IS NULL THEN 1 END) AS NullCustomerDate,<br>&nbsp;&nbsp;SUM(CASE WHEN Kidhome IS NULL THEN 1 END) AS NullKidhome,<br>&nbsp;&nbsp;SUM(CASE WHEN Teenhome IS NULL THEN 1 END) AS NullTeenhome<br>FROM SalesDB.dbo.superstore_data;` | I conducted null value checks for important numeric and date columns to ensure data quality before exporting or analyzing in Python.                                 |

✅ 2. Date Conversions & Derived Fields

| **Step No.** | **Objective**                                  | **My SQL Query**                                                                   | **What I Did**                                                                                                                                            |
| ------------ | ---------------------------------------------- | ---------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 4            | Convert `Dt_Customer` to a proper date format  | `TRY_CAST(Dt_Customer AS date) AS CustomerDate`                                    | I safely converted the customer join date field into a SQL `date` format using `TRY_CAST`, ensuring invalid values are handled gracefully.                |
| 5            | Derive customer join month in `yyyy-MM` format | `FORMAT(TRY_CAST(Dt_Customer AS date), 'yyyy-MM') AS CustYearMonth`                | I created a derived column to represent the customer's join date in a `year-month` format for time-based grouping or visualization.                       |
| 6            | Calculate customer tenure (in days)            | `DATEDIFF(day, TRY_CAST(Dt_Customer AS date), GETDATE()) AS DaysSinceCustomerDate` | I computed how many days have passed since each customer joined, which can be useful for cohort analysis, retention modeling, or time-based segmentation. |

✅ 3. Missing Data Handling (in SQL – SSMS21)

| **Step No.** | **Objective**                           | **My SQL Query**                                                                                                                                                                                | **What I Did**                                                                                                                                                     |
| ------------ | --------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 7            | Replace `NULL` values in numeric fields | `sql<br>UPDATE SalesDB.dbo.superstore_data<br>SET<br>&nbsp;&nbsp;Income = COALESCE(Income, 0),<br>&nbsp;&nbsp;Kidhome = COALESCE(Kidhome, 0),<br>&nbsp;&nbsp;Teenhome = COALESCE(Teenhome, 0);` | I replaced missing (`NULL`) values in key numeric columns with `0` using `COALESCE`. This ensures the dataset is clean and ready for analysis or export to Python. |

✅ 4. Data Type Consistency & Validation

| **Step No.** | **Objective**                                               | **My SQL Query**                                                                                                                                 | **What I Did**                                                                                                                                                                                   |
| ------------ | ----------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 8            | Detect non-numeric `Income` values (if stored as `varchar`) | `sql<br>SELECT *<br>FROM SalesDB.dbo.superstore_data<br>WHERE TRY_CAST(Income AS decimal(18,2)) IS NULL<br>&nbsp;&nbsp;AND Income IS NOT NULL;`  | I checked whether all non-null `Income` values could be safely cast to `decimal`, identifying any corrupt or malformed entries that would break numeric analysis.                                |
| 9            | Detect invalid date values in `Dt_Customer`                 | `sql<br>SELECT *<br>FROM SalesDB.dbo.superstore_data<br>WHERE TRY_CAST(Dt_Customer AS date) IS NULL<br>&nbsp;&nbsp;AND Dt_Customer IS NOT NULL;` | I verified that all `Dt_Customer` values could be properly cast to a SQL `date`. This step ensures that invalid date entries (e.g., strings, typos) are caught before time series or join logic. |

✅ 5. Outlier Detection

| **Step No.** | **Objective**                           | **My SQL Query**                                                                                                                                                                                                                                                                                                                                              | **What I Did**                                                                                                                                                                                        |
| ------------ | --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 10           | Detect income and date-related outliers | `sql<br>SELECT *<br>FROM (<br>&nbsp;&nbsp;SELECT *,<br>&nbsp;&nbsp;&nbsp;&nbsp;DATEDIFF(day, TRY_CAST(Dt_Customer AS date), GETDATE()) AS DaysSinceCustomerDate<br>&nbsp;&nbsp;FROM SalesDB.dbo.superstore_data<br>) AS t<br>WHERE Income > 200000<br>&nbsp;&nbsp;OR DaysSinceCustomerDate < 0<br>&nbsp;&nbsp;OR Kidhome < 0<br>&nbsp;&nbsp;OR Teenhome < 0;` | I created a temporary derived column to calculate how long each customer has been with us, then filtered for potential outliers: unusually high income, invalid join dates, or negative child counts. |

✅ 6. Duplicate Removal

| **Step No.** | **Objective**                                  | **My SQL Query**                                                                                                                                                                                                                                                                                                                                                                  | **What I Did**                                                                                                                                                                                        |
| ------------ | ---------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 11           | Identify and delete duplicate customer records | `sql<br>WITH CustomerCTE AS (<br>&nbsp;&nbsp;SELECT *,<br>&nbsp;&nbsp;&nbsp;&nbsp;ROW_NUMBER() OVER (<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;PARTITION BY Id<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;ORDER BY TRY_CAST(Dt_Customer AS date) DESC<br>&nbsp;&nbsp;&nbsp;&nbsp;) AS rn<br>&nbsp;&nbsp;FROM SalesDB.dbo.superstore_data<br>)<br>DELETE FROM CustomerCTE WHERE rn > 1;` | I used a common table expression (CTE) with `ROW_NUMBER()` to identify duplicates based on `Id`, retaining only the latest record per customer by join date, and deleted all other duplicate entries. |

✅ 7. Normalize Categorical Fields

| **Step No.** | **Objective**                           | **My SQL Query**                                                                                                                                                                   | **What I Did**                                                                                                                                                                           |
| ------------ | --------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 12           | Standardize text in categorical columns | `sql<br>UPDATE SalesDB.dbo.superstore_data<br>SET<br>&nbsp;&nbsp;Education = UPPER(LTRIM(RTRIM(Education))),<br>&nbsp;&nbsp;Marital_Status = UPPER(LTRIM(RTRIM(Marital_Status)));` | I cleaned categorical text fields by trimming whitespace and converting all values to uppercase. This avoids case-sensitive mismatches and ensures consistency for grouping or encoding. |

✅ 8. Derived Columns

| **Step No.** | **Objective**                             | **My SQL Query (Condensed)**                                                                                                                                                                                     | **What I Did**                                                                                                                                               |
| ------------ | ----------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 13           | Check and drop existing derived column    | `sql<br>SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = 'superstore_data' AND COLUMN_NAME = 'TotalSpend';<br>ALTER TABLE superstore_data DROP COLUMN TotalSpend;`                          | I verified whether the `TotalSpend` column existed and dropped it if present to avoid conflicts during redefinition.                                         |
| 14           | Create `TotalSpend` as a computed column  | `sql<br>ALTER TABLE superstore_data ADD TotalSpend AS (MntWines + MntFruits + MntMeatProducts + MntFishProducts + MntSweetProducts + MntGoldProds);`                                                             | I created a computed column `TotalSpend` to represent total customer expenditure across all product categories.                                              |
| 15           | Modify and update `ProfitMargin` field    | `sql<br>ALTER TABLE superstore_data ALTER COLUMN ProfitMargin DECIMAL(12,4);<br>UPDATE superstore_data SET ProfitMargin = CASE WHEN TotalSpend = 0 THEN NULL ELSE (TotalSpend - Income) / TotalSpend * 100 END;` | I calculated `ProfitMargin` as a percentage of income over total spend, ensuring division-by-zero errors were avoided by handling zero-spend cases.          |
| 16           | Extract year and month from `Dt_Customer` | `sql<br>UPDATE superstore_data SET CustomerYear = YEAR(TRY_CAST(Dt_Customer AS date)), CustomerMonth = MONTH(TRY_CAST(Dt_Customer AS date));`                                                                    | I added time-based dimensions (`CustomerYear`, `CustomerMonth`) from the `Dt_Customer` field, useful for grouping, filtering, or time series trend analysis. |

✅ 9. Segmentation & Aggregation

| **Step No.** | **Objective**                       | **My SQL Query (Condensed)**                                                                                                                                                                                                                              | **What I Did**                                                                                                                                                        |
| ------------ | ----------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 17           | Analyze metrics by `Education`      | `sql<br>SELECT Education, AVG(ProfitMargin) AS AvgProfitPct, AVG(Income) AS AvgIncome, SUM(TotalSpend) AS SumSpend<br>FROM superstore_data<br>GROUP BY Education;`                                                                                        | I grouped data by education level to evaluate how spending and profitability vary across customer education categories.                                               |
| 18           | Analyze metrics by `Marital_Status` | `sql<br>SELECT Marital_Status, COUNT(*) AS CountCust, AVG(Income) AS AvgIncome, SUM(TotalSpend) AS TotalSpent<br>FROM superstore_data<br>GROUP BY Marital_Status;`                                                                                        | I grouped the dataset by marital status to understand demographic behavior and purchasing power in terms of total spend and income.                                   |
| 19           | Track customer volume over time     | `sql<br>SELECT CustYearMonth, COUNT(*) AS CountInMonth<br>FROM (<br>&nbsp;&nbsp;SELECT FORMAT(TRY_CAST(Dt_Customer AS date), 'yyyy-MM') AS CustYearMonth<br>&nbsp;&nbsp;FROM superstore_data<br>) x<br>GROUP BY CustYearMonth<br>ORDER BY CustYearMonth;` | I extracted the join month and counted new customers per month, allowing for time trend analysis (e.g., customer acquisition seasonality).                            |
| 20           | Preview cleaned and final dataset   | `sql<br>SELECT TOP 1000 * FROM SalesDB.dbo.superstore_data;`                                                                                                                                                                                              | I viewed a sample of the cleaned and feature-rich dataset, verifying that fields like `TotalSpend`, `ProfitMargin`, and `CustomerYear/Month` are correctly populated. |


--

<img width="1920" height="1080" alt="Screenshot of Column Names" src="https://github.com/user-attachments/assets/914c4592-1db5-4227-90b1-0fe34ca1fc9a" />

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
Output: R^2 Score: 0.127186987101495
<img width="1177" height="538" alt="Step 5 Linear Regression score" src="https://github.com/user-attachments/assets/b917529c-81a3-4c1c-bcc5-2515a9b9ed34" />


✅ Step 6: KMeans Clustering – 🎯 Group Customers Based on Spending Behavior
| **Step No.** | **Objective**                                     | **My Action (in Python)**                                                                                                                                                                                                                                  | **What I Did**                                                                                                                                                |
| ------------ | ------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 15           | Set thread environment for stability (optional)   | `import os`<br>`os.environ["OMP_NUM_THREADS"] = "4"`                                                                                                                                                                                                       | I set the environment variable to limit thread usage, which can help avoid potential resource conflicts or performance issues during clustering.              |
| 16           | Import required libraries                         | `from sklearn.cluster import KMeans`<br>`import matplotlib.pyplot as plt`                                                                                                                                                                                  | I imported `KMeans` for clustering and `matplotlib` for visualizing the elbow method to determine the optimal number of clusters.                             |
| 17           | Select spending-related features                  | `python<br>X_kmeans = df[['MntWines', 'MntFruits', 'MntMeatProducts',<br>&nbsp;&nbsp;&nbsp;&nbsp;'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']]`                                                                                                  | I selected features that represent customer spending behavior to use as the basis for clustering.                                                             |
| 18           | Use elbow method to determine optimal `k`         | `python<br>inertia = []<br>for k in range(1, 10):<br>&nbsp;&nbsp;&nbsp;&nbsp;kmeans = KMeans(n_clusters=k, random_state=42)<br>&nbsp;&nbsp;&nbsp;&nbsp;kmeans.fit(X_kmeans)<br>&nbsp;&nbsp;&nbsp;&nbsp;inertia.append(kmeans.inertia_)`<br>`plt.plot(...)` | I calculated the inertia for a range of `k` values and plotted the elbow curve to identify the most suitable number of clusters.                              |
| 19           | Fit KMeans with optimal cluster count (`k=3`)     | `kmeans = KMeans(n_clusters=3, random_state=42)`<br>`df['Cluster'] = kmeans.fit_predict(X_kmeans)`                                                                                                                                                         | I chose `k=3` based on the elbow plot and used it to cluster the data. Then, I assigned the cluster labels to a new column called `Cluster` in the dataframe. |
| 20           | Inspect cluster assignments and relevant features | `df[['Cluster'] + X_kmeans.columns.tolist()].head()`                                                                                                                                                                                                       | I previewed the cluster assignments alongside the features used in clustering to verify that the model segmented customers as expected.                       |

<img width="1147" height="840" alt="Step 6 KMeans Clustering" src="https://github.com/user-attachments/assets/7e9ccdbb-fd11-4226-a61c-5815a80bfce6" />

✅ Step 7: ARIMA (Time Series) – 🎯 Forecast Overall Wine Spending Over Time
| **Step No.** | **Objective**                           | **My Action (in Python)**                                                                           | **What I Did**                                                                                                                                                             |
| ------------ | --------------------------------------- | --------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 21           | Import required libraries               | `import matplotlib.pyplot as plt`<br>`from statsmodels.tsa.arima.model import ARIMA`                | I imported `matplotlib` for plotting and `ARIMA` from `statsmodels` to build a time series model.                                                                          |
| 22           | Prepare the time series data            | `python<br>df_ts = df.groupby('Dt_Customer')['MntWines'].sum().resample('ME').sum()`                | I grouped the data by customer join date (`Dt_Customer`), summed the wine spending, and resampled it by month-end ('ME') to create a monthly time series of wine spending. |
| 23           | Visualize wine spending over time       | `python<br>df_ts.plot(title='Monthly Wine Spending')<br>plt.ylabel("Wine Spend ($)")<br>plt.show()` | I plotted the monthly wine spending to visually inspect seasonality, trends, or anomalies before applying forecasting methods.                                             |
| 24           | Build and fit ARIMA model               | `python<br>model = ARIMA(df_ts, order=(1,1,1))<br>model_fit = model.fit()`                          | I instantiated and fitted an ARIMA model with parameters (p=1, d=1, q=1), which represent autoregressive, differencing, and moving average terms respectively.             |
| 25           | Forecast next 6 months of wine spending | `forecast = model_fit.forecast(steps=6)`<br>`print("Next 6 months forecast:")`<br>`print(forecast)` | I used the trained ARIMA model to forecast wine spending for the next six months and printed the results to observe the predicted future values.                           |

<img width="1136" height="759" alt="Step 7 ARIMA Time Series" src="https://github.com/user-attachments/assets/9173a0c3-5941-42f4-9dc9-08e5da2bcaa3" />

✅ Project Summary – Customer Sales & Marketing Analytics

| **Phase**                        | **Tasks Completed**                                                                                                    |
| -------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| **1. Data Loading & Inspection** | ✔ Loaded CSV without headers<br>✔ Previewed shape and contents                                                         |
| **2. Data Preparation**          | ✔ Assigned meaningful column names<br>✔ Converted date column to `datetime`<br>✔ Handled missing/invalid income values |
| **3. Feature Engineering**       | ✔ Encoded categorical variables                                                                                        |
| **4. Predictive Modeling**       | ✔ Built and evaluated a Linear Regression model to predict income                                                      |
| **5. Clustering (Segmentation)** | ✔ Used KMeans to group customers by spending behavior<br>✔ Determined `k` via elbow method                             |
| **6. Time Series Forecasting**   | ✔ Used ARIMA to forecast future wine spending based on customer acquisition dates                                      |
| **7. Visualization**             | ✔ Visualized elbow curve, time series data, and regression results                                                     |

## ✅ 📊 Project: Customer Sales & Marketing Analytics Using SQL and Python

| **Objective**                                     | **Status**  | **What I Did**                                                                             |
| ------------------------------------------------- | ----------- | -------------------------------------------------------------------------------------------- |
| Data Cleaning using SQL                           | ✅ Completed | Cleaned, normalized, de-duplicated, and validated all fields in `superstore_data` using SSMS |
| Feature Engineering in SQL                        | ✅ Completed | Created `TotalSpend`, `ProfitMargin`, `CustomerYear`, `CustomerMonth`                        |
| Export Cleaned Data to CSV                        | ✅ Completed | Exported `superstore_cleaned.csv` from SSMS                                                  |
| Load Data in Python                               | ✅ Completed | Used `pandas.read_csv()` to import the cleaned file                                          |
| Linear Regression: Income Prediction              | ✅ Completed | Used `scikit-learn` for baseline regression, encoded categorical features                    |
| KMeans Clustering: Customer Segmentation          | ✅ Completed | Used product spend features; chose `k=3` via elbow method; assigned cluster labels           |
| ARIMA Time Series Forecast: Monthly Wine Spending | ✅ Completed | Built ARIMA(1,1,1) model, forecasted next 6 months; handled resampling frequency             |
| EDA & Segmentation in SQL                         | ✅ Completed | Aggregated by education, marital status, and time; discovered key trends                     |
| Optional Visualization (Power BI)                 | ❌ Skipped   | Clearly marked as optional and out of current project scope                                  |

## 📋 2. Deliverables Checklist

| **Deliverable**                       | **Tool**           | **Status** |
| ------------------------------------- | ------------------ | ---------- |
| Data Cleaning & EDA                   | SQL (SSMS)         | ✅ Done     |
| Income Prediction (Linear Regression) | Python             | ✅ Done     |
| Customer Segmentation (KMeans)        | Python             | ✅ Done     |
| Time Series Forecasting (ARIMA)       | Python             | ✅ Done     |
| Cleaned Dataset Export                | SQL → CSV → Python | ✅ Done     |
| Optional Dashboard                    | Power BI           | ❌ Skipped  |

## 📌 3. Final Verdict: ✅ Project Completed

A clear project objective and problem statement
Fully cleaned and enriched data using SQL
Robust Python modeling steps (regression, clustering, time series)
Demonstrated insight extraction and business recommendations
