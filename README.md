Online Retail Data Mining & Analytics Project
Overview
This repository explores an Online Retail dataset containing transactional records from an e-commerce business. The primary goals are to clean and preprocess the raw data, conduct Exploratory Data Analysis (EDA), build RFM metrics, cluster customers, recommend products, predict average order values, and forecast future sales.

Through these steps, we aim to derive data-driven insights into customer behavior, marketing opportunities, and sales trends. The project leverages multiple data science techniques—including outlier detection, feature engineering, machine learning, and time-series forecasting (Prophet).

Project Pipeline
Data Import & Cleaning

Load the Online Retail dataset (Excel, *.xlsx) into a Pandas DataFrame.
Handle missing CustomerID and Description, remove duplicates, and detect outliers.
Compute TotalPrice = Quantity * UnitPrice.
Exploratory Data Analysis (EDA)

Summarize by Country, CustomerID, InvoiceNo to understand spending patterns.
Identify and separate returns (negative quantity or price) to calculate ReturnRate, ReturnCount, etc.
Visualize distributions (histograms, boxplots) for improved understanding of skew and potential anomalies.
RFM Metrics & Additional Features

Recency: Days since last purchase.
Frequency: Unique purchase invoices.
Monetary: Total spending per customer.
Compute AvgOrderValue, approximate CLV with a discount factor, and calculate a LoyaltyScore.
Customer Segmentation

KMeans clustering on PCA-reduced features (e.g., Recency, Frequency, ReturnRate, CLV, AvgOrderValue).
Manually handle high-CLV outliers in a separate cluster.
Use PCA scatterplots to visualize clusters. Summarize cluster-level insights.
Recommendation Systems

Collaborative Filtering (user-user similarity) using cosine_similarity on a user-item purchase matrix.
Content-Based approach using product metadata (Price, Popularity, Brand, Category).
Generate suggestions to either a customer (top products not yet purchased) or a product (similar items).
Predictive Modeling

Build a GradientBoostingRegressor to predict AvgOrderValue from features like Recency, Frequency, ReturnRate, and CLV.
Hyperparameter tuning with GridSearchCV or RandomizedSearchCV.
Compare baseline models (RandomForest, GradientBoosting, XGB, LGB, CatBoost) for better performance.
Time-Series Forecasting (Prophet)

Convert invoices to daily aggregates of TotalPrice.
Use Prophet to forecast future sales; evaluate on the last 6 months of data for RMSE.
Generate a 6-month forecast for business planning. Incorporate holiday or event regressors as needed.
Insights & Strategy

Identify premium or loyal customers, tailor VIP campaigns.
Segment high-return-rate users to improve product descriptions, reduce friction.
Reveal seasonal or monthly patterns in sales to align marketing or inventory decisions.
Key Files
OnlineRetail.xlsx
Primary dataset containing transactional data.
purchase_data.csv
Derived from non-return transactions for pivot-based recommendation.
product_metadata.csv
Contains enriched product details (Price, Popularity, Category, Brand).
RFM & Segmentation Scripts
Generate RFM metrics, cluster customers, and store segmented results.
Recommendation Scripts
Implements collaborative filtering and content-based product similarity.
Predictive Modeling & Forecasting
Gradient Boosting for AvgOrderValue, Prophet for daily sales forecasting.
How to Run
Install Requirements

Python 3.7+
Typical data libraries: pandas, numpy, matplotlib, seaborn
Additional: scikit-learn, statsmodels, prophet, etc.
(Optional) If you want XGBoost, LightGBM, or CatBoost, install them separately.
Place the Dataset

Ensure OnlineRetail.xlsx is in the correct file path (adjust as needed within the code).
Execute the Scripts

Run the main script or Jupyter notebook from the top, which:
Loads and cleans data.
Conducts EDA and outlier removal.
Builds RFM, segments customers.
Generates a recommendation system and predictive models.
Forecasts future sales with Prophet.
Review Outputs

Check plots (boxplots, distributions, PCA scatterplots) for data understanding.
Explore recommendations: see which products are suggested for sample customers.
Inspect forecast charts for sales trends and seasonality.
Future Enhancements
Advanced Outlier Detection

Explore domain thresholds or advanced algorithms (IsolationForest, DBSCAN) for more nuanced outlier handling.
Time-Based Splits

Use rolling or expanding windows in the predictive tasks to mirror real-life deployment.
More Sophisticated Clustering

Consider non-spherical methods (DBSCAN, HDBSCAN), or hierarchical clustering for deeper segmentation.
Hybrid Recommendations

Combine collaborative filtering & content-based into a single hybrid approach or use specialized libraries like LightFM.
Additional Prophet Regressors

Incorporate holiday or event calendars if your business has known peaks (e.g., Black Friday, Christmas, or region-specific festivals).
Conclusion
This project demonstrates a complete data mining approach on an Online Retail dataset—covering everything from cleaning and RFM segmentation to predictive modeling and Prophet forecasting. By integrating these methods, businesses can:

Improve marketing decisions with clear customer segmentation.
Boost revenue by serving relevant product recommendations.
Plan inventory and budget using short- to mid-range sales forecasts.
If you have any questions or suggestions, feel free to open an issue or submit a pull request. Happy analyzing!


