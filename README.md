Machine Learning & Data Mining Project

Project Overview

This project analyzes the Online Retail Dataset using Python to perform:

Data preprocessing: Handling missing values, outliers, and duplicates.

Exploratory Data Analysis (EDA): Generating insights about customer and product behavior.

Advanced analytics: RFM (Recency, Frequency, Monetary) metrics calculation and customer segmentation.

Statistical and visual insights: Distribution of sales, returns, and trends.

Advanced systems: Product recommendation system and time-series sales forecasting.

Dataset

The dataset, "OnlineRetail.xlsx," contains:

InvoiceNo: Transaction identifier.

StockCode: Product code.

Description: Product description.

Quantity: Quantity of product sold.

InvoiceDate: Date of transaction.

UnitPrice: Price per product unit.

CustomerID: Unique identifier for customers.

Country: Country of the transaction.

Key Features

1. Data Preprocessing

Handling Missing Values: Missing CustomerID rows were removed. Missing Description values were filled using the mode of StockCode.

Outlier Removal: Implemented the Interquartile Range (IQR) method for numerical columns.

Duplicate Removal: Identified and removed duplicate rows.

2. Exploratory Data Analysis (EDA)

Generated descriptive statistics for numerical and categorical columns.

Analyzed correlations to identify multicollinearity issues.

Visualized distributions of key metrics.

3. Advanced Analytics

RFM Metrics:

Recency: Days since the last purchase.

Frequency: Number of unique transactions.

Monetary: Total spending.

Returns Analysis:

Return rate, return count, and total return amount per customer.

Log-transformed skewed variables for better interpretability.

4. Insights and Visualizations

Boxplots, heatmaps, and histograms to uncover data trends.

Time-series analysis of sales trends.

5. Product Recommendation System

Implemented a collaborative filtering-based recommendation system.

Leveraged customer-product interaction data to suggest products based on similarity.

Generated recommendations for selected customers and analyzed their purchase behavior.

6. Time-Series Forecasting

Utilized Prophet for forecasting daily total sales.

Trained the model on historical sales data and predicted the next six months' trends.

Visualized forecast results and calculated Root Mean Squared Error (RMSE) for the test set.

Installation and Requirements

Clone this repository:

git clone https://github.com/Taylanozveren/MachineLearning-DataMiningProject.git

Install required libraries:

pip install -r requirements.txt

Usage

Place the dataset (OnlineRetail.xlsx) in the root directory.

Run the script:

python main.py

Outputs include:

Cleaned dataset.

CSV files for missing CustomerID rows and duplicates.

Visualizations, RFM analysis, product recommendations, and time-series forecasts.

Results

Key Findings

Customer Behavior: Identified high-value customer segments using RFM metrics.

Product Insights: Popular and underperforming products.

Returns: Key patterns in product returns and strategies to reduce return rates.

Recommendations: Suggested personalized product recommendations for customers.

Forecasts: Predicted future sales trends and seasonal behaviors.

Strategic Recommendations

Focus marketing efforts on high RFM-score customers.

Address high-return products with better descriptions and policies.

Utilize sales forecasts to optimize inventory and plan marketing campaigns.

Leverage product recommendations to boost cross-selling and up-selling opportunities.

Future Work

Implement advanced clustering algorithms for deeper customer segmentation.

Enhance the recommendation system with hybrid models.

Incorporate predictive models for Customer Lifetime Value (CLV).

Explore automated anomaly detection for sales trends.

Contributors

Taylan Özveren
