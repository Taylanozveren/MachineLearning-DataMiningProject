    ###############################################################################
    # STEP 1: IMPORT LIBRARIES
    ###############################################################################
    # 1.1) Import the required libraries for data processing and visualization.
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    # NumPy and Pandas for data processing,
    # Seaborn and Matplotlib for data visualization.


    ###############################################################################
    # STEP 2: LOAD THE DATA
    ###############################################################################
    # 2.1) Load the dataset from the specified file path.
    #      Adjust the file path as necessary based on your system.
    df = pd.read_excel("C:\\Users\\Win10\\Desktop\\ACM476 Data Mining Course\\OnlineRetail.xlsx")

    ###############################################################################
    # STEP 3: INITIAL DATA INSPECTION
    ###############################################################################
    # 3.1) Display the first few rows of the dataset for an overview.
    display(df.head())

    # 3.2) Check the shape (rows, columns) of the dataset and inspect its structure.
    print("Data Shape:", df.shape)
    df.info()
    print("Column Names:", df.columns)

    # 3.3) Check for missing values in each column.
    print("Missing Values:\n", df.isnull().sum())

    # 3.4) Display descriptive statistics for all columns (both numeric and object types).
    print("Descriptive Statistics:\n", df.describe(include='all'))

    ###############################################################################
    # STEP 4: HANDLE MISSING CUSTOMERID
    ###############################################################################
    # 4.1) The CustomerID column is critical for customer-level analyses.
    #      Remove rows where CustomerID is missing.
    missing_customer_id_df = df[df['CustomerID'].isna()].copy()

    if not missing_customer_id_df.empty:
        # Save rows with missing CustomerID into a separate file.
        missing_customer_id_df.to_csv("missing_customer_id_rows.csv", index=False)
        print(f"Saved {len(missing_customer_id_df)} rows with missing CustomerID.")
    else:
        print("No rows with missing CustomerID found.")

    # 4.2) Drop rows with missing CustomerID.
    df = df.dropna(subset=['CustomerID'])
    print("After removing missing CustomerID, shape:", df.shape)

    ###############################################################################
    # STEP 5: FILL MISSING DESCRIPTIONS
    ###############################################################################
    # 5.1) For missing 'Description' values, fill them based on the mode
    #      (most frequent value) within each 'StockCode' group.
    df['Description'] = df.groupby('StockCode')['Description'].transform(
        lambda x: x.fillna(x.mode()[0] if not x.mode().empty else '')
    )

    ###############################################################################
    # STEP 6: DEAL WITH DUPLICATES
    ###############################################################################
    # 6.1) Check the number of duplicate rows.
    duplicates = df[df.duplicated()]
    print("Number of duplicate rows:", duplicates.shape[0])

    if not duplicates.empty:
        # Optionally save duplicates to a separate file before removing.
        duplicates.to_csv("duplicates.csv", index=False)

    # 6.2) Drop the duplicate rows.
    df = df.drop_duplicates()
    print("After removing duplicates, shape:", df.shape)

    ###############################################################################
    # STEP 7: CORRELATION & MULTICOLLINEARITY CHECK
    ###############################################################################
    # 7.1) Compute correlation among numeric columns to analyze multicollinearity.
    numeric_df = df.select_dtypes(include=[np.number])
    cor_matrix = numeric_df.corr().abs()

    plt.figure(figsize=(12, 10))
    sns.heatmap(cor_matrix, cmap='RdBu', annot=True)
    plt.title("Correlation Matrix of Numeric Features")
    plt.show()

    # 7.2) Identify columns with correlation greater than 0.90.
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
    high_corr = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > 0.90)]
    print("Highly correlated columns (>0.90):", high_corr)


    ###############################################################################
    # STEP 8: OUTLIER REMOVAL (IQR METHOD)
    ###############################################################################
    def remove_outliers(data, column, multiplier=3.0):
        """
        Removes outliers based on the IQR (Interquartile Range) method.
        The 'multiplier' controls how aggressively outliers are removed.
        A higher multiplier => fewer outliers removed.
        """
        q1 = data[column].quantile(0.25)
        q3 = data[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr
        return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]


    # 8.1) Apply the outlier removal process to each numeric column.
    numeric_columns = numeric_df.columns
    for col in numeric_columns:
        df = remove_outliers(df, col, multiplier=3.0)

    # 8.2) Verify the distribution with boxplots after outlier removal.
    columns_to_plot = numeric_columns[:3]  # Selecting the first three numeric columns
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))  # Layout: 1 row, 3 columns

    for ax, col in zip(axes, columns_to_plot):
        sns.boxplot(x=df[col], ax=ax)
        ax.set_title(f"Boxplot of {col} After Outlier Removal")

    plt.tight_layout()
    plt.show()

    print("Shape after outlier removal:", df.shape)
    print("Statistics after outlier removal:\n", df.describe())


    ###############################################################################
    # STEP 9: EXPLORATORY STATISTICS BASED ON TARGET VARIABLES
    ###############################################################################
    def target_summary_with_num(dataframe, target, num_col):
        """
        Computes the mean of 'num_col' grouped by 'target'.
        Useful for comparing numeric features across different categories.
        """
        summary = dataframe.groupby(target)[num_col].mean()
        print(f"\nMean {num_col} by {target}:\n{summary}\n")
        return summary


    def calculate_total_price(dataframe):
        """
        Ensures 'TotalPrice' = Quantity * UnitPrice.
        Prints how many transactions are above or below the average spending.
        Helps to understand distribution and skewness.
        """
        dataframe['TotalPrice'] = dataframe['Quantity'] * dataframe['UnitPrice']
        mean_total_price = dataframe['TotalPrice'].mean()
        above_mean_count = (dataframe['TotalPrice'] > mean_total_price).sum()
        below_mean_count = (dataframe['TotalPrice'] < mean_total_price).sum()
        print("Transactions above average:", above_mean_count)
        print("Transactions below average:", below_mean_count)
        return dataframe


    # 9.1) Calculate and add 'TotalPrice' to the dataframe.
    df = calculate_total_price(df)

    # 9.2) Analyze metrics grouped by different categories.
    target_summary_with_num(df, 'Country', 'Quantity')
    target_summary_with_num(df, 'Country', 'TotalPrice')
    target_summary_with_num(df, 'CustomerID', 'TotalPrice')
    target_summary_with_num(df, 'InvoiceNo', 'TotalPrice')
    target_summary_with_num(df, 'StockCode', 'Quantity')

    # 9.3) Add a 'Year' column for yearly trend analysis.
    df['Year'] = df['InvoiceDate'].dt.year
    target_summary_with_num(df, 'Year', 'TotalPrice')

    ###############################################################################
    # STEP 10: CHECK FOR NEGATIVE VALUES (RETURNS) AND CREATE RETURN METRICS
    ###############################################################################
    # 10.1) Ensure 'TotalPrice' exists and analyze negative values if present.
    if 'TotalPrice' not in df.columns:
        df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    else:
        negative_total_price_count = (df['TotalPrice'] < 0).sum()
        negative_quantity_count = (df['Quantity'] < 0).sum()
        print("Negative TotalPrice count:", negative_total_price_count)
        print("Negative Quantity count:", negative_quantity_count)

        # Separate return transactions (negative Quantity or TotalPrice) from non-returns.
        returns_df = df[(df['Quantity'] < 0) | (df['TotalPrice'] < 0)]
        non_returns_df = df[(df['Quantity'] >= 0) & (df['TotalPrice'] >= 0)]

        # Recalculate total and returned quantities per customer.
        total_quantity = non_returns_df.groupby('CustomerID')['Quantity'].sum()
        returned_quantity = returns_df.groupby('CustomerID')['Quantity'].sum().abs()

        # Return rate per customer (percentage of items returned).
        return_rate = (returned_quantity / total_quantity).fillna(0)
        return_count = returns_df.groupby('CustomerID').size()
        total_return_amount = returns_df.groupby('CustomerID')['TotalPrice'].sum().abs()

        # Map these return-related metrics back to the main dataframe.
        df['ReturnRate'] = df['CustomerID'].map(return_rate)
        df['ReturnCount'] = df['CustomerID'].map(return_count).fillna(0)
        df['TotalReturnAmount'] = df['CustomerID'].map(total_return_amount).fillna(0)

        # Apply log transformation to reduce skewness for selected columns.
        df['LogTotalPrice'] = np.log1p(df['TotalPrice'].clip(lower=0))
        df['LogQuantity'] = np.log1p(df['Quantity'].clip(lower=0))

        # Check skewness after log transform.
        log_total_price_skewness = df['LogTotalPrice'].skew()
        log_quantity_skewness = df['LogQuantity'].skew()
        print("LogTotalPrice Skewness:", log_total_price_skewness)
        print("LogQuantity Skewness:", log_quantity_skewness)

        # Visualize the distributions of the transformed features.
        plt.figure(figsize=(8, 5))
        sns.histplot(df['LogTotalPrice'].dropna(), bins=30, kde=True)
        plt.title('LogTotalPrice Distribution')
        plt.xlabel('LogTotalPrice')
        plt.ylabel('Frequency')
        plt.show()

        plt.figure(figsize=(8, 5))
        sns.histplot(df['ReturnRate'].dropna(), bins=30, kde=True)
        plt.title('ReturnRate Distribution')
        plt.xlabel('ReturnRate')
        plt.ylabel('Frequency')
        plt.show()

    # 10.2) Identify customers with ReturnRate > 1 (possible data issues or anomalies).
    high_return_customers = df[df['ReturnRate'] > 1]
    display(high_return_customers[['CustomerID', 'ReturnRate', 'ReturnCount', 'TotalReturnAmount']])
    high_return_customers_count = high_return_customers.shape[0]
    print("Number of customers with ReturnRate > 1:", high_return_customers_count)

    ###############################################################################
    # STEP 11: FINAL INSPECTIONS AFTER EDA
    ###############################################################################
    # 11.1) Final shape, info, and columns after EDA steps.
    print("Final Data Shape:", df.shape)
    df.info()
    print("Columns:", df.columns)

    # 11.2) Create a 'year-month' period column for time-series analyses.
    df['InvoiceYearMonth'] = df['InvoiceDate'].dt.to_period('M')

    # 11.3) Compute monthly total sales to visualize sales trends over time.
    monthly_sales = df.groupby('InvoiceYearMonth')['TotalPrice'].sum()

    plt.figure(figsize=(12, 6))
    monthly_sales.plot(kind='line', marker='o')
    plt.title('Monthly Sales Trend')
    plt.xlabel('Year-Month')
    plt.ylabel('Total Sales')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.show()

    # 11.4) Check the most recent transaction date in the dataset.
    current_date = df['InvoiceDate'].max()
    print("Most recent transaction date:", current_date)

    ###############################################################################
    # STEP 12: RFM METRICS CALCULATION
    ###############################################################################
    # 12.1) Calculate the RFM metrics:
    #       - Recency: Number of days since the last purchase
    #       - Frequency: Number of unique transactions
    #       - Monetary: Total spending
    rfm = df.groupby('CustomerID').agg(
        Recency=('InvoiceDate', lambda x: (current_date - x.max()).days),
        Frequency=('InvoiceNo', 'nunique'),
        Monetary=('TotalPrice', 'sum')
    )

    display(rfm.head())
    print("RFM Summary Statistics:\n", rfm.describe())

    ###############################################################################
    # STEP 13: DISTRIBUTION OF RFM METRICS
    ###############################################################################
    # 13.1) Visualize the distribution of RFM metrics (Recency, Frequency, Monetary).
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))  # 1 row, 3 plots

    # Recency Distribution
    sns.histplot(rfm['Recency'], kde=True, bins=20, ax=axes[0])
    axes[0].set_title('Recency Distribution')
    axes[0].set_xlabel('Recency (Days)')
    axes[0].set_ylabel('Frequency')

    # Frequency Distribution
    sns.histplot(rfm['Frequency'], kde=True, bins=20, ax=axes[1])
    axes[1].set_title('Frequency Distribution')
    axes[1].set_xlabel('Frequency (Number of unique transactions)')
    axes[1].set_ylabel('Frequency')

    # Monetary Distribution
    sns.histplot(rfm['Monetary'], kde=True, bins=20, ax=axes[2])
    axes[2].set_title('Monetary Distribution')
    axes[2].set_xlabel('Monetary (Total spending)')
    axes[2].set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()

    ###############################################################################
    # STEP 14: ADDITIONAL METRICS
    ###############################################################################
    # 14.1) Calculate the Average Order Value (AvgOrderValue = Monetary / Frequency).
    rfm['AvgOrderValue'] = rfm['Monetary'] / rfm['Frequency']

    # 14.2) Encode the 'Country' column for modeling purposes.
    df['Country'] = pd.factorize(df['Country'])[0]
    print("After encoding the Country column, the dataframe columns are:\n", df.columns)
    df.shape

    ###############################################################################
    # 0. IMPORTS & SETUP
    ###############################################################################
    import pandas as pd
    import numpy as np
    import os
    import seaborn as sns
    import matplotlib.pyplot as plt
    import warnings

    # Sklearn & Statsmodels
    from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.model_selection import (
        train_test_split,
        GridSearchCV,
        RandomizedSearchCV,
        cross_validate
    )
    from sklearn.ensemble import (
        GradientBoostingRegressor,
        RandomForestRegressor
    )
    from sklearn.metrics import (
        mean_squared_error,
        silhouette_score,
        calinski_harabasz_score
    )
    from sklearn.metrics.pairwise import cosine_similarity

    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    # Prophet (Time Series)
    from prophet import Prophet
    import math

    warnings.filterwarnings("ignore")

    # (Optional) Checking the installation of XGBoost, LightGBM, CatBoost
    try:
        from xgboost import XGBRegressor

        xgb_installed = True
    except ImportError:
        xgb_installed = False

    try:
        from lightgbm import LGBMRegressor

        lgb_installed = True
    except ImportError:
        lgb_installed = False

    try:
        from catboost import CatBoostRegressor

        cat_installed = True
    except ImportError:
        cat_installed = False

    ###############################################################################
    # 1. LOAD & CLEAN DATA
    ###############################################################################
    """
    Here we:
    - Convert InvoiceDate in the DataFrame to datetime.
    - Drop invalid rows for InvoiceDate.
    - Drop duplicates.
    - Create the TotalPrice column (Quantity * UnitPrice).
    """
    print("Initial DataFrame Columns:\n", df.columns)
    print("Preview of the DataFrame:\n", df.head())
    df.shape

    if not pd.api.types.is_datetime64_any_dtype(df['InvoiceDate']):
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')

    initial_rows = len(df)
    df.dropna(subset=['InvoiceDate'], inplace=True)
    print(f"Dropped {initial_rows - len(df)} rows with invalid InvoiceDate.")

    before_duplicates = len(df)
    df.drop_duplicates(inplace=True)
    print(f"Dropped {before_duplicates - len(df)} duplicate rows.")

    print("Data after cleaning:\n", df.head())

    if 'TotalPrice' not in df.columns:
        df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
        print("Calculated TotalPrice.\n", df[['Quantity', 'UnitPrice', 'TotalPrice']].head())

    ###############################################################################
    # 2. SEPARATE RETURNS & POSITIVE SALES
    ###############################################################################
    """
    - Negative rows (returns) => returns_df
    - Positive rows (normal sales) => non_returns_df
    - ReturnRate, ReturnCount, etc.
    - purchase_data.csv (CustomerID, ProductID, PurchaseAmount) for pivot-based systems
    """
    returns_df = df[(df['Quantity'] < 0) or (df['TotalPrice'] < 0)]
    non_returns_df = df[(df['Quantity'] >= 0) & (df['TotalPrice'] >= 0)]

    total_quantity = non_returns_df.groupby('CustomerID')['Quantity'].sum()
    returned_quantity = returns_df.groupby('CustomerID')['Quantity'].sum().abs()

    return_rate = (returned_quantity / total_quantity).fillna(0)
    return_count = returns_df.groupby('CustomerID').size()

    purchase_data = non_returns_df.groupby(['CustomerID', 'StockCode'], as_index=False).agg({'TotalPrice': 'sum'})
    purchase_data.rename(columns={'StockCode': 'ProductID', 'TotalPrice': 'PurchaseAmount'}, inplace=True)
    purchase_data.to_csv("purchase_data.csv", index=False)
    print("Created purchase_data.csv from non_returns_df successfully!")

    ###############################################################################
    # 3. RFM METRICS & EXTRA FEATURES
    ###############################################################################
    """
    - RFM (Recency, Frequency, Monetary) from normal sales
    - ReturnRate, ReturnCount, TotalReturnAmount
    - AvgOrderValue, CLV (3 years, discount=0.1), LoyaltyScore = Frequency/(1+ReturnRate)
    """
    current_date = df['InvoiceDate'].max()
    print("Max Invoice Date in dataset:", current_date)

    rfm = non_returns_df.groupby('CustomerID').agg(
        Recency=('InvoiceDate', lambda x: (current_date - x.max()).days),
        Frequency=('InvoiceNo', 'nunique'),
        Monetary=('TotalPrice', 'sum')
    ).reset_index()

    returned_amount = returns_df.groupby('CustomerID')['TotalPrice'].sum().abs()
    rfm['ReturnRate'] = rfm['CustomerID'].map(return_rate).fillna(0)
    rfm['ReturnCount'] = rfm['CustomerID'].map(return_count).fillna(0)
    rfm['TotalReturnAmount'] = rfm['CustomerID'].map(returned_amount).fillna(0)

    rfm['AvgOrderValue'] = np.where(rfm['Frequency'] == 0, 0, rfm['Monetary'] / rfm['Frequency'])

    discount_rate = 0.10
    years = 3
    discount_factors = sum([1 / ((1 + discount_rate) ** t) for t in range(1, years + 1)])
    rfm['CLV'] = rfm['AvgOrderValue'] * rfm['Frequency'] * discount_factors

    rfm['LoyaltyScore'] = rfm['Frequency'] * (1 / (1 + rfm['ReturnRate']))
    print("RFM after aggregation & extra features:\n", rfm.head())

    ###############################################################################
    # 4. TIME FEATURES (MONTH, QUARTER, SEASON)
    ###############################################################################
    """
    - Based on last purchase date: ShoppingMonth, Quarter, Season
    - OneHotEncoder
    """
    last_invoice = non_returns_df.groupby('CustomerID')['InvoiceDate'].max()
    rfm['ShoppingTime'] = rfm['CustomerID'].map(last_invoice)
    rfm['ShoppingMonth'] = rfm['ShoppingTime'].dt.month.astype(str)
    rfm['ShoppingQuarter'] = rfm['ShoppingTime'].dt.quarter.astype(str)

    season_map = {'1': 'Winter', '2': 'Spring', '3': 'Summer', '4': 'Autumn'}
    rfm['Season'] = rfm['ShoppingQuarter'].map(season_map)

    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    time_features = rfm[['ShoppingMonth', 'ShoppingQuarter', 'Season']]
    encoded_time = encoder.fit_transform(time_features)
    time_columns = encoder.get_feature_names_out(['ShoppingMonth', 'ShoppingQuarter', 'Season'])
    time_df = pd.DataFrame(encoded_time, columns=time_columns, index=rfm.index)

    rfm = pd.concat([rfm, time_df], axis=1)
    print("RFM with time-based one-hot encoding:\n", rfm.head(15))

    ###############################################################################
    # 5. SCALING FOR CLUSTERING (REMOVE LOYALTYSCORE, MANUAL OUTLIERS)
    ###############################################################################
    """
    Here we:
    1. Remove Monetary and LoyaltyScore due to multicollinearity concerns.
    2. Mark customers with CLV > 50k as a manual outlier (cluster 9).
    """

    # For clustering, define numeric and categorical features
    numeric_features = ['Recency', 'Frequency', 'ReturnRate', 'CLV', 'AvgOrderValue']
    categorical_features = time_df.columns.tolist()

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score, calinski_harabasz_score
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    import statsmodels.api as sm

    ###############################################################################
    # 1. RFM METRICS CALCULATION (ASSUMES `rfm` IS PREDEFINED)
    ###############################################################################
    # For clustering, define numeric and categorical features again (example usage):
    numeric_features = ['Recency', 'Frequency', 'ReturnRate', 'CLV', 'AvgOrderValue']
    categorical_features = ["Season_Spring", "Season_Summer", "Season_Winter"]


    ###############################################################################
    # 2. MULTICOLLINEARITY CHECK
    ###############################################################################
    def check_multicollinearity(df, threshold=5.0):
        """
        Calculates VIF (Variance Inflation Factor) to check multicollinearity.
        Features with VIF > threshold may cause issues.
        """
        numeric_df = df.select_dtypes(include=[np.number]).dropna(axis=1)
        X = sm.add_constant(numeric_df)
        vif_data = []
        for i in range(X.shape[1]):
            vif_val = variance_inflation_factor(X.values, i)
            vif_data.append((X.columns[i], vif_val))
        vif_df = pd.DataFrame(vif_data, columns=['Feature', 'VIF'])
        high_vif = vif_df[vif_df['VIF'] > threshold]
        print("\n--- VIF Results ---\n", vif_df)
        if not high_vif.empty:
            print(f"\nFeatures with VIF > {threshold}:\n", high_vif)
        return vif_df


    # VIF Analysis
    print("\n=== Checking VIF without LoyaltyScore & Monetary ===")
    vif_result = check_multicollinearity(rfm[numeric_features], threshold=5.0)

    ###############################################################################
    # 3. SCALING NUMERIC FEATURES
    ###############################################################################
    scaler = StandardScaler()
    scaled_numeric = scaler.fit_transform(rfm[numeric_features])
    scaled_df = pd.DataFrame(scaled_numeric, columns=numeric_features, index=rfm.index)

    # Combine numeric and categorical features
    cluster_features = pd.concat([scaled_df, rfm[categorical_features]], axis=1)
    print("Cluster Features shape:", cluster_features.shape)
    print("Cluster Features preview:\n", cluster_features.head())

    ###############################################################################
    # 4. DETERMINE OPTIMAL NUMBER OF CLUSTERS (ELBOW, SILHOUETTE, CALINSKI)
    ###############################################################################
    rfm_pca_for_det = PCA(n_components=2).fit_transform(cluster_features)
    silhouette_scores = []
    calinski_scores = []
    wcss = []
    K_range = range(2, 10)


    def calculate_clustering_metrics():
        """
        Computes clustering metrics for different k values:
        - Silhouette Score
        - Calinski-Harabasz Score
        - WCSS (Elbow Method)
        """
        for k in K_range:
            kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init='auto')
            labels_temp = kmeans_temp.fit_predict(cluster_features)
            silhouette_scores.append(silhouette_score(cluster_features, labels_temp))
            calinski_scores.append(calinski_harabasz_score(cluster_features, labels_temp))
            wcss.append(kmeans_temp.inertia_)

        plt.figure(figsize=(15, 4))

        plt.subplot(1, 3, 1)
        plt.plot(K_range, silhouette_scores, marker='o')
        plt.title("Silhouette Score")

        plt.subplot(1, 3, 2)
        plt.plot(K_range, calinski_scores, marker='o')
        plt.title("Calinski-Harabasz Score")

        plt.subplot(1, 3, 3)
        plt.plot(K_range, wcss, marker='o')
        plt.title("Elbow (WCSS)")

        plt.tight_layout()
        plt.show()


    calculate_clustering_metrics()

    ###############################################################################
    # 5. MANUAL OUTLIER REMOVAL (CLV > 50k => cluster=9)
    ###############################################################################
    CLV_THRESHOLD = 50000
    outlier_customers = rfm[rfm['CLV'] > CLV_THRESHOLD]['CustomerID'].unique()
    print(f"\nNumber of High CLV Outlier Customers (> {CLV_THRESHOLD}): {len(outlier_customers)}")

    # Separate normal customers from potential wholesale/outliers
    rfm['ManualCluster'] = -1
    normal_mask = rfm['CLV'] <= CLV_THRESHOLD
    rfm_normal = rfm[normal_mask].copy()
    cluster_features_normal = cluster_features.loc[normal_mask].copy()

    # Apply KMeans (k=5) for normal customers
    pca_normal = PCA(n_components=2)
    rfm_pca_normal = pca_normal.fit_transform(cluster_features_normal)
    kmeans_normal = KMeans(n_clusters=5, random_state=42, n_init='auto')
    rfm_normal['Cluster_temp'] = kmeans_normal.fit_predict(rfm_pca_normal)

    # Assign results
    rfm.loc[normal_mask, 'ManualCluster'] = rfm_normal['Cluster_temp']
    rfm.loc[~normal_mask, 'ManualCluster'] = 9  # Outlier customers (CLV>50k)

    rfm['Cluster'] = rfm['ManualCluster']
    rfm.drop(columns=['ManualCluster'], inplace=True, errors='ignore')
    print("\nAssigned cluster labels (0..4 for normal, 9 for outlier).")

    ###############################################################################
    # 6. CLUSTER SUMMARY
    ###############################################################################
    agg_dict = {
        'Recency': 'mean',
        'Frequency': 'mean',
        'ReturnRate': 'mean',
        'CLV': 'mean',
        'AvgOrderValue': 'mean'
    }
    for col in categorical_features:
        agg_dict[col] = 'sum'

    cluster_summary = rfm.groupby('Cluster').agg(agg_dict).reset_index()
    print("Cluster Summary:\n", cluster_summary)
    rfm.to_csv("rfm_segmented_with_month.csv", index=False)
    print("Segmented data saved to rfm_segmented_with_month.csv")

    ###############################################################################
    # 7. PCA VISUALIZATION (2D Scatter Plot for Clusters)
    ###############################################################################
    pca_all = PCA(n_components=2)
    rfm_pca_all = pca_all.fit_transform(cluster_features)

    # Add PCA results to the DataFrame
    rfm['PCA1'] = rfm_pca_all[:, 0]
    rfm['PCA2'] = rfm_pca_all[:, 1]

    # Visualization
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=rfm,
        x='PCA1',
        y='PCA2',
        hue='Cluster',  # Color by cluster
        palette='Set1',
        s=100,
        alpha=0.7
    )
    plt.title('PCA Visualization of Clusters (2D)')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    ###############################################################################
    # 8. PRODUCT DATA (PRICE & POPULARITY) + RECOMMENDATION
    ###############################################################################
    def extract_category_brand(description):
        """
        Extract a rough category and brand from product descriptions.
        """
        desc = description.upper()
        if 'RETROSPOT' in desc:
            brand = 'Retrospot'
        elif 'PANTRY' in desc:
            brand = 'Pantry'
        elif 'REGENCY' in desc:
            brand = 'Regency'
        elif 'SKULL' in desc:
            brand = 'Skull'
        elif 'GARDEN' in desc:
            brand = 'Garden'
        else:
            brand = 'GenericBrand'

        if 'LUNCH BAG' in desc:
            category = 'Bag'
        elif 'HEART' in desc or 'LOVE' in desc:
            category = 'HeartDecoration'
        elif 'BUNTING' in desc or 'PARTY' in desc or 'CELEBRATION' in desc:
            category = 'PartyDecor'
        elif any(kw in desc for kw in
                 ['JAM MAKING', 'TEA SET', 'CAKE TINS', 'BAKING SET', 'RECIPE BOX', 'BAKE', 'KITCHEN']):
            category = 'Kitchen'
        elif 'PAPER CHAIN' in desc or 'CRAFT' in desc:
            category = 'Party'
        elif any(kw in desc for kw in ['ALARM CLOCK', 'CLOCK', 'TIME']):
            category = 'Home'
        elif any(kw in desc for kw in ['FRAME', 'PICTURE', 'PHOTO']):
            category = 'Decor'
        elif any(kw in desc for kw in ['TOY', 'CHILD', 'KIDS']):
            category = 'Toys'
        elif any(kw in desc for kw in ['GARDEN', 'OUTDOOR']):
            category = 'Garden'
        elif any(kw in desc for kw in ['CHRISTMAS', 'HOLIDAY']):
            category = 'Seasonal'
        else:
            category = 'Other'
        return category, brand


    product_info = df[['StockCode', 'Description', 'UnitPrice', 'Quantity']].copy()
    product_info.rename(columns={'StockCode': 'ProductID'}, inplace=True)
    product_info.dropna(subset=['Description'], inplace=True)

    product_stats = product_info.groupby('ProductID').agg({
        'UnitPrice': 'mean',
        'Quantity': 'sum',
        'Description': 'first'
    }).reset_index()

    product_stats[['Category', 'Brand']] = product_stats['Description'].apply(
        lambda x: pd.Series(extract_category_brand(x))
    )
    product_stats.rename(columns={'UnitPrice': 'Price', 'Quantity': 'Popularity'}, inplace=True)
    product_stats.to_csv("product_metadata.csv", index=False)
    print("Created product_metadata.csv with real Price & Popularity.")

    product_data = pd.read_csv("product_metadata.csv")
    print("Loaded product_metadata.csv. Columns:", product_data.columns)

    if not {'ProductID', 'Price', 'Popularity', 'Category', 'Brand'}.issubset(product_data.columns):
        raise ValueError("product_metadata must have ProductID, Price, Popularity, Category, Brand columns.")

    product_data.drop_duplicates(subset=['ProductID'], inplace=True)
    product_data['Price'] = pd.to_numeric(product_data['Price'], errors='coerce').fillna(product_data['Price'].mean())
    product_data['Popularity'] = pd.to_numeric(product_data['Popularity'], errors='coerce') \
        .fillna(product_data['Popularity'].mean())

    brand_encoder = LabelEncoder()
    category_encoder = LabelEncoder()
    product_data['Brand_enc'] = brand_encoder.fit_transform(product_data['Brand'])
    product_data['Category_enc'] = category_encoder.fit_transform(product_data['Category'])

    feature_cols = ['Price', 'Popularity', 'Brand_enc', 'Category_enc']
    scaler_product = StandardScaler()
    product_data[feature_cols] = scaler_product.fit_transform(product_data[feature_cols])

    product_similarity = cosine_similarity(product_data[feature_cols])
    product_similarity_df = pd.DataFrame(product_similarity,
                                         index=product_data['ProductID'],
                                         columns=product_data['ProductID'])


    ###############################################################################
    # 8.1. RECOMMENDATION SYSTEMS (CF & Content-Based)
    ###############################################################################
    def recommend_products(customer_id, user_item_mat, cust_sim_df, top_n=5):
        """
        Simple Collaborative Filtering:
        1. Find similar customers for the given customer_id.
        2. Weighted average of their purchases as a recommendation.
        """
        if customer_id not in cust_sim_df.index:
            print(f"Customer {customer_id} not in similarity index.")
            return []
        sim_scores = cust_sim_df.loc[customer_id].drop(customer_id, errors='ignore')

        common_customers = sim_scores.index.intersection(user_item_mat.index)
        if len(common_customers) == 0:
            print("No common customers found. Possibly dimension mismatch.")
            return []

        sim_scores_sub = sim_scores.loc[common_customers].sort_index()
        user_item_sub = user_item_mat.loc[common_customers].sort_index()

        weighted_purchases = user_item_sub.multiply(sim_scores_sub, axis='index').sum(axis=0)

        if customer_id not in user_item_mat.index:
            print(f"Customer {customer_id} not in user_item_mat index.")
            return []
        already_purchased = user_item_mat.loc[customer_id] > 0
        candidates = weighted_purchases[~already_purchased]

        return candidates.sort_values(ascending=False).head(top_n).index.tolist()


    def recommend_similar_products(product_id, product_sim_df, top_n=5):
        """
        Content-Based Filtering:
        1. Use product similarity matrix (cosine).
        2. Retrieve top_n similar products.
        """
        if product_id not in product_sim_df.index:
            print(f"Product {product_id} not in similarity index.")
            return []
        similar_scores = product_sim_df.loc[product_id].sort_values(ascending=False)
        return similar_scores.iloc[1:top_n + 1].index.tolist()


    user_item_matrix = purchase_data.pivot_table(index='CustomerID',
                                                 columns='ProductID',
                                                 values='PurchaseAmount',
                                                 fill_value=0)
    customer_similarity = cosine_similarity(user_item_matrix)
    customer_similarity_df = pd.DataFrame(customer_similarity,
                                          index=user_item_matrix.index,
                                          columns=user_item_matrix.index)

    # Example: CF recommendation for a few customers
    if not purchase_data.empty:
        some_customers = user_item_matrix.index[:5]  # for instance, first 5
        for cust_id in some_customers:
            print(f"\nGetting CF recommendations for Customer {cust_id}...")
            recs = recommend_products(cust_id, user_item_matrix, customer_similarity_df, top_n=5)
            print(f"Recommended Products: {recs}")
            recommended_details = product_data[product_data['ProductID'].isin(recs)][
                ['ProductID', 'Description', 'Category', 'Brand']
            ]
            print(recommended_details)

    test_product_id = product_data['ProductID'].iloc[0]
    print(f"\nGetting similar products for Product {test_product_id} (Content-Based)...")
    sim_products = recommend_similar_products(test_product_id, product_similarity_df)
    print(f"Similar Products: {sim_products}")
    similar_details = product_data[product_data['ProductID'].isin(sim_products)][
        ['ProductID', 'Description', 'Category', 'Brand']
    ]
    print("\nSimilar Products Details:")
    print(similar_details)

    ###############################################################################
    # 9. PREDICTIVE MODELING FOR AvgOrderValue
    ###############################################################################
    """
    We will predict AvgOrderValue.
     - Remove 'AvgOrderValue' and 'LoyaltyScore' from X (they are not in the cluster).
     - Remaining: [Recency, Frequency, ReturnRate, CLV]
    """
    print("\nBuilding a predictive model for AvgOrderValue (GradientBoosting)...")
    X = rfm[['Recency', 'Frequency', 'ReturnRate', 'CLV']]
    y = rfm['AvgOrderValue']

    print("Features (X):\n", X.head())
    print("Target (y):\n", y.head())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    gbr = GradientBoostingRegressor(random_state=42)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.05, 0.1],
        'min_samples_split': [2, 5]
    }
    print("Performing grid search for GradientBoosting parameters...")
    grid = GridSearchCV(gbr, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid.fit(X_train, y_train)

    best_gbr = grid.best_estimator_
    print("Best parameters for GradientBoosting:", grid.best_params_)

    y_pred = best_gbr.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"GradientBoosting RMSE (test): {rmse:.2f}")

    feat_importances = pd.Series(best_gbr.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("Feature Importances:\n", feat_importances)

    # Show actual vs predicted for the first 10 test samples
    test_predictions = pd.DataFrame({
        "Actual": y_test.values[:10],
        "Predicted": y_pred[:10]
    })
    print("\nFirst 10 Customers' Actual vs Predicted AvgOrderValue:")
    print(test_predictions)

    ###############################################################################
    # 10. TIME SERIES FORECASTING (PROPHET)
    ###############################################################################
    """
    Daily TotalPrice forecast:
     - Last 6 months => RMSE
     - All data => next 6 months
    """
    df_time = df.copy()
    df_time['InvoiceDate'] = pd.to_datetime(df_time['InvoiceDate']).dt.date
    daily_sales = df_time.groupby('InvoiceDate')['TotalPrice'].sum().reset_index()
    daily_sales.columns = ['ds', 'y']

    cutoff_date = pd.to_datetime(daily_sales['ds'].max()) - pd.Timedelta(days=180)
    train_data = daily_sales[daily_sales['ds'] <= cutoff_date.date()]
    test_data = daily_sales[daily_sales['ds'] > cutoff_date.date()]

    print(f"Train set: {train_data['ds'].min()} to {train_data['ds'].max()}")
    print(f"Test set : {test_data['ds'].min()} to {test_data['ds'].max()}")

    prophet_model = Prophet(seasonality_mode='multiplicative', yearly_seasonality=True)
    prophet_model.fit(train_data)

    days_to_forecast = (test_data['ds'].max() - test_data['ds'].min()).days + 1
    future_test = prophet_model.make_future_dataframe(periods=days_to_forecast, freq='D')
    forecast_test = prophet_model.predict(future_test)

    forecast_test_mod = forecast_test[['ds', 'yhat']].set_index('ds')
    test_data_mod = test_data.set_index('ds')
    df_merged_ts = test_data_mod.join(forecast_test_mod, how='inner')
    rmse_test_prophet = math.sqrt(mean_squared_error(df_merged_ts['y'], df_merged_ts['yhat']))
    print(f"Prophet Test RMSE (last 6 months): {rmse_test_prophet:.2f}")

    prophet_final = Prophet(seasonality_mode='multiplicative', yearly_seasonality=True)
    prophet_final.fit(daily_sales)
    future_6m = prophet_final.make_future_dataframe(periods=180, freq='D')
    forecast_6m = prophet_final.predict(future_6m)

    prophet_final.plot(forecast_6m)
    plt.title("Prophet Forecast: Next 6 months of sales")
    plt.show()

    forecast_6m['ds'] = forecast_6m['ds'].dt.date
    forecast_period = forecast_6m[forecast_6m['ds'] > daily_sales['ds'].max()]

    sum_6m = forecast_period['yhat'].sum()
    avg_6m = forecast_period['yhat'].mean()
    print(f"Estimated total sales (next 6 months): {sum_6m:,.2f}")
    print(f"Estimated average daily sales (next 6 months): {avg_6m:,.2f}")
    print("\n--- Prophet forecasting completed successfully. ---\n")

    ###############################################################################
    # 11. ADDITIONAL VISUALIZATIONS & INSIGHTS
    ###############################################################################
    # Distribution of clusters (0..4 normal, 9 = outlier)
    cluster_counts = rfm['Cluster'].value_counts()
    plt.figure(figsize=(6, 6))
    plt.pie(cluster_counts, labels=cluster_counts.index, autopct='%1.1f%%', startangle=90)
    plt.title("Cluster Distribution (k=5 + Outlier=9)")
    plt.show()

    # Means of key metrics (Frequency, CLV, AvgOrderValue, ReturnRate) by cluster
    metrics_to_plot = ['Frequency', 'CLV', 'AvgOrderValue', 'ReturnRate']
    cluster_means = rfm.groupby('Cluster')[metrics_to_plot].mean()
    cluster_means.plot(kind='bar', figsize=(12, 6))
    plt.title("Cluster Means of Key Metrics (k=5 + Outlier=9)")
    plt.xlabel("Cluster")
    plt.ylabel("Value")
    plt.grid()
    plt.legend()
    plt.savefig("ClusterKeyMetrics.png")
    plt.show()

    # Seasonal CLV
    season_clv = rfm.groupby('Season')['CLV'].mean().sort_values(ascending=False)
    plt.figure(figsize=(8, 5))
    sns.barplot(x=season_clv.index, y=season_clv.values)
    plt.title("Seasonal Trends: Avg CLV per Season")
    plt.xlabel("Season")
    plt.ylabel("Avg CLV")
    plt.grid()
    plt.show()

    # Cluster vs. Category Heatmap
    df_merged2 = df.merge(rfm[['CustomerID', 'Cluster']], on='CustomerID', how='left')
    product_info_2 = pd.read_csv("product_metadata.csv")
    df_full = df_merged2.merge(product_info_2[['ProductID', 'Category', 'Brand']],
                               left_on='StockCode', right_on='ProductID', how='left')
    cluster_category_sales = df_full.groupby(['Cluster', 'Category'])['TotalPrice'].sum().unstack().fillna(0)

    plt.figure(figsize=(12, 8))
    sns.heatmap(cluster_category_sales, cmap="viridis", annot=False, cbar=True)
    plt.title("Cluster vs Category Sales (k=5 + Outlier=9)")
    plt.xlabel("Category")
    plt.ylabel("Cluster")
    plt.xticks(rotation=45)
    plt.show()

    # Marketing strategy (based on cluster_summary)
    cluster_summary = cluster_summary.sort_values(by='Cluster')
    for cluster_id in cluster_summary['Cluster']:
        c_data = cluster_summary[cluster_summary['Cluster'] == cluster_id]
        avg_clv = c_data['CLV'].values[0]
        avg_freq = c_data['Frequency'].values[0]
        avg_ret = c_data['ReturnRate'].values[0]
        print(f"Cluster {cluster_id}:")

        if cluster_id == 9:
            print(" - Manually separated VIP/Wholesaler segment (CLV>50k).")
            print("   Special marketing strategies, B2B deals, VIP services, etc.")
        elif avg_clv > 1500 and avg_freq > 15 and avg_ret < 0.1:
            print(" - Highly Premium Customers:")
            print("   Exclusive services, special product access, VIP event invitations.")
        elif avg_clv > 1000 and avg_freq > 10 and avg_ret < 0.1:
            print(" - Premium Customers:")
            print("   VIP discounts, personalized emails, special product launches.")
        elif avg_ret > 0.2:
            print(" - High Return Rate:")
            print("   Improve return policy, enhance product descriptions, customer surveys.")
        elif avg_clv > 500 and avg_freq > 5:
            print(" - Mid-Long Term Loyal Customers:")
            print("   Periodic campaigns, reminder emails, incentive coupons.")
        else:
            print(" - New/Low Value Customers:")
            print("   Cross-sell & up-sell opportunities, entry-level promotions.")
        print()

    ###############################################################################
    # 12. BASE MODEL COMPARISON & RANDOMIZED SEARCH (OPTIONAL)
    ###############################################################################
    """
    Optional: Compare with other models (RF, GB, XGB, etc.)
    RandomizedSearch example for an RF with a wide parameter range
    """


    def base_models_comparison(X, y, scoring=('neg_mean_squared_error', 'r2', 'neg_mean_absolute_error')):
        """
        Compare multiple base models using cross-validation:
        - RandomForest
        - GradientBoosting
        - (Optionally XGB, LGB, CatBoost if installed)
        """
        models = {}
        models["RandomForest"] = RandomForestRegressor(random_state=42)
        models["GradientBoosting"] = GradientBoostingRegressor(random_state=42)

        if xgb_installed:
            from xgboost import XGBRegressor
            models["XGBRegressor"] = XGBRegressor(random_state=42, eval_metric='rmse', use_label_encoder=False)
        if lgb_installed:
            from lightgbm import LGBMRegressor
            models["LGBMRegressor"] = LGBMRegressor(random_state=42)
        if cat_installed:
            from catboost import CatBoostRegressor
            models["CatBoostRegressor"] = CatBoostRegressor(verbose=0, random_state=42)

        results = {}
        for model_name, model in models.items():
            cv_res = cross_validate(model, X, y, scoring=scoring, cv=5, return_train_score=True, n_jobs=-1)
            res_dict = {}
            for sc in scoring:
                train_sc = f'train_{sc}'
                test_sc = f'test_{sc}'
                res_dict[f'Train_{sc}'] = cv_res[train_sc].mean()
                res_dict[f'Test_{sc}'] = cv_res[test_sc].mean()
            results[model_name] = res_dict
        return pd.DataFrame(results).T


    def random_search_example(X, y):
        """
        Conducts a RandomizedSearchCV for a RandomForestRegressor with a wide parameter range.
        """
        param_dist = {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [3, 5, 7, 10, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        rf = RandomForestRegressor(random_state=42)
        random_search = RandomizedSearchCV(
            rf,
            param_dist,
            n_iter=10,
            scoring='neg_root_mean_squared_error',
            cv=5,
            random_state=42,
            n_jobs=-1
        )
        random_search.fit(X, y)
        print("RandomizedSearch Best Params:", random_search.best_params_)
        print("RandomizedSearch Best RMSE:", -random_search.best_score_)


    print("\n=== Checking some advanced comparisons ===")
    X_reg = X
    y_reg = y
    bm_results = base_models_comparison(X_reg, y_reg)
    print(bm_results)

    random_search_example(X_reg, y_reg)

    # Results and Recommendations
    # RandomForest vs. GradientBoosting Comparison
    # RandomForest:
    # Generally provides balanced performance.
    # After hyperparameter optimization, a good RMSE is achieved.
    # GradientBoosting:
    # Performs better on the training set, but may show signs of overfitting on the test set.

###############################################################################
# STEP X: FUTURE ENHANCEMENTS & REFACTORING IDEAS
###############################################################################
"""
Below are potential enhancements and ideas for improving or refactoring
the project without disrupting the current working structure:
"""

# 1) Improve time-series splits for certain predictive tasks:
#    - Use rolling or expanding windows instead of random splits
#      when dealing with time-based data (beyond Prophet).
#
# 2) Expand feature engineering (particularly for RFM or Country encoding):
#    - Possibly encode Country with one-hot if interpretability is needed.
#    - Additional time-based features like ActiveDays, etc.
#
# 3) Consider alternative outlier detection beyond IQR:
#    - For instance, domain-specific thresholds or advanced algorithms (IsolationForest, DBSCAN).
#
# 4) Unify or remove duplicated sections:
#    - Repeated imports at the top and middle of the file.
#    - Repeated definitions of numeric_features, categorical_features.
#    - Possibly unify data cleaning in one place instead of repeated steps.
#
# 5) Explore non-KMeans clustering or advanced recommendation approaches:
#    - DBSCAN or HDBSCAN for non-spherical clusters.
#    - Hybrid or advanced matrix factorization techniques for recommendation.
#
# 6) Try hyperparameter tuning with advanced techniques or additional models:
#    - LightGBM, CatBoost if installed, or Bayesian Optimization for tuning.
#
# 7) Add holiday/event effects to Prophet if relevant:
#    - Incorporate known holidays (e.g., Christmas, Black Friday, etc.)
#    - Use Prophet’s built-in holiday regressors or add external regressors.


# --------------- END OF CODE ---------------
