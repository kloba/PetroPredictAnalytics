# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   },
# META   "dependencies": {
# META     "lakehouse": {
# META       "default_lakehouse": "7023c41f-8e27-473c-bbb4-7916a15fe30a",
# META       "default_lakehouse_name": "PetroPredictLakehouse",
# META       "default_lakehouse_workspace_id": "9cca7133-c6b1-4dc6-a7f1-93e2962df03b",
# META       "known_lakehouses": [
# META         {
# META           "id": "7023c41f-8e27-473c-bbb4-7916a15fe30a"
# META         }
# META       ]
# META     }
# META   }
# META }

# CELL ********************

import requests
import pandas as pd
from datetime import datetime, timedelta
import json
import numpy as np

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Configuration
# Get API key from environment variable
import os
EIA_API_KEY = os.getenv("EIA_API_KEY")  # Set this as an environment variable

# Date range - Last 1000 days by default
END_DATE = datetime.now().strftime("%Y-%m-%d")
START_DATE = (datetime.now() - timedelta(days=1000)).strftime("%Y-%m-%d")

print(f"Configured to fetch data from {START_DATE} to {END_DATE}")
print(f"API Key: {EIA_API_KEY[:10]}..." if EIA_API_KEY else "No API key provided")

# Check if API key is set
if not EIA_API_KEY:
    raise ValueError("EIA_API_KEY environment variable not set. Please set it before running this notebook.")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

def fetch_oil_prices(api_key, start_date, end_date):
    """
    Fetch historical oil prices from EIA API
    
    Parameters:
    - api_key: EIA API key
    - start_date: Start date in format 'YYYY-MM-DD'
    - end_date: End date in format 'YYYY-MM-DD'
    
    Returns:
    - DataFrame with columns: date, oil_type, price_usd
    """
    
    # EIA API v2 endpoint for spot prices
    base_url = "https://api.eia.gov/v2/petroleum/pri/spt/data/"
    
    all_data = []
    
    # Product codes for different oil types
    products = {
        "EPCBRENT": "Brent",  # Europe Brent Spot Price FOB (Free on Board)
        "EPCWTI": "WTI"       # West Texas Intermediate Spot Price
    }
    
    for product_code, oil_type in products.items():
        # API parameters
        params = {
            "api_key": api_key,
            "frequency": "daily",
            "data[0]": "value",
            "start": start_date,
            "end": end_date,
            "facets[product][]": product_code,
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "length": 5000  # Max records to return
        }
        
        try:
            print(f"\nFetching {oil_type} prices...")
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if "response" in data and "data" in data["response"]:
                records = data["response"]["data"]
                print(f"API returned {len(records)} records for {oil_type}")
                
                if records:
                    # Convert to DataFrame
                    df = pd.DataFrame(records)
                    df["period"] = pd.to_datetime(df["period"])
                    df = df.rename(columns={"period": "date", "value": "price_usd"})
                    df["oil_type"] = oil_type
                    df["price_usd"] = pd.to_numeric(df["price_usd"], errors='coerce')
                    
                    # Select and clean columns
                    df = df[["date", "oil_type", "price_usd"]]
                    df = df.dropna(subset=["price_usd"])
                    
                    all_data.append(df)
                    print(f"Successfully processed {len(df)} {oil_type} price records")
                else:
                    print(f"No data records found for {oil_type}")
            else:
                print(f"Unexpected API response structure for {oil_type}")
                
        except requests.exceptions.RequestException as e:
            print(f"Request error fetching {oil_type} prices: {e}")
        except Exception as e:
            print(f"Unexpected error fetching {oil_type} prices: {type(e).__name__}: {e}")
    
    # Combine all data
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"\nTotal combined records: {len(combined_df)}")
        return combined_df
    else:
        print("\nNo oil price data could be fetched")
        return pd.DataFrame(columns=["date", "oil_type", "price_usd"])

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Fetch the data
oil_prices_df = fetch_oil_prices(EIA_API_KEY, START_DATE, END_DATE)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Display basic information
print("DataFrame Info:")
print(oil_prices_df.info())
print("\n" + "="*60 + "\n")

# Display summary statistics
if not oil_prices_df.empty:
    print(f"Total records: {len(oil_prices_df)}")
    print(f"Date range: {oil_prices_df['date'].min()} to {oil_prices_df['date'].max()}")
    print(f"\nRecords by oil type:")
    print(oil_prices_df['oil_type'].value_counts())
    
    print("\nPrice statistics by oil type:")
    stats = oil_prices_df.groupby('oil_type')['price_usd'].agg(['mean', 'min', 'max', 'std'])
    print(stats.round(2))
else:
    print("No data available")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Display first 10 records
print("First 10 records:")
oil_prices_df.head(10)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Display last 10 records
print("Last 10 records:")
oil_prices_df.tail(10)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Optional: Create a simple visualization
try:
    import matplotlib.pyplot as plt
    
    if not oil_prices_df.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for oil_type in oil_prices_df['oil_type'].unique():
            data = oil_prices_df[oil_prices_df['oil_type'] == oil_type]
            ax.plot(data['date'], data['price_usd'], marker='o', label=oil_type, markersize=4)
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Price (USD per barrel)')
        ax.set_title('Oil Prices Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
except ImportError:
    print("Matplotlib not installed. Install with: pip install matplotlib")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Import necessary PySpark functions
from pyspark.sql import SparkSession

# Create a Spark session
spark = SparkSession.builder.appName("StoreOilPrices").getOrCreate()

# Convert pandas DataFrame to Spark DataFrame
import pandas as pd
if isinstance(oil_prices_df, pd.DataFrame):
    spark_df = spark.createDataFrame(oil_prices_df)
else:
    spark_df = oil_prices_df

# Define the table name in the lakehouse
table_name = "PetroPredictLakehouse.OilPrices"

# Save the Spark DataFrame to the table in the lakehouse
#spark_df.write.format("delta").saveAsTable(table_name)

# If you need to overwrite the table, use the following line instead
spark_df.write.mode("overwrite").format("delta").saveAsTable(table_name)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
