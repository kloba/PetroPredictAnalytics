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
import time
import json
import numpy as np

# For visualization (optional)
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style("whitegrid")
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Matplotlib/Seaborn not available. Install for visualizations: pip install matplotlib seaborn")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Configuration
DAYS_BACK = 1000  # Number of days to go back from today
CURRENCIES = ['USD', 'EUR']  # Currencies to fetch (against PLN)

# Calculate date range
END_DATE = datetime.now()
START_DATE = END_DATE - timedelta(days=DAYS_BACK)

print(f"Configuration:")
print(f"- Days to fetch: {DAYS_BACK}")
print(f"- Date range: {START_DATE.strftime('%Y-%m-%d')} to {END_DATE.strftime('%Y-%m-%d')}")
print(f"- Currencies: {', '.join(CURRENCIES)} (against PLN)")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

def fetch_nbp_exchange_rates(currency_code, start_date, end_date):
    """
    Fetch exchange rates from NBP API for a specific currency
    
    Parameters:
    - currency_code: Currency code (e.g., 'USD', 'EUR')
    - start_date: Start date as string 'YYYY-MM-DD'
    - end_date: End date as string 'YYYY-MM-DD'
    
    Returns:
    - List of dictionaries with date and rate
    """
    
    # NBP API endpoint for exchange rates table A (mid rates)
    base_url = "http://api.nbp.pl/api/exchangerates/rates/a"
    
    # NBP API has a limit of 367 days per request
    max_days_per_request = 367
    
    all_rates = []
    current_start = datetime.strptime(start_date, "%Y-%m-%d")
    final_end = datetime.strptime(end_date, "%Y-%m-%d")
    
    request_count = 0
    
    while current_start <= final_end:
        # Calculate chunk end date
        chunk_end = min(current_start + timedelta(days=max_days_per_request - 1), final_end)
        
        # Format dates for API
        api_start = current_start.strftime("%Y-%m-%d")
        api_end = chunk_end.strftime("%Y-%m-%d")
        
        # Construct URL
        url = f"{base_url}/{currency_code.lower()}/{api_start}/{api_end}/"
        
        request_count += 1
        print(f"  Request {request_count}: Fetching {currency_code}/PLN from {api_start} to {api_end}...", end="")
        
        try:
            response = requests.get(url, headers={'Accept': 'application/json'})
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract rates
                rates = data.get('rates', [])
                for rate_entry in rates:
                    all_rates.append({
                        'date': rate_entry['effectiveDate'],
                        'rate': rate_entry['mid']  # NBP provides mid rate
                    })
                
                print(f" ✓ ({len(rates)} rates)")
                
            elif response.status_code == 404:
                print(f" ⚠ No data available")
            else:
                print(f" ✗ Error: HTTP {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f" ✗ Request error: {type(e).__name__}")
        except Exception as e:
            print(f" ✗ Unexpected error: {type(e).__name__}")
        
        # Move to next chunk
        current_start = chunk_end + timedelta(days=1)
        
        # Be nice to the API - small delay between requests
        if current_start <= final_end:
            time.sleep(0.5)
    
    return all_rates

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

def collect_fx_rates(currencies, start_date, end_date):
    """
    Collect FX rates for multiple currencies against PLN
    
    Returns:
    - DataFrame with columns: date, from_currency, to_currency, exchange_rate
    """
    
    all_fx_data = []
    
    print("Fetching exchange rates from NBP API...\n")
    
    for currency in currencies:
        print(f"\nFetching {currency}/PLN rates:")
        rates = fetch_nbp_exchange_rates(currency, start_date, end_date)
        
        # Convert to standard format
        for rate_entry in rates:
            all_fx_data.append({
                'date': pd.to_datetime(rate_entry['date']),
                'from_currency': currency,
                'to_currency': 'PLN',
                'exchange_rate': rate_entry['rate']
            })
        
        print(f"  Total {currency}/PLN rates collected: {len(rates)}")
    
    # Create DataFrame
    df = pd.DataFrame(all_fx_data)
    
    # Sort by date and currency
    if not df.empty:
        df = df.sort_values(['date', 'from_currency']).reset_index(drop=True)
    
    return df

# Fetch the data
start_str = START_DATE.strftime("%Y-%m-%d")
end_str = END_DATE.strftime("%Y-%m-%d")

fx_rates_df = collect_fx_rates(CURRENCIES, start_str, end_str)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Display DataFrame info
print("DataFrame Info:")
print(fx_rates_df.info())
print("\n" + "="*60 + "\n")

if not fx_rates_df.empty:
    # Summary statistics
    print(f"Total records: {len(fx_rates_df):,}")
    print(f"Date range: {fx_rates_df['date'].min()} to {fx_rates_df['date'].max()}")
    print(f"Days covered: {(fx_rates_df['date'].max() - fx_rates_df['date'].min()).days + 1:,}")
    
    print("\nRecords by currency pair:")
    pair_counts = fx_rates_df.groupby(['from_currency', 'to_currency']).size()
    for pair, count in pair_counts.items():
        print(f"  {pair[0]}/{pair[1]}: {count:,} records")
    
    print("\nExchange rate statistics:")
    stats = fx_rates_df.groupby('from_currency')['exchange_rate'].agg(['mean', 'min', 'max', 'std'])
    print(stats.round(4))
    
    # Data completeness check
    print("\nData completeness analysis:")
    for currency in fx_rates_df['from_currency'].unique():
        currency_data = fx_rates_df[fx_rates_df['from_currency'] == currency]
        date_range = pd.date_range(start=currency_data['date'].min(), 
                                  end=currency_data['date'].max(), 
                                  freq='D')
        business_days = pd.bdate_range(start=currency_data['date'].min(), 
                                      end=currency_data['date'].max())
        missing_dates = len(date_range) - len(currency_data)
        completeness = (len(currency_data) / len(business_days) * 100) if len(business_days) > 0 else 0
        
        print(f"  {currency}/PLN:")
        print(f"    - Records: {len(currency_data):,}")
        print(f"    - Missing dates: {missing_dates:,} (weekends/holidays)")
        print(f"    - Business days coverage: {completeness:.1f}%")
else:
    print("No data collected")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

if not fx_rates_df.empty:
    print("First 10 records:")
    display(fx_rates_df.head(10))
    
    print("\nLast 10 records:")
    display(fx_rates_df.tail(10))
    
    # Show some statistics by currency
    print("\nRecent rates (last 5 days):")
    recent_dates = fx_rates_df['date'].unique()[-5:]
    recent_data = fx_rates_df[fx_rates_df['date'].isin(recent_dates)]
    
    # Create pivot table and reset index to avoid display issues
    pivot_recent = recent_data.pivot(index='date', columns='from_currency', values='exchange_rate')
    pivot_recent = pivot_recent.reset_index()
    
    # Display the pivot table
    display(pivot_recent)
    
    # Alternative: print the pivot table if display still has issues
    print("\nRecent exchange rates:")
    print(pivot_recent.to_string())

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

if PLOTTING_AVAILABLE and not fx_rates_df.empty:
    # Create subplots for each currency
    currencies = fx_rates_df['from_currency'].unique()
    
    fig, axes = plt.subplots(len(currencies), 1, figsize=(14, 6*len(currencies)))
    if len(currencies) == 1:
        axes = [axes]
    
    for idx, currency in enumerate(currencies):
        currency_data = fx_rates_df[fx_rates_df['from_currency'] == currency]
        
        # Plot the exchange rate
        axes[idx].plot(currency_data['date'], currency_data['exchange_rate'], 
                      color='darkblue', linewidth=1.5, label=f'{currency}/PLN')
        
        # Add moving averages
        currency_data_sorted = currency_data.sort_values('date')
        ma_30 = currency_data_sorted['exchange_rate'].rolling(window=30, center=True).mean()
        ma_90 = currency_data_sorted['exchange_rate'].rolling(window=90, center=True).mean()
        
        axes[idx].plot(currency_data_sorted['date'], ma_30, 
                      color='red', linewidth=1, alpha=0.7, label='30-day MA')
        axes[idx].plot(currency_data_sorted['date'], ma_90, 
                      color='green', linewidth=1, alpha=0.7, label='90-day MA')
        
        # Formatting
        axes[idx].set_title(f'{currency}/PLN Exchange Rate Over Time', fontsize=14, fontweight='bold')
        axes[idx].set_xlabel('Date')
        axes[idx].set_ylabel('Exchange Rate')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
        
        # Add min/max annotations
        min_rate = currency_data['exchange_rate'].min()
        max_rate = currency_data['exchange_rate'].max()
        min_date = currency_data.loc[currency_data['exchange_rate'].idxmin(), 'date']
        max_date = currency_data.loc[currency_data['exchange_rate'].idxmax(), 'date']
        
        axes[idx].annotate(f'Min: {min_rate:.4f}', 
                          xy=(min_date, min_rate), 
                          xytext=(10, 10), 
                          textcoords='offset points',
                          ha='left',
                          bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))
        
        axes[idx].annotate(f'Max: {max_rate:.4f}', 
                          xy=(max_date, max_rate), 
                          xytext=(10, -10), 
                          textcoords='offset points',
                          ha='left',
                          bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))
    
    plt.tight_layout()
    plt.show()
    
    # Create a correlation plot if we have both currencies
    if len(currencies) == 2:
        print("\nExchange Rate Correlation Analysis:")
        
        # Pivot the data
        pivot_df = fx_rates_df.pivot(index='date', columns='from_currency', values='exchange_rate')
        
        # Calculate correlation
        correlation = pivot_df.corr()
        print(f"\nCorrelation between {currencies[0]}/PLN and {currencies[1]}/PLN: {correlation.iloc[0, 1]:.4f}")
        
        # Scatter plot
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(pivot_df[currencies[0]], pivot_df[currencies[1]], alpha=0.5)
        ax.set_xlabel(f'{currencies[0]}/PLN')
        ax.set_ylabel(f'{currencies[1]}/PLN')
        ax.set_title('Exchange Rate Correlation', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(pivot_df[currencies[0]].dropna(), pivot_df[currencies[1]].dropna(), 1)
        p = np.poly1d(z)
        ax.plot(pivot_df[currencies[0]], p(pivot_df[currencies[0]]), "r--", alpha=0.8, label=f'Trend line')
        ax.legend()
        
        plt.tight_layout()
        plt.show()
elif not PLOTTING_AVAILABLE:
    print("Install matplotlib and seaborn for visualizations: pip install matplotlib seaborn")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Import necessary PySpark functions
from pyspark.sql import SparkSession

# Create a Spark session
spark = SparkSession.builder.appName("StoreFXRates").getOrCreate()

# Convert pandas DataFrame to Spark DataFrame
import pandas as pd
if isinstance(fx_rates_df, pd.DataFrame):
    spark_df = spark.createDataFrame(fx_rates_df)
else:
    spark_df = fx_rates_df

# Define the table name in the lakehouse
table_name = "PetroPredictLakehouse.FXRates"

# Save the Spark DataFrame to the table in the lakehouse
#spark_df.write.format("delta").saveAsTable(table_name)

# If you need to overwrite the table, use the following line instead
spark_df.write.mode("overwrite").format("delta").saveAsTable(table_name)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
