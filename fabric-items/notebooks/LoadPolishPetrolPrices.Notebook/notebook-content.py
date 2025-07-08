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
import numpy as np
from datetime import datetime, timedelta
import re
from bs4 import BeautifulSoup
import json
from pyspark.sql import SparkSession
import subprocess

# Import visualization libraries
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

# Create Spark session for accessing lakehouse data
spark = SparkSession.builder.appName("CollectPolishPetrolPrices").getOrCreate()
print("Spark session initialized")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

def fetch_lotos_archive_data():
    """
    Fetch historical petrol prices from Lotos archive page using curl/wget
    URL: https://www.lotos.pl/145/type,oil_95/dla_biznesu/hurtowe_ceny_paliw/archiwum_cen_paliw
    """
    url = "https://www.lotos.pl/145/type,oil_95/dla_biznesu/hurtowe_ceny_paliw/archiwum_cen_paliw"
    
    print("Fetching data from Lotos archive...")
    
    # Method 1: Try using curl first
    try:
        print("Trying curl...")
        result = subprocess.run(
            ['curl', '-s', '-L', 
             '-H', 'User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
             '-H', 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
             '-H', 'Accept-Language: pl-PL,pl;q=0.9,en;q=0.8',
             url],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            html_content = result.stdout
            print("✓ Successfully fetched with curl")
        else:
            raise Exception(f"Curl failed with code {result.returncode}")
            
    except Exception as e:
        print(f"Curl failed: {e}")
        
        # Method 2: Try wget as fallback
        try:
            print("Trying wget...")
            result = subprocess.run(
                ['wget', '-q', '-O', '-',
                 '--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                 '--header=Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                 '--header=Accept-Language: pl-PL,pl;q=0.9,en;q=0.8',
                 url],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                html_content = result.stdout
                print("✓ Successfully fetched with wget")
            else:
                raise Exception(f"Wget failed with code {result.returncode}")
                
        except Exception as e2:
            print(f"Wget failed: {e2}")
            
            # Method 3: Fall back to requests
            print("Falling back to requests...")
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'pl-PL,pl;q=0.9,en;q=0.8',
            }
            
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            html_content = response.text
            print("✓ Successfully fetched with requests")
    
    # Parse the HTML content
    return parse_lotos_html(html_content)

def parse_lotos_html(html_content):
    """Parse Lotos HTML content to extract price data"""
    
    # Try pandas read_html first
    try:
        tables = pd.read_html(html_content)
        if tables:
            print(f"Found {len(tables)} tables using pandas")
            
            # Find the table with dates
            for i, table in enumerate(tables):
                # Check if table has date-like column
                for col in table.columns:
                    try:
                        dates = pd.to_datetime(table[col], errors='coerce')
                        if dates.notna().sum() > 10:  # At least 10 valid dates
                            print(f"Found price table at index {i}")
                            return process_lotos_table(table)
                    except:
                        pass
    except Exception as e:
        print(f"Pandas read_html failed: {e}")
    
    # If pandas fails, use BeautifulSoup
    soup = BeautifulSoup(html_content, 'lxml')
    
    # Look for table elements
    tables = soup.find_all('table')
    print(f"Found {len(tables)} tables in HTML")
    
    # Try to extract data from each table
    for table in tables:
        rows = table.find_all('tr')
        if len(rows) > 10:  # Reasonable size for price archive
            data_rows = []
            
            for row in rows:
                cells = row.find_all(['td', 'th'])
                if cells:
                    row_data = [cell.get_text(strip=True) for cell in cells]
                    data_rows.append(row_data)
            
            if data_rows:
                df = pd.DataFrame(data_rows[1:], columns=data_rows[0] if data_rows[0] else None)
                processed = process_lotos_table(df)
                if processed is not None and not processed.empty:
                    return processed
    
    # Last resort: regex extraction
    print("Using regex extraction as last resort...")
    
    # Extract text and look for date/price patterns
    text = soup.get_text()
    lines = text.split('\n')
    
    data_rows = []
    date_pattern = re.compile(r'(\d{4}-\d{2}-\d{2})')
    
    for i, line in enumerate(lines):
        date_match = date_pattern.search(line)
        if date_match:
            date = date_match.group(1)
            
            # Look for prices in same or next few lines
            prices = []
            for j in range(i, min(i+5, len(lines))):
                # Find numbers that could be prices (3000-6000 range for PLN/1000L)
                nums = re.findall(r'(\d{3,4}[,.]?\d{0,2})', lines[j])
                for num in nums:
                    clean_num = float(num.replace(',', '.').replace(' ', ''))
                    if 2000 < clean_num < 7000:  # Reasonable PLN price range
                        prices.append(clean_num)
            
            if prices:
                data_rows.append({
                    'date': date,
                    'price': prices[0],  # Main price
                    'excise_tax': prices[1] if len(prices) > 1 else 1529,  # Typical excise tax
                    'fuel_charge': prices[2] if len(prices) > 2 else 150  # Typical fuel charge
                })
    
    if data_rows:
        df = pd.DataFrame(data_rows)
        df['date'] = pd.to_datetime(df['date'])
        print(f"Extracted {len(df)} rows using regex")
        return df
    
    return None

def process_lotos_table(df):
    """Process Lotos table to extract date and price columns"""
    if df.empty:
        return None
        
    processed_df = pd.DataFrame()
    
    # Find date column
    date_col = None
    for col in df.columns:
        try:
            # Try to parse as dates
            test_col = df[col].astype(str).str.strip()
            dates = pd.to_datetime(test_col, errors='coerce')
            if dates.notna().sum() > len(df) * 0.5:
                date_col = col
                processed_df['date'] = dates
                print(f"Found date column: {col}")
                break
        except:
            pass
    
    if date_col is None:
        print("Could not find date column")
        return None
    
    # Find price columns (numeric columns)
    price_cols = []
    for col in df.columns:
        if col != date_col:
            try:
                # Convert to numeric, handling Polish decimal separator
                test_col = df[col].astype(str).str.strip()
                numeric_data = pd.to_numeric(
                    test_col.str.replace(',', '.').str.replace(' ', ''), 
                    errors='coerce'
                )
                
                if numeric_data.notna().sum() > len(df) * 0.5:
                    # Check if values are in reasonable range
                    mean_val = numeric_data.mean()
                    if 100 < mean_val < 10000:  # Reasonable price range
                        price_cols.append((col, numeric_data))
                        print(f"Found numeric column: {col} (mean: {mean_val:.2f})")
            except:
                pass
    
    # Assign price columns
    if price_cols:
        processed_df['price'] = price_cols[0][1]  # Main price
        if len(price_cols) > 1:
            processed_df['excise_tax'] = price_cols[1][1]
        else:
            processed_df['excise_tax'] = 1529  # Default excise tax
            
        if len(price_cols) > 2:
            processed_df['fuel_charge'] = price_cols[2][1]
        else:
            processed_df['fuel_charge'] = 150  # Default fuel charge
    else:
        print("No price columns found")
        return None
    
    # Clean and sort
    processed_df = processed_df.dropna(subset=['date', 'price'])
    processed_df = processed_df.sort_values('date')
    
    print(f"Processed {len(processed_df)} valid rows")
    
    return processed_df

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Fetch data from Lotos archive
print("=" * 80)
print("FETCHING LOTOS HISTORICAL DATA")
print("=" * 80)

lotos_df = fetch_lotos_archive_data()

if lotos_df is not None and not lotos_df.empty:
    print(f"\n✓ Successfully fetched {len(lotos_df)} records from Lotos archive")
    print(f"Date range: {lotos_df['date'].min()} to {lotos_df['date'].max()}")
    print("\nSample data:")
    display(lotos_df.head(10))
    
    # Convert PLN prices to USD using exchange rate
    try:
        # Get exchange rate from lakehouse or use default
        fx_rates_df = spark.read.table("PetroPredictLakehouse.FXRates")
        latest_usd_pln = fx_rates_df.filter(
            (fx_rates_df.from_currency == "USD") & 
            (fx_rates_df.to_currency == "PLN")
        ).orderBy(fx_rates_df.date.desc()).first()
        
        if latest_usd_pln:
            exchange_rate = float(latest_usd_pln['exchange_rate'])
        else:
            exchange_rate = 4.0
    except:
        exchange_rate = 4.0
    
    # Add USD prices
    lotos_df['price_usd_per_liter'] = (lotos_df['price'] / exchange_rate / 1000).round(3)  # Convert from PLN/1000L to USD/L
    lotos_df['price_pln_per_liter'] = (lotos_df['price'] / 1000).round(3)  # Convert from PLN/1000L to PLN/L
    
    print(f"\nPrice statistics:")
    print(f"Average price: {lotos_df['price_pln_per_liter'].mean():.2f} PLN/L ({lotos_df['price_usd_per_liter'].mean():.2f} USD/L)")
    print(f"Min price: {lotos_df['price_pln_per_liter'].min():.2f} PLN/L ({lotos_df['price_usd_per_liter'].min():.2f} USD/L)")
    print(f"Max price: {lotos_df['price_pln_per_liter'].max():.2f} PLN/L ({lotos_df['price_usd_per_liter'].max():.2f} USD/L)")
    
    # Prepare for lakehouse storage
    lotos_records = []
    for _, row in lotos_df.iterrows():
        lotos_records.append({
            'date': row['date'],
            'fuel_type': 'Gasoline 95',
            'price_usd_per_liter': row['price_usd_per_liter'],
            'price_pln_per_liter': row['price_pln_per_liter'],
            'country': 'Poland',
            'currency_original': 'PLN',
            'source': 'Lotos',
            'data_type': 'historical',
            'exchange_rate_used': exchange_rate
        })
    
    lotos_prices_df = pd.DataFrame(lotos_records)
    print(f"\nPrepared {len(lotos_prices_df)} records for storage")
    
else:
    print("\n❌ Failed to fetch Lotos data.")
    lotos_prices_df = pd.DataFrame()  # Empty DataFrame

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Visualize Lotos historical data
if lotos_df is not None and not lotos_df.empty and PLOTTING_AVAILABLE:
    print("\n" + "=" * 80)
    print("LOTOS DATA VISUALIZATION")
    print("=" * 80)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Plot 1: Price components over time
    ax1 = axes[0, 0]
    ax1.stackplot(lotos_df['date'], 
                  lotos_df['price'] - lotos_df['excise_tax'] - lotos_df['fuel_charge'],
                  lotos_df['excise_tax'],
                  lotos_df['fuel_charge'],
                  labels=['Base Price', 'Excise Tax', 'Fuel Charge'],
                  alpha=0.7,
                  colors=['#3498db', '#e74c3c', '#f39c12'])
    
    ax1.set_title('Price Components Over Time (PLN/1000L)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price (PLN/1000L)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Price per liter over time
    ax2 = axes[0, 1]
    ax2.plot(lotos_df['date'], lotos_df['price_pln_per_liter'], 
             color='#2ecc71', linewidth=2, label='PLN/Liter')
    ax2.set_title('Petrol Price per Liter', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Price (PLN/Liter)')
    ax2.grid(True, alpha=0.3)
    
    # Add statistics
    mean_price = lotos_df['price_pln_per_liter'].mean()
    ax2.axhline(y=mean_price, color='red', linestyle='--', alpha=0.7,
                label=f'Average: {mean_price:.2f} PLN/L')
    ax2.legend()
    
    # Plot 3: Monthly average prices
    ax3 = axes[1, 0]
    lotos_df['year_month'] = pd.to_datetime(lotos_df['date']).dt.to_period('M')
    monthly_avg = lotos_df.groupby('year_month')['price_pln_per_liter'].mean()
    
    # Convert period index to timestamp for plotting
    monthly_dates = monthly_avg.index.to_timestamp()
    ax3.bar(monthly_dates, monthly_avg.values, width=20, color='#9b59b6', alpha=0.7)
    ax3.set_title('Monthly Average Prices', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Month')
    ax3.set_ylabel('Average Price (PLN/Liter)')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Price distribution histogram
    ax4 = axes[1, 1]
    ax4.hist(lotos_df['price_pln_per_liter'], bins=30, color='#34495e', 
             alpha=0.7, edgecolor='black')
    ax4.axvline(lotos_df['price_pln_per_liter'].mean(), color='red', 
                linestyle='--', linewidth=2, 
                label=f'Mean: {lotos_df["price_pln_per_liter"].mean():.2f} PLN')
    ax4.axvline(lotos_df['price_pln_per_liter'].median(), color='blue', 
                linestyle='--', linewidth=2, 
                label=f'Median: {lotos_df["price_pln_per_liter"].median():.2f} PLN')
    ax4.set_title('Price Distribution', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Price (PLN/Liter)')
    ax4.set_ylabel('Frequency')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    # Additional statistics
    print(f"\nPrice Statistics (PLN/Liter):")
    print(f"  Mean: {lotos_df['price_pln_per_liter'].mean():.3f}")
    print(f"  Median: {lotos_df['price_pln_per_liter'].median():.3f}")
    print(f"  Std Dev: {lotos_df['price_pln_per_liter'].std():.3f}")
    print(f"  Min: {lotos_df['price_pln_per_liter'].min():.3f}")
    print(f"  Max: {lotos_df['price_pln_per_liter'].max():.3f}")
    print(f"  Range: {lotos_df['price_pln_per_liter'].max() - lotos_df['price_pln_per_liter'].min():.3f}")
    
elif not PLOTTING_AVAILABLE:
    print("Install matplotlib and seaborn for visualizations: pip install matplotlib seaborn")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Save Lotos data to lakehouse
print("\n" + "=" * 80)
print("SAVING LOTOS DATA TO LAKEHOUSE")
print("=" * 80)

if not lotos_prices_df.empty:
    # Convert to Spark DataFrame
    spark_df = spark.createDataFrame(lotos_prices_df)
    
    # Define the table name in the lakehouse
    table_name = "PetroPredictLakehouse.PolishPetrolPrices"
    
    try:
        # Try to read existing table to check schema
        existing_table = spark.read.table(table_name)
        existing_columns = existing_table.columns
        
        # Save with append mode
        spark_df.write.mode("append").format("delta").saveAsTable(table_name)
        
    except Exception as e:
        if "Table or view not found" in str(e):
            # Table doesn't exist, create it with full schema
            print("Creating new table...")
            spark_df.write.mode("overwrite").format("delta").saveAsTable(table_name)
        else:
            # Schema mismatch - enable merge schema
            print("Schema mismatch detected, enabling schema merge...")
            spark_df.write.mode("append").option("mergeSchema", "true").format("delta").saveAsTable(table_name)
    
    print(f"\n✓ Data successfully saved to {table_name}")
    print(f"Total records saved: {spark_df.count()}")
    
    # Show the latest records from the table
    print("\nLatest records in the table:")
    latest_records = spark.read.table(table_name).orderBy("date", ascending=False).limit(10)
    display(latest_records.toPandas())
    
    # Show summary statistics
    print("\nData summary:")
    total_records = spark.read.table(table_name).count()
    print(f"Total records in table: {total_records}")
    
    # Date range - use proper aggregation syntax
    from pyspark.sql import functions as F
    date_stats = spark.read.table(table_name).select(
        F.min("date").alias("min_date"),
        F.max("date").alias("max_date")
    ).collect()[0]
    print(f"Date range: {date_stats['min_date']} to {date_stats['max_date']}")
    
else:
    print("\n❌ No Lotos data to save!")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
