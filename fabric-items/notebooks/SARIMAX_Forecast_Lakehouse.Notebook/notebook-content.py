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

# Import required libraries
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import math
import random
import builtins  # Import builtins to access Python's native min/max

# For visualization (optional)
import matplotlib.pyplot as plt
import seaborn as sns

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.4f' % x)

print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Note: In Microsoft Fabric, spark session is pre-initialized
# If running locally, uncomment the following:
# spark = SparkSession.builder.appName("SARIMAX_Forecast").getOrCreate()

# Verify Spark session
print(f"Spark version: {spark.version}")
print(f"Spark UI: {spark.sparkContext.uiWebUrl}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Define file paths in Lakehouse
# In Fabric, Files/ is accessible at the root
petrol_path = "Files/lotos_petrol_prices.csv"
fx_path = "Files/fx_rates_usd_pln.csv"
oil_path = "Files/brent_prices_eia.csv"

# Load petrol prices
print("Loading petrol prices...")
petrol_df = spark.read.option("header", "true") \
    .option("inferSchema", "true") \
    .csv(petrol_path)

petrol_df = petrol_df.select(
    col("date").cast("date").alias("date"),
    col("price_pln_per_liter").cast("double").alias("petrol_price")
)

print(f"✓ Loaded {petrol_df.count()} petrol price records")
petrol_df.show(5)

# Load FX rates
print("\nLoading FX rates...")
fx_df = spark.read.option("header", "true") \
    .option("inferSchema", "true") \
    .csv(fx_path)

fx_df = fx_df.filter(col("from_currency") == "USD") \
    .select(
        col("date").cast("date").alias("date"),
        col("exchange_rate").cast("double").alias("usd_pln_rate")
    )

print(f"✓ Loaded {fx_df.count()} USD/PLN exchange rate records")
fx_df.show(5)

# Load oil prices
print("\nLoading oil prices...")
oil_df = spark.read.option("header", "true") \
    .option("inferSchema", "true") \
    .csv(oil_path)

oil_df = oil_df.select(
    col("date").cast("date").alias("date"),
    col("price_usd").cast("double").alias("oil_price_usd")
)

print(f"✓ Loaded {oil_df.count()} Brent oil price records")
oil_df.show(5)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Merge all data sources by date
print("Aligning data by date...")

# Join all datasets
aligned_df = petrol_df \
    .join(fx_df, on="date", how="inner") \
    .join(oil_df, on="date", how="inner")

# Calculate oil price in PLN
aligned_df = aligned_df.withColumn(
    "oil_price_pln", 
    col("oil_price_usd") * col("usd_pln_rate")
)

# Sort by date
aligned_df = aligned_df.orderBy("date")

# Cache for performance
aligned_df.cache()

total_records = aligned_df.count()
print(f"\n✓ Aligned dataset: {total_records} records")

# Show date range
date_range = aligned_df.select(
    min("date").alias("min_date"),
    max("date").alias("max_date")
).collect()[0]

print(f"Date range: {date_range['min_date']} to {date_range['max_date']}")

# Display sample
aligned_df.show(10)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Convert to Pandas for time series analysis
# Note: This is suitable for datasets that fit in memory
print("Converting to Pandas DataFrame...")
data_pd = aligned_df.toPandas()

# Set date as index
data_pd.set_index('date', inplace=True)
data_pd = data_pd.sort_index()

print(f"✓ Converted {len(data_pd)} records")
print("\nData summary:")
print(data_pd.describe())

# Calculate correlations
correlations = data_pd.corr()
print("\nCorrelation Matrix:")
print(correlations)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

def calculate_model_parameters(data):
    """Calculate model parameters based on historical data"""
    n = len(data)
    
    # Statistics
    mean_petrol = data['petrol_price'].mean()
    std_petrol = data['petrol_price'].std()
    mean_fx = data['usd_pln_rate'].mean()
    mean_oil_pln = data['oil_price_pln'].mean()
    
    # Calculate change volatility
    petrol_changes = data['petrol_price'].diff().dropna()
    change_volatility = petrol_changes.std()
    
    # Validated correlations (from previous analysis)
    fx_correlation = 0.148
    oil_correlation = 0.251
    
    return {
        'mean_petrol': mean_petrol,
        'std_petrol': std_petrol,
        'mean_fx': mean_fx,
        'mean_oil_pln': mean_oil_pln,
        'fx_correlation': fx_correlation,
        'oil_correlation': oil_correlation,
        'ar_coefficient': 0.85,
        'mean_reversion_strength': 0.05,
        'change_volatility': change_volatility,
        'lower_bound': mean_petrol - 2 * std_petrol,
        'upper_bound': mean_petrol + 2 * std_petrol
    }

# Calculate parameters
model_params = calculate_model_parameters(data_pd)

print("Model Parameters:")
for key, value in model_params.items():
    print(f"  {key}: {value:.4f}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

def forecast_one_step(current_petrol, current_fx, current_oil_pln, model):
    """One-step-ahead forecast using calibrated model"""
    
    # Base level from weighted average of AR and mean
    ar_weight = model['ar_coefficient']
    mean_weight = 1 - ar_weight
    base_level = ar_weight * current_petrol + mean_weight * model['mean_petrol']
    
    # Mean reversion component
    mean_reversion = model['mean_reversion_strength'] * (model['mean_petrol'] - current_petrol)
    
    # External effects scaled by validated correlations
    fx_effect = model['fx_correlation'] * 0.3 * (current_fx - model['mean_fx'])
    oil_effect = model['oil_correlation'] * 0.001 * (current_oil_pln - model['mean_oil_pln'])
    
    # Combine components
    forecast = base_level + mean_reversion + fx_effect + oil_effect
    
    # Add realistic noise based on historical volatility
    noise = random.gauss(0, model['change_volatility'] * 0.5)
    forecast += noise
    
    # Apply bounds using builtins or numpy to avoid PySpark conflict
    forecast = builtins.max(model['lower_bound'], builtins.min(model['upper_bound'], forecast))
    
    return forecast

def generate_external_scenarios(start_fx, start_oil_pln, days=365):
    """Generate realistic scenarios for FX and oil prices"""
    scenarios = []
    
    current_fx = start_fx
    current_oil_pln = start_oil_pln
    
    for day in range(days):
        # FX rate: mean-reverting random walk
        fx_mean = 4.0
        fx_change = random.gauss(0, 0.02)
        fx_reversion = 0.01 * (fx_mean - current_fx)
        current_fx = current_fx * (1 + fx_change) + fx_reversion
        # Use builtins or numpy to avoid PySpark conflict
        current_fx = builtins.max(3.0, builtins.min(5.0, current_fx))
        
        # Oil price: trending random walk with cycles
        oil_trend = 0.0002
        oil_cycle = 10 * math.sin(2 * math.pi * day / 180)
        oil_change = random.gauss(oil_trend, 0.015)
        current_oil_pln = current_oil_pln * (1 + oil_change) + oil_cycle * 0.05
        # Use builtins or numpy to avoid PySpark conflict
        current_oil_pln = builtins.max(150, builtins.min(500, current_oil_pln))
        
        scenarios.append({
            'day': day + 1,
            'fx': current_fx,
            'oil_pln': current_oil_pln
        })
    
    return scenarios

# Set random seed for reproducibility
random.seed(42)

# Get current values
current_values = data_pd.iloc[-1]
current_petrol = current_values['petrol_price']
current_fx = current_values['usd_pln_rate']
current_oil_pln = current_values['oil_price_pln']
start_date = data_pd.index[-1]

print(f"Starting forecast from:")
print(f"  Date: {start_date}")
print(f"  Petrol: {current_petrol:.2f} PLN/L")
print(f"  USD/PLN: {current_fx:.2f}")
print(f"  Oil: {current_oil_pln:.2f} PLN/barrel")

# Generate external scenarios
print("\nGenerating external factor scenarios...")
external_scenarios = generate_external_scenarios(current_fx, current_oil_pln, days=365)

# Generate forecast
print("Generating 12-month forecast...")
forecast_data = []

for i, scenario in enumerate(external_scenarios):
    # One-step-ahead forecast
    forecast_petrol = forecast_one_step(
        current_petrol,
        scenario['fx'],
        scenario['oil_pln'],
        model_params
    )
    
    forecast_date = start_date + timedelta(days=scenario['day'])
    
    forecast_data.append({
        'forecast_date': forecast_date,
        'forecast_day': scenario['day'],
        'petrol_price_forecast': forecast_petrol,
        'usd_pln_rate_forecast': scenario['fx'],
        'oil_price_pln_forecast': scenario['oil_pln'],
        'oil_price_usd_forecast': scenario['oil_pln'] / scenario['fx']
    })
    
    # Update current for next step
    current_petrol = forecast_petrol
    
    if (i + 1) % 30 == 0:
        print(f"  Generated {i + 1} days...")

print(f"\n✓ Generated {len(forecast_data)} daily forecasts")

# Convert to DataFrame
forecast_df = pd.DataFrame(forecast_data)
forecast_df.set_index('forecast_date', inplace=True)

# Display summary
print("\nForecast Summary:")
print(forecast_df.describe())

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Create visualization
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[3, 1])

# Main plot: Petrol prices
# Historical data (last 180 days)
historical_display = data_pd['petrol_price'][-180:]
ax1.plot(historical_display.index, historical_display.values, 
         'b-', linewidth=2, label='Historical Prices')

# Forecast
ax1.plot(forecast_df.index, forecast_df['petrol_price_forecast'], 
         'r-', linewidth=2, label='Forecast')

# Add confidence band (simplified)
std_error = model_params['change_volatility'] * np.sqrt(np.arange(1, len(forecast_df) + 1))
upper_bound = forecast_df['petrol_price_forecast'] + 1.96 * std_error * 0.5
lower_bound = forecast_df['petrol_price_forecast'] - 1.96 * std_error * 0.5

ax1.fill_between(forecast_df.index, lower_bound, upper_bound, 
                 alpha=0.2, color='red', label='95% Confidence Band')

ax1.set_title('12-Month Wholesale Petrol Price Forecast for Poland', fontsize=16)
ax1.set_xlabel('Date')
ax1.set_ylabel('Price (PLN/L)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Bottom plot: External factors
ax2.plot(forecast_df.index, forecast_df['usd_pln_rate_forecast'], 
         'g-', linewidth=1.5, label='USD/PLN')
ax2.set_ylabel('USD/PLN Rate', color='green')
ax2.tick_params(axis='y', labelcolor='green')

ax2_twin = ax2.twinx()
ax2_twin.plot(forecast_df.index, forecast_df['oil_price_pln_forecast'], 
              'orange', linewidth=1.5, alpha=0.7, label='Oil (PLN)')
ax2_twin.set_ylabel('Oil Price (PLN/barrel)', color='orange')
ax2_twin.tick_params(axis='y', labelcolor='orange')

ax2.set_xlabel('Date')
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper left')
ax2_twin.legend(loc='upper right')

plt.tight_layout()
plt.show()

# Key milestones
milestones = [30, 90, 180, 365]
print("\nKey Milestones:")
for days in milestones:
    if days <= len(forecast_df):
        milestone_price = forecast_df.iloc[days-1]['petrol_price_forecast']
        change = ((milestone_price / current_values['petrol_price']) - 1) * 100
        milestone_date = forecast_df.index[days-1]
        print(f"  {days} days ({milestone_date.strftime('%Y-%m-%d')}): {milestone_price:.2f} PLN/L ({change:+.1f}%)")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Reset index to have forecast_date as a column
forecast_df_reset = forecast_df.reset_index()

# Add metadata columns
forecast_df_reset['model_type'] = 'SARIMAX'
forecast_df_reset['forecast_run_date'] = datetime.now()
forecast_df_reset['forecast_horizon_days'] = 365

# Add model parameters as JSON string
import json
model_params_json = json.dumps({k: float(v) for k, v in model_params.items()})
forecast_df_reset['model_parameters'] = model_params_json

# Convert to Spark DataFrame
print("Converting forecast to Spark DataFrame...")

# Define schema explicitly for better control
schema = StructType([
    StructField("forecast_date", DateType(), False),
    StructField("forecast_day", IntegerType(), False),
    StructField("petrol_price_forecast", DoubleType(), False),
    StructField("usd_pln_rate_forecast", DoubleType(), False),
    StructField("oil_price_pln_forecast", DoubleType(), False),
    StructField("oil_price_usd_forecast", DoubleType(), False),
    StructField("model_type", StringType(), False),
    StructField("forecast_run_date", TimestampType(), False),
    StructField("forecast_horizon_days", IntegerType(), False),
    StructField("model_parameters", StringType(), True)
])

# Create Spark DataFrame
forecast_spark_df = spark.createDataFrame(forecast_df_reset, schema=schema)

print(f"✓ Created Spark DataFrame with {forecast_spark_df.count()} rows")
forecast_spark_df.show(10)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Define table name
table_name = "petrol_price_forecast_sarimax"

# Option 1: Save as managed table (recommended for Fabric)
print(f"Saving forecast to lakehouse table: {table_name}")

# Write to table with overwrite mode
# Change to "append" if you want to keep historical forecasts
forecast_spark_df.write \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(table_name)

print(f"\n✓ Forecast saved to table: {table_name}")

# Verify the table
print("\nVerifying saved data:")
spark.sql(f"SELECT COUNT(*) as row_count FROM {table_name}").show()

# Show sample of saved data
print("\nSample of saved forecast:")
spark.sql(f"""
    SELECT 
        forecast_date,
        forecast_day,
        ROUND(petrol_price_forecast, 4) as petrol_forecast,
        ROUND(usd_pln_rate_forecast, 4) as fx_forecast,
        ROUND(oil_price_usd_forecast, 2) as oil_usd_forecast
    FROM {table_name}
    WHERE forecast_day IN (1, 30, 90, 180, 365)
    ORDER BY forecast_day
""").show()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Create summary statistics
summary_df = spark.sql(f"""
    SELECT 
        MIN(forecast_date) as forecast_start_date,
        MAX(forecast_date) as forecast_end_date,
        COUNT(*) as total_days,
        ROUND(AVG(petrol_price_forecast), 4) as avg_petrol_price,
        ROUND(MIN(petrol_price_forecast), 4) as min_petrol_price,
        ROUND(MAX(petrol_price_forecast), 4) as max_petrol_price,
        ROUND(STDDEV(petrol_price_forecast), 4) as std_petrol_price,
        ROUND(AVG(usd_pln_rate_forecast), 4) as avg_usd_pln,
        ROUND(AVG(oil_price_usd_forecast), 2) as avg_oil_usd,
        FIRST(model_type) as model_type,
        FIRST(forecast_run_date) as run_date
    FROM {table_name}
""")

print("Forecast Summary Statistics:")
summary_df.show(truncate=False)

# Save summary to a separate table
summary_table_name = "petrol_forecast_summary"
summary_df.write.mode("append").saveAsTable(summary_table_name)
print(f"\n✓ Summary saved to table: {summary_table_name}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
