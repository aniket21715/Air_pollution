import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# Configuration
CITIES = [
    "Delhi", "Mumbai", "Bengaluru", 
    "New York", "London", "Paris", 
    "Tokyo", "Beijing", "Sydney", "Sao Paulo"
]
START_DATE = "2020-01-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")

def generate_city_data(city_name, start_date, end_date):
    """Generates synthetic AQI data with seasonality and trends."""
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    n_days = len(date_range)
    
    # Base pollution levels (randomized per city for variety)
    base_pm25 = np.random.uniform(20, 100)
    base_pm10 = base_pm25 * np.random.uniform(1.2, 2.0)
    base_no2 = np.random.uniform(10, 50)
    
    # Seasonality (Sin wave to simulate winter smog vs summer clear sky)
    # Winter (around day 0 and 365) has higher pollution
    time_idx = np.arange(n_days)
    seasonality = np.cos(2 * np.pi * time_idx / 365) # +1 in winter, -1 in summer
    
    # Random Walk (Trend)
    trend = np.cumsum(np.random.normal(0, 1, n_days))
    
    # Generate features
    pm25 = base_pm25 + (20 * seasonality) + trend + np.random.normal(0, 10, n_days)
    pm10 = base_pm10 + (30 * seasonality) + trend + np.random.normal(0, 15, n_days)
    no2 = base_no2 + (5 * seasonality) + (0.5 * trend) + np.random.normal(0, 5, n_days)
    
    # Clip negative values
    pm25 = np.maximum(pm25, 5)
    pm10 = np.maximum(pm10, 10)
    no2 = np.maximum(no2, 2)
    
    df = pd.DataFrame({
        'Date': date_range,
        'City': city_name,
        'PM2.5': pm25.round(2),
        'PM10': pm10.round(2),
        'NO2': no2.round(2),
        'AQI': (pm25 * 1.5).round(0) # Simplified AQI calculation
    })
    
    return df

def main():
    print(f"Generating data from {START_DATE} to {END_DATE}...")
    
    all_data = []
    for city in CITIES:
        print(f"Generating data for {city}...")
        city_df = generate_city_data(city, START_DATE, END_DATE)
        all_data.append(city_df)
    
    final_df = pd.concat(all_data, ignore_index=True)
    
    output_dir = os.path.join("data", "raw")
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "global_air_quality_history.csv")
    final_df.to_csv(output_path, index=False)
    
    print(f"Success! Saved {len(final_df)} rows to {output_path}")

if __name__ == "__main__":
    main()
