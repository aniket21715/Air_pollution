"""
Historical Data Collector
Fetches historical air quality data from Open-Meteo API (FREE).

Open-Meteo provides historical data from 2022 onwards.
This script will create a complete dataset combining:
- Kaggle data (2015-2020)
- Open-Meteo historical data (2022-present)

Note: 2020-2022 gap may need interpolation or alternative sources.
The script automatically fetches data up to the current year.
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import os
import time
from indian_aqi_calculator import calculate_indian_aqi


# All 31 Indian cities with coordinates
CITY_COORDS = {
    "Delhi": {"lat": 28.6139, "lon": 77.2090},
    "Ghaziabad": {"lat": 28.6692, "lon": 77.4538},
    "Noida": {"lat": 28.5355, "lon": 77.3910},
    "Faridabad": {"lat": 28.4089, "lon": 77.3178},
    "Gurgaon": {"lat": 28.4595, "lon": 77.0266},
    "Mumbai": {"lat": 19.0760, "lon": 72.8777},
    "Bengaluru": {"lat": 12.9716, "lon": 77.5946},
    "Hyderabad": {"lat": 17.3850, "lon": 78.4867},
    "Chennai": {"lat": 13.0827, "lon": 80.2707},
    "Kolkata": {"lat": 22.5726, "lon": 88.3639},
    "Ahmedabad": {"lat": 23.0225, "lon": 72.5714},
    "Pune": {"lat": 18.5204, "lon": 73.8567},
    "Jaipur": {"lat": 26.9124, "lon": 75.7873},
    "Lucknow": {"lat": 26.8467, "lon": 80.9462},
    "Kanpur": {"lat": 26.4499, "lon": 80.3319},
    "Nagpur": {"lat": 21.1458, "lon": 79.0882},
    "Indore": {"lat": 22.7196, "lon": 75.8577},
    "Patna": {"lat": 25.5941, "lon": 85.1376},
    "Bhopal": {"lat": 23.2599, "lon": 77.4126},
    "Ludhiana": {"lat": 30.9010, "lon": 75.8573},
    "Agra": {"lat": 27.1767, "lon": 78.0081},
    "Varanasi": {"lat": 25.3176, "lon": 82.9739},
    "Meerut": {"lat": 28.9845, "lon": 77.7064},
    "Vadodara": {"lat": 22.3072, "lon": 73.1812},
    "Visakhapatnam": {"lat": 17.6868, "lon": 83.2185},
    "Surat": {"lat": 21.1702, "lon": 72.8311},
    "Chandigarh": {"lat": 30.7333, "lon": 76.7794},
    "Amritsar": {"lat": 31.6340, "lon": 74.8723},
    "Jodhpur": {"lat": 26.2389, "lon": 73.0243},
    "Kochi": {"lat": 9.9312, "lon": 76.2673},
    "Coimbatore": {"lat": 11.0168, "lon": 76.9558},
}


def fetch_historical_data(city_name, start_date, end_date):
    """
    Fetch historical air quality data from Open-Meteo API.
    
    Args:
        city_name: Name of the city
        start_date: Start date (YYYY-MM-DD format)
        end_date: End date (YYYY-MM-DD format)
    
    Returns:
        DataFrame with daily AQI data
    """
    coords = CITY_COORDS.get(city_name)
    if not coords:
        return pd.DataFrame()
    
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    
    params = {
        "latitude": coords['lat'],
        "longitude": coords['lon'],
        "hourly": "pm10,pm2_5,nitrogen_dioxide,ozone,carbon_monoxide,sulphur_dioxide",
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "Asia/Kolkata"
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        data = response.json()
        
        hourly = data.get('hourly', {})
        
        if not hourly.get('time'):
            return pd.DataFrame()
        
        df = pd.DataFrame({
            'datetime': pd.to_datetime(hourly.get('time', [])),
            'pm25': hourly.get('pm2_5', []),
            'pm10': hourly.get('pm10', []),
            'no2': hourly.get('nitrogen_dioxide', []),
            'o3': hourly.get('ozone', []),
            'co': hourly.get('carbon_monoxide', []),
            'so2': hourly.get('sulphur_dioxide', []),
        })
        
        # Aggregate hourly to daily
        df['Date'] = df['datetime'].dt.date
        daily = df.groupby('Date').agg({
            'pm25': 'mean',
            'pm10': 'mean',
            'no2': 'mean',
            'o3': 'mean',
            'co': 'mean',
            'so2': 'mean',
        }).reset_index()
        
        # Calculate Indian AQI for each day
        daily['AQI'] = daily.apply(
            lambda row: calculate_indian_aqi(
                pm25=row['pm25'],
                pm10=row['pm10'],
                no2=row['no2'],
                o3=row['o3']
            ).get('aqi', 0) if calculate_indian_aqi(pm25=row['pm25'], pm10=row['pm10']) else 0,
            axis=1
        )
        
        # Add city name
        daily['City'] = city_name
        
        # Rename columns to match Kaggle format
        daily = daily.rename(columns={
            'pm25': 'PM2.5',
            'pm10': 'PM10',
            'no2': 'NO2',
            'o3': 'O3',
            'co': 'CO',
            'so2': 'SO2',
        })
        
        return daily
        
    except Exception as e:
        print(f"  Error fetching {city_name}: {e}")
        return pd.DataFrame()


def collect_all_historical_data(start_year=2022, end_year=None):
    """
    Collect historical data for all cities from start_year to end_year.
    Open-Meteo historical data is available from 2022 onwards.
    
    Args:
        start_year: Starting year for data collection (default: 2022)
        end_year: Ending year for data collection (default: current year)
    """
    # If end_year is not specified, use the current year
    current_date = datetime.now()
    if end_year is None:
        end_year = current_date.year
    
    # Determine the actual end date (for current year, use today's date)
    if end_year == current_date.year:
        actual_end_date = current_date.strftime('%Y-%m-%d')
        display_end_date = actual_end_date
    else:
        actual_end_date = f"{end_year}-12-31"
        display_end_date = actual_end_date
    
    print("=" * 70)
    print(f"  COLLECTING HISTORICAL AIR QUALITY DATA ({start_year}-{end_year})")
    print("=" * 70)
    print(f"  Date range: {start_year}-01-01 to {display_end_date}")
    print(f"  Cities: {len(CITY_COORDS)}")
    print("=" * 70)
    
    all_data = []
    
    for idx, city_name in enumerate(CITY_COORDS.keys(), 1):
        print(f"\n[{idx}/{len(CITY_COORDS)}] Fetching {city_name}...")
        
        city_data = []
        
        # Fetch year by year to avoid API limits
        for year in range(start_year, end_year + 1):
            start_date = f"{year}-01-01"
            
            # For current year, only fetch up to today; for past years, fetch entire year
            if year == current_date.year:
                end_date = current_date.strftime('%Y-%m-%d')
                print(f"  {year} (up to today)...", end=" ")
            else:
                end_date = f"{year}-12-31"
                print(f"  {year}...", end=" ")
            
            df = fetch_historical_data(city_name, start_date, end_date)
            
            if not df.empty:
                city_data.append(df)
                print(f"✅ {len(df)} days")
            else:
                print("⚠️ No data")
            
            time.sleep(0.5)  # Rate limiting
        
        if city_data:
            city_df = pd.concat(city_data, ignore_index=True)
            all_data.append(city_df)
            print(f"  Total: {len(city_df)} days for {city_name}")
    
    # Combine all city data
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df['Date'] = pd.to_datetime(combined_df['Date'])
        combined_df = combined_df.sort_values(['City', 'Date']).reset_index(drop=True)
        
        print("\n" + "=" * 70)
        print("  COLLECTION COMPLETE")
        print("=" * 70)
        print(f"  Total records: {len(combined_df):,}")
        print(f"  Cities: {combined_df['City'].nunique()}")
        print(f"  Date range: {combined_df['Date'].min()} to {combined_df['Date'].max()}")
        
        return combined_df
    
    return pd.DataFrame()


def merge_with_kaggle(historical_df, kaggle_path="data/raw/india_aqi_kaggle_real.csv"):
    """
    Merge historical API data with Kaggle dataset to create complete dataset.
    """
    print("\n" + "=" * 70)
    print("  MERGING WITH KAGGLE DATA")
    print("=" * 70)
    
    # Load Kaggle data
    if os.path.exists(kaggle_path):
        kaggle_df = pd.read_csv(kaggle_path)
        kaggle_df['Date'] = pd.to_datetime(kaggle_df['Date'])
        
        print(f"  Kaggle data: {len(kaggle_df):,} rows ({kaggle_df['Date'].min()} to {kaggle_df['Date'].max()})")
        print(f"  API data: {len(historical_df):,} rows ({historical_df['Date'].min()} to {historical_df['Date'].max()})")
        
        # Keep only common columns
        common_cols = ['City', 'Date', 'PM2.5', 'PM10', 'NO2', 'AQI']
        
        kaggle_subset = kaggle_df[[c for c in common_cols if c in kaggle_df.columns]].copy()
        api_subset = historical_df[[c for c in common_cols if c in historical_df.columns]].copy()
        
        # Combine (Kaggle first, then API data for later dates)
        kaggle_max_date = kaggle_subset['Date'].max()
        api_subset = api_subset[api_subset['Date'] > kaggle_max_date]
        
        combined = pd.concat([kaggle_subset, api_subset], ignore_index=True)
        combined = combined.sort_values(['City', 'Date']).reset_index(drop=True)
        
        print(f"  Combined: {len(combined):,} rows ({combined['Date'].min()} to {combined['Date'].max()})")
        
        return combined
    else:
        print(f"  Kaggle file not found: {kaggle_path}")
        return historical_df


def save_complete_dataset(df, output_path="data/raw/india_aqi_complete.csv"):
    """Save the complete dataset."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n✅ Complete dataset saved: {output_path}")
    print(f"   Rows: {len(df):,}")
    print(f"   Cities: {df['City'].nunique()}")
    print(f"   Date range: {df['Date'].min()} to {df['Date'].max()}")


if __name__ == "__main__":
    # Collect historical data from Open-Meteo (2022-present)
    # end_year automatically defaults to current year
    historical_df = collect_all_historical_data(start_year=2022)
    
    if not historical_df.empty:
        # Merge with Kaggle data
        complete_df = merge_with_kaggle(historical_df)
        
        # Save complete dataset
        save_complete_dataset(complete_df)
        
        print("\n" + "=" * 70)
        print("  DONE! Complete dataset ready for use")
        print("=" * 70)
    else:
        print("\n❌ No data collected. Check your internet connection.")
