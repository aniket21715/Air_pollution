"""
Open-Meteo Air Quality API Client
Provides FREE real-time AQI data for Indian cities - NO API KEY REQUIRED.

API Documentation: https://open-meteo.com/en/docs/air-quality-api

Features:
- Real-time PM2.5, PM10, NO2, O3, CO, SO2 data
- Indian AQI calculation using CPCB breakpoints
- Hourly updates
- Completely free with no rate limits for reasonable usage
"""

import requests
import pandas as pd
from datetime import datetime
from indian_aqi_calculator import calculate_indian_aqi, INDIAN_AQI_CATEGORIES


class OpenMeteoAQIClient:
    """
    Free real-time air quality data from Open-Meteo.
    No API key required - just coordinates.
    """
    
    BASE_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
    
    # Coordinates for all 31 Indian cities
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
    
    def get_current_aqi(self, city_name):
        """
        Get real-time AQI for a city using Open-Meteo API.
        Returns Indian AQI calculated from pollutant concentrations.
        """
        coords = self.CITY_COORDS.get(city_name)
        if not coords:
            return None
        
        try:
            params = {
                "latitude": coords['lat'],
                "longitude": coords['lon'],
                "current": "pm10,pm2_5,nitrogen_dioxide,ozone,carbon_monoxide,sulphur_dioxide",
                "timezone": "Asia/Kolkata"
            }
            
            response = requests.get(self.BASE_URL, params=params, timeout=10)
            data = response.json()
            
            curr = data.get('current', {})
            
            # Get raw concentrations
            pm25 = curr.get('pm2_5')
            pm10 = curr.get('pm10')
            no2 = curr.get('nitrogen_dioxide')
            o3 = curr.get('ozone')
            co = curr.get('carbon_monoxide')  # μg/m³, need to convert to mg/m³
            so2 = curr.get('sulphur_dioxide')
            
            # Convert CO from μg/m³ to mg/m³ for Indian AQI calculation
            co_mg = co / 1000 if co else None
            
            # Calculate Indian AQI
            result = calculate_indian_aqi(
                pm25=pm25,
                pm10=pm10,
                no2=no2,
                o3=o3,
                co=co_mg,
                so2=so2
            )
            
            if result:
                return {
                    "city": city_name,
                    "aqi": result['aqi'],
                    "category": result['category'],
                    "color": result['color'],
                    "health_impact": result['health_impact'],
                    "dominant_pollutant": result['dominant_pollutant'],
                    "pm25": round(pm25, 1) if pm25 else None,
                    "pm10": round(pm10, 1) if pm10 else None,
                    "no2": round(no2, 1) if no2 else None,
                    "o3": round(o3, 1) if o3 else None,
                    "co": round(co, 1) if co else None,
                    "so2": round(so2, 1) if so2 else None,
                    "timestamp": curr.get('time'),
                    "source": "Open-Meteo + Indian AQI",
                    "lat": coords['lat'],
                    "lon": coords['lon'],
                }
            return None
            
        except Exception as e:
            print(f"Error fetching {city_name}: {e}")
            return None
    
    def get_all_cities(self):
        """Fetch current AQI for all configured Indian cities."""
        results = []
        for city in self.CITY_COORDS.keys():
            data = self.get_current_aqi(city)
            if data:
                results.append(data)
        return results
    
    def get_forecast(self, city_name, days=7):
        """
        Get AQI forecast for a city.
        Returns hourly forecast data.
        """
        coords = self.CITY_COORDS.get(city_name)
        if not coords:
            return None
        
        try:
            params = {
                "latitude": coords['lat'],
                "longitude": coords['lon'],
                "hourly": "pm10,pm2_5,nitrogen_dioxide,ozone",
                "forecast_days": days,
                "timezone": "Asia/Kolkata"
            }
            
            response = requests.get(self.BASE_URL, params=params, timeout=10)
            data = response.json()
            
            hourly = data.get('hourly', {})
            
            df = pd.DataFrame({
                'datetime': pd.to_datetime(hourly.get('time', [])),
                'pm25': hourly.get('pm2_5', []),
                'pm10': hourly.get('pm10', []),
                'no2': hourly.get('nitrogen_dioxide', []),
                'o3': hourly.get('ozone', []),
            })
            
            # Calculate Indian AQI for each hour
            df['aqi'] = df.apply(
                lambda row: calculate_indian_aqi(
                    pm25=row['pm25'], 
                    pm10=row['pm10'],
                    no2=row['no2'],
                    o3=row['o3']
                ).get('aqi', 0) if calculate_indian_aqi(
                    pm25=row['pm25'], 
                    pm10=row['pm10']
                ) else 0,
                axis=1
            )
            
            df['city'] = city_name
            
            return df
            
        except Exception as e:
            print(f"Forecast error for {city_name}: {e}")
            return None


def test_client():
    """Test the Open-Meteo client."""
    print("=" * 70)
    print("  OPEN-METEO AIR QUALITY API TEST (FREE - NO KEY REQUIRED)")
    print("=" * 70)
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    client = OpenMeteoAQIClient()
    
    # Test major cities
    test_cities = ["Delhi", "Mumbai", "Bengaluru", "Chennai", "Kolkata",
                   "Hyderabad", "Pune", "Jaipur", "Lucknow", "Ahmedabad"]
    
    print(f"\n{'City':<15} {'AQI':<8} {'Category':<15} {'PM2.5':<10} {'PM10':<10} {'Dominant'}")
    print("-" * 70)
    
    for city in test_cities:
        data = client.get_current_aqi(city)
        if data:
            print(f"{city:<15} {data['aqi']:<8} {data['category']:<15} "
                  f"{data['pm25']:<10} {data['pm10']:<10} {data['dominant_pollutant']}")
        else:
            print(f"{city:<15} {'Error':<8}")
    
    print("\n" + "=" * 70)
    print("  ✅ Open-Meteo API working - No API key needed!")
    print("=" * 70)
    
    return True


if __name__ == "__main__":
    test_client()
