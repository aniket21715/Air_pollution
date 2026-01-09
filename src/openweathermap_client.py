"""
OpenWeatherMap Air Quality API Client
Provides accurate real-time AQI data for Indian cities.

API Documentation: https://openweathermap.org/api/air-pollution

Features:
- Current AQI with all pollutant concentrations
- Forecast data (up to 4 days)
- Historical data
- 1,000,000 calls/month free tier
- Indian AQI calculation using CPCB breakpoints
"""

import requests
import pandas as pd
from datetime import datetime
from indian_aqi_calculator import calculate_indian_aqi, INDIAN_AQI_CATEGORIES


class OpenWeatherMapAQIClient:
    """
    OpenWeatherMap Air Pollution API client.
    Provides accurate real-time AQI data with proper pollutant concentrations.
    """
    
    BASE_URL = "https://api.openweathermap.org/data/2.5/air_pollution"
    
    # Coordinates for all 31+ Indian cities
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
    
    def __init__(self, api_key):
        """
        Initialize OpenWeatherMap client.
        
        api_key: Get free API key at https://openweathermap.org/api
        """
        self.api_key = api_key
    
    def get_current_aqi(self, city_name):
        """
        Get real-time AQI for a city.
        Returns Indian AQI calculated from actual pollutant concentrations.
        """
        coords = self.CITY_COORDS.get(city_name)
        if not coords:
            return None
        
        try:
            params = {
                "lat": coords['lat'],
                "lon": coords['lon'],
                "appid": self.api_key
            }
            
            response = requests.get(self.BASE_URL, params=params, timeout=10)
            data = response.json()
            
            if response.status_code != 200:
                print(f"API Error: {data.get('message', 'Unknown error')}")
                return None
            
            # Extract pollutant data
            components = data['list'][0]['components']
            
            # OpenWeatherMap provides concentrations in µg/m³
            pm25 = components.get('pm2_5', 0)
            pm10 = components.get('pm10', 0)
            no2 = components.get('no2', 0)
            o3 = components.get('o3', 0)
            co = components.get('co', 0) / 1000  # Convert to mg/m³
            so2 = components.get('so2', 0)
            nh3 = components.get('nh3', 0)
            no = components.get('no', 0)
            
            # Calculate Indian AQI from concentrations
            result = calculate_indian_aqi(
                pm25=pm25,
                pm10=pm10,
                no2=no2,
                o3=o3,
                co=co,
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
                    "pm25": round(pm25, 1),
                    "pm10": round(pm10, 1),
                    "no2": round(no2, 1),
                    "o3": round(o3, 1),
                    "co": round(co * 1000, 1),  # Back to µg/m³ for display
                    "so2": round(so2, 1),
                    "nh3": round(nh3, 1),
                    "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M"),
                    "source": "OpenWeatherMap + Indian AQI",
                    "lat": coords['lat'],
                    "lon": coords['lon'],
                    "owm_aqi": data['list'][0]['main']['aqi'],  # OWM's own AQI (1-5 scale)
                }
            return None
            
        except Exception as e:
            print(f"Error fetching {city_name}: {e}")
            return None
    
    def get_forecast(self, city_name, hours=96):
        """
        Get AQI forecast for a city (up to 4 days / 96 hours).
        
        Returns DataFrame with hourly forecast data.
        """
        coords = self.CITY_COORDS.get(city_name)
        if not coords:
            return None
        
        try:
            params = {
                "lat": coords['lat'],
                "lon": coords['lon'],
                "appid": self.api_key
            }
            
            response = requests.get(f"{self.BASE_URL}/forecast", params=params, timeout=10)
            data = response.json()
            
            if response.status_code != 200:
                return None
            
            forecast_data = []
            for item in data['list'][:hours]:
                components = item['components']
                pm25 = components.get('pm2_5', 0)
                pm10 = components.get('pm10', 0)
                
                # Calculate Indian AQI
                result = calculate_indian_aqi(pm25=pm25, pm10=pm10)
                aqi = result['aqi'] if result else 0
                
                forecast_data.append({
                    'datetime': datetime.fromtimestamp(item['dt']),
                    'aqi': aqi,
                    'pm25': pm25,
                    'pm10': pm10,
                    'no2': components.get('no2', 0),
                    'o3': components.get('o3', 0),
                })
            
            df = pd.DataFrame(forecast_data)
            df['city'] = city_name
            
            return df
            
        except Exception as e:
            print(f"Forecast error for {city_name}: {e}")
            return None
    
    def get_all_cities(self):
        """Fetch current AQI for all configured Indian cities."""
        results = []
        for city in self.CITY_COORDS.keys():
            print(f"Fetching {city}...", end=" ")
            data = self.get_current_aqi(city)
            if data:
                print(f"✅ AQI: {data['aqi']} ({data['category']})")
                results.append(data)
            else:
                print("❌")
        return results


def test_client(api_key):
    """Test the OpenWeatherMap client."""
    print("=" * 70)
    print("  OPENWEATHERMAP AIR QUALITY TEST")
    print("=" * 70)
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    client = OpenWeatherMapAQIClient(api_key)
    
    # Test major cities
    test_cities = ["Delhi", "Mumbai", "Bengaluru", "Chennai", "Kolkata"]
    
    print(f"\n{'City':<15} {'AQI (IN)':<10} {'Category':<15} {'PM2.5':<10} {'PM10':<10} {'Dominant'}")
    print("-" * 70)
    
    for city in test_cities:
        data = client.get_current_aqi(city)
        if data:
            print(f"{city:<15} {data['aqi']:<10} {data['category']:<15} "
                  f"{data['pm25']:<10} {data['pm10']:<10} {data['dominant_pollutant']}")
        else:
            print(f"{city:<15} {'Error':<10}")
    
    print("\n" + "=" * 70)
    print("  Values are calculated using Indian CPCB AQI breakpoints")
    print("=" * 70)


if __name__ == "__main__":
    # Test with API key
    API_KEY = input("Enter your OpenWeatherMap API key: ").strip()
    if API_KEY:
        test_client(API_KEY)
    else:
        print("No API key provided. Get one free at https://openweathermap.org/api")
