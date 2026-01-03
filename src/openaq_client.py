import requests
import pandas as pd
from datetime import datetime

class OpenAQClient:
    """Client to interact with the OpenAQ API."""
    
    BASE_URL = "https://api.openaq.org/v2"
    
    def __init__(self):
        self.headers = {"Accept": "application/json"}

    def get_city_data(self, city_name):
        """Fetches latest measurements for a specific city."""
        url = f"{self.BASE_URL}/latest"
        params = {
            "city": city_name,
            "limit": 1,
            "order_by": "local",
            "sort": "desc",
            "dump_raw": "false"
        }
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            if not data['results']:
                return None
                
            result = data['results'][0]
            measurements = {m['parameter']: m['value'] for m in result['measurements']}
            
            return {
                'city': city_name,
                'location': result.get('location', 'Unknown'),
                'timestamp_local': result.get('date', {}).get('local', datetime.now().isoformat()),
                'pm25': measurements.get('pm25', None),
                'pm10': measurements.get('pm10', None),
                'no2': measurements.get('no2', None),
                'o3': measurements.get('o3', None),
                'unit': result['measurements'][0]['unit'] if result['measurements'] else 'µg/m³'
            }
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data for {city_name}: {e}")
            return None

if __name__ == "__main__":
    # Test the client
    client = OpenAQClient()
    print("Testing OpenAQ Client for Delhi...")
    data = client.get_city_data("Delhi")
    print(data)
