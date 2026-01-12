from google import genai
import requests
from datetime import datetime
import os
import sys

# Add parent directory to path to import sibling modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openmeteo_client import OpenMeteoAQIClient

class PollutionDetectiveAgent:
    """
    The Scientist Agent.
    Role: Objective analysis of pollution data and weather conditions.
    Goal: Determine the state and cause of pollution.
    """
    
    WEATHER_API_URL = "https://api.open-meteo.com/v1/forecast"
    
    def __init__(self, api_key):
        # New SDK: Use Client object instead of genai.configure
        self.client = genai.Client(api_key=api_key)
        self.model_name = 'gemini-2.5-flash-lite'
        self.aqi_client = OpenMeteoAQIClient()
        
    def _get_weather_conditions(self, lat, lon):
        """Fetch current weather conditions from Open-Meteo."""
        try:
            params = {
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m",
                "timezone": "Asia/Kolkata"
            }
            response = requests.get(self.WEATHER_API_URL, params=params, timeout=10)
            data = response.json()
            return data.get('current', {})
        except Exception as e:
            print(f"Weather fetch error: {e}")
            return {}

    def analyze_city(self, city_name):
        """
        Conduct a full analysis of the city's air quality.
        1. Fetch AQI
        2. Fetch Weather (Wind, Temp, Humidity)
        3. Reason about the cause
        """
        # 1. Fetch Data
        aqi_data = self.aqi_client.get_current_aqi(city_name)
        if not aqi_data:
            return {"error": f"Could not fetch data for {city_name}"}
            
        weather = self._get_weather_conditions(aqi_data['lat'], aqi_data['lon'])
        
        # 2. Construct Analysis Request
        prompt = f"""You are the 'Pollution Detective', a scientist agent.
        
        DATA:
        City: {city_name}
        AQI: {aqi_data['aqi']} ({aqi_data['category']})
        PM2.5: {aqi_data['pm25']}
        PM10: {aqi_data['pm10']}
        
        WEATHER:
        Wind Speed: {weather.get('wind_speed_10m', 'N/A')} km/h
        Wind Direction: {weather.get('wind_direction_10m', 'N/A')}°
        Temperature: {weather.get('temperature_2m', 'N/A')}°C
        Humidity: {weather.get('relative_humidity_2m', 'N/A')}%
        
        TASK:
        Analyze WHY the air quality is {aqi_data['category']}.
        - If Bad (>200): Is low wind speed trapping pollutants? Is direction bringing smoke (e.g. from stubble areas)?
        - If Good (<100): Is good wind ventilation helping? Rain?
        
        OUTPUT:
        Produce a concise 'Situation Report' (3-4 sentences). Focus on CAUSE and EFFECT.
        Do not give health advice (that's the Guardian's job). Just analyze the physics."""
        
        # 3. Agent Reasoning (New SDK pattern)
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            report = response.text
            
            return {
                "city": city_name,
                "aqi_data": aqi_data,
                "weather_data": weather,
                "situation_report": report,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    key = os.getenv("GEMINI_API_KEY")
    agent = PollutionDetectiveAgent(key)
    print(agent.analyze_city("Delhi"))
