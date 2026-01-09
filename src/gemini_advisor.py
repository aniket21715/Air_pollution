"""
Gemini AI Health Advisor
Provides dynamic, personalized health recommendations using Google's Gemini 2.0 Flash.

Features:
- Context-aware health advice based on current AQI and user profile
- Interactive chat for follow-up questions
- City-specific pollution insights
"""

import google.generativeai as genai
from datetime import datetime


class GeminiHealthAdvisor:
    """AI-powered health advisor using Gemini 2.0 Flash."""
    
    def __init__(self, api_key):
        """Initialize the Gemini client."""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        
    def get_personalized_advice(self, city, aqi, category, pm25, pm10, user_profile, activity=None):
        """
        Get AI-generated personalized health advice.
        
        Args:
            city: City name
            aqi: Current AQI value
            category: AQI category (Good, Moderate, Poor, etc.)
            pm25: PM2.5 concentration
            pm10: PM10 concentration
            user_profile: User's health profile (general, asthma, elderly, etc.)
            activity: Planned activity (optional)
        
        Returns:
            AI-generated health advice string
        """
        prompt = f"""You are an expert health advisor specializing in air quality and its health impacts in India.

CURRENT CONDITIONS:
- City: {city}
- Air Quality Index (AQI): {aqi} ({category})
- PM2.5: {pm25} µg/m³
- PM10: {pm10} µg/m³
- Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}

USER PROFILE:
- Health Category: {user_profile}
{"- Planned Activity: " + activity if activity else ""}

Provide a brief, personalized health recommendation (3-4 sentences max) for this user based on the current air quality. Include:
1. One specific action they should take RIGHT NOW
2. Any precautions based on their health profile
3. When conditions might improve (morning/evening typically better)

Be concise, practical, and specific to Indian context. Use simple language."""

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Unable to generate AI advice: {str(e)}"
    
    def chat(self, user_message, city, aqi, category, user_profile, chat_history=None):
        """
        Interactive chat for follow-up questions about air quality.
        
        Args:
            user_message: User's question
            city: Current city
            aqi: Current AQI
            category: AQI category
            user_profile: User's health profile
            chat_history: Previous conversation context
        
        Returns:
            AI response to the user's question
        """
        context = f"""You are a helpful AI assistant specializing in air quality health advice for India.

CURRENT CONDITIONS:
- City: {city}
- AQI: {aqi} ({category})
- User Profile: {user_profile}

Answer the user's question specifically about air quality, health precautions, or activities. Be helpful, concise, and practical.

User Question: {user_message}"""

        try:
            response = self.model.generate_content(context)
            return response.text
        except Exception as e:
            return f"Error: {str(e)}"
    
    def get_activity_recommendation(self, city, aqi, activity_type, user_profile, forecast_data=None):
        """
        Get AI recommendation for a specific activity.
        
        Args:
            city: City name
            aqi: Current AQI
            activity_type: Type of activity (jogging, cycling, outdoor event, etc.)
            user_profile: User's health profile
            forecast_data: Optional forecast data for timing recommendations
        
        Returns:
            Activity-specific recommendation
        """
        prompt = f"""You are an air quality health advisor for {city}.

Current AQI: {aqi}
User Profile: {user_profile}
Planned Activity: {activity_type}

Provide a SHORT recommendation (2-3 sentences):
1. Should they do this activity today? (Yes/No/With precautions)
2. Best time to do it if applicable
3. Any specific precautions

Be direct and practical."""

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Unable to generate recommendation: {str(e)}"


# Global instance (will be initialized with API key)
_gemini_advisor = None

def get_advisor(api_key=None):
    """Get or create the Gemini advisor instance."""
    global _gemini_advisor
    if _gemini_advisor is None and api_key:
        _gemini_advisor = GeminiHealthAdvisor(api_key)
    return _gemini_advisor


if __name__ == "__main__":
    # Test the advisor
    API_KEY = "AIzaSyA6SzVuFQQDyGx4wlCfnflCdYl1J4JQkH0"
    
    advisor = GeminiHealthAdvisor(API_KEY)
    
    print("=" * 70)
    print("  GEMINI AI HEALTH ADVISOR TEST")
    print("=" * 70)
    
    # Test personalized advice
    advice = advisor.get_personalized_advice(
        city="Delhi",
        aqi=344,
        category="Very Poor",
        pm25=177,
        pm10=220,
        user_profile="asthma",
        activity="jogging"
    )
    
    print("\n📋 AI Health Advice:\n")
    print(advice)
    
    # Test chat
    print("\n" + "-" * 70)
    print("\n💬 Chat Test:\n")
    
    response = advisor.chat(
        user_message="Is it safe for my 5-year-old child to play outside today?",
        city="Delhi",
        aqi=344,
        category="Very Poor",
        user_profile="children"
    )
    
    print(response)
