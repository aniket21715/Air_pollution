"""
Gemini AI Health Advisor
Provides dynamic, personalized health recommendations using Google's Gemini 2.5 Flash.

Features:
- Context-aware health advice based on current AQI and user profile
- Interactive chat for follow-up questions
- City-specific pollution insights
"""

from google import genai
from datetime import datetime


class GeminiHealthAdvisor:
    """AI-powered health advisor using Gemini 2.5 Flash."""
    
    def __init__(self, api_key):
        """Initialize the Gemini client."""
        # New SDK: Use Client object
        self.client = genai.Client(api_key=api_key)
        self.model_name = 'gemini-2.5-flash-lite'
        
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
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
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
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=context
            )
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
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            return response.text
        except Exception as e:
            return f"Unable to generate recommendation: {str(e)}"
    
    def get_dynamic_activity_suggestions(self, city, aqi, category, user_profile, current_hour=None):
        """
        Generate dynamic activity suggestions based on current AQI.
        Returns structured suggestions with icons and safety ratings.
        
        Args:
            city: City name
            aqi: Current AQI value
            category: AQI category
            user_profile: User's health profile
            current_hour: Current hour of day (0-23)
        
        Returns:
            Dictionary with activity suggestions
        """
        if current_hour is None:
            current_hour = datetime.now().hour
        
        time_of_day = "morning" if current_hour < 12 else "afternoon" if current_hour < 17 else "evening"
        
        prompt = f"""You are an air quality advisor for {city}. Current AQI is {aqi} ({category}).
User profile: {user_profile}. Time: {time_of_day}.

Generate EXACTLY 4 activity suggestions in this JSON format (no markdown, just pure JSON):
{{
  "suggestions": [
    {{"activity": "Activity Name", "icon": "emoji", "safety": "safe/caution/avoid", "tip": "One short tip"}},
    {{"activity": "Activity Name", "icon": "emoji", "safety": "safe/caution/avoid", "tip": "One short tip"}},
    {{"activity": "Activity Name", "icon": "emoji", "safety": "safe/caution/avoid", "tip": "One short tip"}},
    {{"activity": "Activity Name", "icon": "emoji", "safety": "safe/caution/avoid", "tip": "One short tip"}}
  ],
  "best_time": "Best time for outdoor activities today",
  "general_tip": "One general tip for the day"
}}

Include a mix of indoor and outdoor activities appropriate for the AQI level.
For poor AQI, suggest more indoor alternatives. Be practical and India-specific."""

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            import json
            # Clean up response - remove markdown code blocks if present
            text = response.text.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            return json.loads(text.strip())
        except Exception as e:
            # Return fallback suggestions
            return self._get_fallback_suggestions(aqi)


    def _get_fallback_suggestions(self, aqi):
        """Return fallback suggestions when API fails."""
        if aqi <= 100:
            return {
                "suggestions": [
                    {"activity": "Morning Jog", "icon": "🏃", "safety": "safe", "tip": "Great conditions for outdoor exercise"},
                    {"activity": "Outdoor Yoga", "icon": "🧘", "safety": "safe", "tip": "Perfect for breathing exercises"},
                    {"activity": "Cycling", "icon": "🚴", "safety": "safe", "tip": "Enjoy the fresh air"},
                    {"activity": "Park Picnic", "icon": "🧺", "safety": "safe", "tip": "Ideal weather for outdoor activities"}
                ],
                "best_time": "Any time today is suitable",
                "general_tip": "Excellent air quality - enjoy outdoor activities!"
            }
        elif aqi <= 200:
            return {
                "suggestions": [
                    {"activity": "Indoor Gym", "icon": "🏋️", "safety": "safe", "tip": "Better to exercise indoors"},
                    {"activity": "Short Walk", "icon": "🚶", "safety": "caution", "tip": "Limit to 30 minutes"},
                    {"activity": "Work from Home", "icon": "💻", "safety": "safe", "tip": "Reduce commute exposure"},
                    {"activity": "Indoor Games", "icon": "🎮", "safety": "safe", "tip": "Keep windows closed"}
                ],
                "best_time": "Early morning (6-8 AM) or late evening",
                "general_tip": "Moderate air quality - limit outdoor exposure"
            }
        else:
            return {
                "suggestions": [
                    {"activity": "Stay Indoors", "icon": "🏠", "safety": "safe", "tip": "Keep windows and doors closed"},
                    {"activity": "Use Air Purifier", "icon": "🌬️", "safety": "safe", "tip": "Run purifier on high"},
                    {"activity": "Indoor Exercise", "icon": "🏋️", "safety": "safe", "tip": "Do yoga or stretching at home"},
                    {"activity": "Wear N95 Mask", "icon": "😷", "safety": "caution", "tip": "Essential if going outside"}
                ],
                "best_time": "Avoid outdoor activities until AQI improves",
                "general_tip": "Poor air quality - prioritize indoor activities and use masks outdoors"
            }


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
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    API_KEY = os.getenv("GEMINI_API_KEY")
    
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
    
    # Test dynamic activity suggestions
    print("\n" + "-" * 70)
    print("\n🎯 Dynamic Activity Suggestions Test:\n")
    
    suggestions = advisor.get_dynamic_activity_suggestions(
        city="Delhi",
        aqi=344,
        category="Very Poor",
        user_profile="general"
    )
    
    print(suggestions)
    
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
