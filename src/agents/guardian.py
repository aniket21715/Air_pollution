from google import genai
import os

class HealthGuardianAgent:
    """
    The Doctor Agent.
    Role: Empathetic health advisor.
    Goal: Translate technical analysis into actionable safety advice.
    """
    
    def __init__(self, api_key):
        # New SDK: Use Client object
        self.client = genai.Client(api_key=api_key)
        self.model_name = 'gemini-2.5-flash-lite'
        
    def provide_advice(self, analysis_result, user_profile):
        """
        Generate advice based on the Detective's situation report.
        """
        if "error" in analysis_result:
            return "I cannot provide advice as I am identifying technical issues with the data sensors."
            
        city = analysis_result['city']
        report = analysis_result['situation_report']
        aqi = analysis_result['aqi_data']['aqi']
        category = analysis_result['aqi_data']['category']
        
        prompt = f"""You are the 'Health Guardian', a doctor agent.
        
        INPUT CONTEXT:
        User Profile: {user_profile}
        City: {city}
        Current Status: AQI {aqi} ({category})
        
        SCIENTIST'S SITUATION REPORT:
        "{report}"
        
        TASK:
        Provide actionable health advice.
        - Reference the scientist's finding (e.g. "Since the wind is calm as reported...").
        - Be specific to the user's profile.
        - Suggest: Activity modifications, Mask usage, Indoor precautions.
        
        OUTPUT:
        A friendly, protective message (3-4 sentences)."""
        
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            return response.text
        except Exception as e:
            return f"Advice generation failed: {str(e)}"
