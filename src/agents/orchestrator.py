import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.detective import PollutionDetectiveAgent
from agents.guardian import HealthGuardianAgent
from agents.researcher import PolicyResearcherAgent

class AgentOrchestrator:
    """
    Coordinator for the Multi-Agent System.
    Manages the workflow between:
    - Scientist (Detective): Analyzes pollution data
    - Doctor (Guardian): Provides health advice
    - Researcher: Fetches recent policy information
    """
    
    def __init__(self, api_key):
        self.api_key = api_key
        self.detective = PollutionDetectiveAgent(api_key)
        self.guardian = HealthGuardianAgent(api_key)
        self.researcher = PolicyResearcherAgent(api_key)
        
    def run_analysis(self, city_name, user_profile="general"):
        """
        Run the full agentic workflow.
        1. Activates Detective to analyze data.
        2. Passes findings to Guardian for advice.
        3. Returns combined insight.
        """
        print(f"🕵️‍♂️ Detective Agent activated for {city_name}...")
        analysis = self.detective.analyze_city(city_name)
        
        if "error" in analysis:
            return {
                "error": True,
                "message": analysis["error"]
            }
            
        print(f"👩‍⚕️ Guardian Agent activated with user profile: {user_profile}...")
        advice = self.guardian.provide_advice(analysis, user_profile)
        
        return {
            "city": city_name,
            "aqi": analysis['aqi_data']['aqi'],
            "category": analysis['aqi_data']['category'],
            "situation_report": analysis['situation_report'],
            "health_advice": advice,
            "weather": analysis['weather_data']
        }
    
    def run_policy_research(self, city_name, timeframe="2 years"):
        """
        Run policy research workflow.
        1. Researcher fetches recent policies from web.
        2. Optionally gets recommendations based on current AQI.
        """
        print(f"📚 Policy Researcher Agent activated for {city_name}...")
        
        # Get recent policies
        policies = self.researcher.research_recent_policies(
            city=city_name, 
            timeframe=timeframe
        )
        
        # Get current AQI for context (via detective)
        aqi_data = self.detective.aqi_client.get_current_aqi(city_name)
        current_aqi = aqi_data.get('aqi', 0) if aqi_data else 0
        
        # Get policy recommendations if AQI is available
        recommendations = None
        if current_aqi > 0:
            print(f"🎯 Getting policy recommendations (Current AQI: {current_aqi})...")
            recommendations = self.researcher.get_policy_recommendations(city_name, current_aqi)
        
        return {
            "city": city_name,
            "current_aqi": current_aqi,
            "policies": policies,
            "recommendations": recommendations
        }
    
    def run_full_analysis_with_policy(self, city_name, user_profile="general"):
        """
        Run complete analysis including health advice AND policy research.
        """
        # Standard analysis
        analysis_result = self.run_analysis(city_name, user_profile)
        
        if "error" in analysis_result and analysis_result.get("error"):
            return analysis_result
        
        # Add policy research
        print(f"📚 Policy Researcher Agent activated...")
        policy_result = self.researcher.research_recent_policies(city=city_name)
        
        analysis_result["recent_policies"] = policy_result
        return analysis_result

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    API_KEY = os.getenv("GEMINI_API_KEY")
    if not API_KEY:
        print("❌ Error: GEMINI_API_KEY not found in environment variables.")
        sys.exit(1)
        
    orchestrator = AgentOrchestrator(API_KEY)
    
    # Test Run
    city = "Delhi"
    profile = "Athletic, likes outdoor running"
    
    print(f"\n🚀 Starting Agentic Workflow for {city}...\n")
    result = orchestrator.run_analysis(city, profile)
    
    if "error" in result and result.get("error"):
        print(f"❌ Workflow Failed: {result['message']}")
    else:
        print("\n" + "="*50)
        print(f"📊 REPORT FOR {city.upper()}")
        print("="*50)
        print(f"AQI: {result['aqi']} ({result['category']})")
        print("\n🕵️‍♂️ SITUATION REPORT (SCIENTIST):")
        print(result['situation_report'])
        print("\n👩‍⚕️ HEALTH ADVICE (DOCTOR):")
        print(result['health_advice'])
        print("="*50)

