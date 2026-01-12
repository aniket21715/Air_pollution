"""
Insurance Planning Agent
Helps users plan healthcare coverage based on city air pollution levels.

Role: The Insurance Planner
Goal: Provide actionable insurance and healthcare planning advice based on pollution exposure.
"""

from google import genai
import os
import json
from datetime import datetime
import hashlib

# Cache settings
_CACHE_FILE = os.path.join(os.path.dirname(__file__), ".insurance_cache.json")
_CACHE_DURATION_HOURS = 48  # Insurance data doesn't change often


class InsurancePlannerAgent:
    """
    AI agent that helps users plan healthcare coverage based on city pollution levels.
    Uses web search to find current insurance plans and health recommendations.
    """
    
    def __init__(self, api_key):
        self.client = genai.Client(api_key=api_key)
        self.model_name = 'gemini-2.5-flash-lite'
    
    def _get_cache_key(self, *args):
        """Generate unique cache key."""
        key_str = "_".join(str(a) for a in args) + f"_{datetime.now().strftime('%Y-%m-%d')}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _load_cache(self):
        try:
            if os.path.exists(_CACHE_FILE):
                with open(_CACHE_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception:
            pass
        return {}
    
    def _save_cache(self, cache_data):
        try:
            with open(_CACHE_FILE, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2)
        except Exception:
            pass
    
    def get_pollution_health_risk(self, city, aqi, user_age=30, conditions=[]):
        """
        Calculate health risk score based on pollution and user profile.
        
        Args:
            city: City name
            aqi: Current AQI value
            user_age: User's age
            conditions: List of pre-existing conditions
        
        Returns:
            Risk assessment with score and recommendations
        """
        # Base risk from AQI
        if aqi <= 50:
            base_risk = "Low"
            risk_score = 15
        elif aqi <= 100:
            base_risk = "Moderate"
            risk_score = 30
        elif aqi <= 200:
            base_risk = "High"
            risk_score = 55
        elif aqi <= 300:
            base_risk = "Very High"
            risk_score = 75
        else:
            base_risk = "Severe"
            risk_score = 95
        
        # Age modifier
        if user_age < 18 or user_age > 60:
            risk_score = min(100, risk_score + 15)
        
        # Condition modifiers
        high_risk_conditions = ['asthma', 'copd', 'heart_disease', 'diabetes']
        for condition in conditions:
            if condition.lower() in high_risk_conditions:
                risk_score = min(100, risk_score + 10)
        
        return {
            "city": city,
            "aqi": aqi,
            "base_risk": base_risk,
            "risk_score": risk_score,
            "risk_level": "Low" if risk_score < 30 else "Moderate" if risk_score < 50 else "High" if risk_score < 75 else "Critical",
            "user_age": user_age,
            "conditions": conditions
        }
    
    def get_health_checkup_recommendations(self, city, aqi, user_profile="general"):
        """
        Get recommended health checkups for pollution exposure.
        """
        prompt = f"""You are a preventive healthcare specialist. Based on the air quality situation, recommend health checkups.

CONTEXT:
- City: {city}
- Current AQI: {aqi}
- User Profile: {user_profile}

Provide health checkup recommendations in JSON format:
{{
    "risk_level": "low/moderate/high",
    "annual_checkups": [
        {{
            "test_name": "Test Name",
            "purpose": "Why this test is needed",
            "frequency": "How often",
            "approximate_cost_inr": 1000
        }}
    ],
    "pollution_specific_tests": [
        {{
            "test_name": "Pulmonary Function Test",
            "purpose": "Check lung capacity affected by pollution",
            "recommended_if_aqi_above": 200,
            "approximate_cost_inr": 1500
        }}
    ],
    "total_annual_checkup_cost_inr": 5000,
    "tips": ["Tip 1", "Tip 2"]
}}

Focus on tests specifically relevant to air pollution exposure like:
- Lung function tests (spirometry, PFT)
- Chest X-ray
- Blood oxygen levels
- Cardiovascular screening
- Allergy tests

Be practical and India-specific with costs."""

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            
            result_text = response.text.strip()
            if result_text.startswith("```"):
                result_text = result_text.split("```")[1]
                if result_text.startswith("json"):
                    result_text = result_text[4:]
            
            return json.loads(result_text.strip())
        except Exception as e:
            return self._get_default_checkup_recommendations(aqi)
    
    def _get_default_checkup_recommendations(self, aqi):
        """Fallback recommendations when API fails."""
        tests = [
            {"test_name": "Pulmonary Function Test (PFT)", "purpose": "Assess lung capacity", "frequency": "Annual", "approximate_cost_inr": 1500},
            {"test_name": "Chest X-Ray", "purpose": "Check for lung abnormalities", "frequency": "Annual", "approximate_cost_inr": 500},
            {"test_name": "Complete Blood Count (CBC)", "purpose": "General health marker", "frequency": "6 months", "approximate_cost_inr": 400},
        ]
        
        if aqi > 200:
            tests.append({"test_name": "HRCT Chest", "purpose": "Detailed lung imaging", "frequency": "As needed", "approximate_cost_inr": 4000})
        
        return {
            "risk_level": "high" if aqi > 200 else "moderate",
            "annual_checkups": tests,
            "pollution_specific_tests": [],
            "total_annual_checkup_cost_inr": sum(t["approximate_cost_inr"] for t in tests),
            "tips": ["Schedule checkups in morning for accurate results", "Carry previous reports for comparison"]
        }
    
    def get_insurance_recommendations(self, city, aqi, user_age=30, family_size=1, conditions=[]):
        """
        Get insurance plan recommendations using web search for real data.
        """
        risk_level = "high" if aqi > 200 else "moderate" if aqi > 100 else "low"
        
        prompt = f"""You are an insurance advisor specializing in health coverage for pollution-affected areas.

CONTEXT:
- City: {city} (AQI: {aqi}, Risk: {risk_level})
- User Age: {user_age}
- Family Size: {family_size}
- Pre-existing Conditions: {conditions if conditions else "None"}

Search for and recommend health insurance plans available in India that cover:
1. Respiratory diseases
2. Pollution-related illness
3. Regular health checkups

OUTPUT FORMAT (JSON only):
{{
    "recommended_coverage_amount_lakhs": 10,
    "coverage_justification": "Why this amount",
    "insurance_plans": [
        {{
            "provider": "Star Health / HDFC Ergo / etc",
            "plan_name": "Actual plan name",
            "coverage_lakhs": 10,
            "annual_premium_inr": 15000,
            "key_features": ["Feature 1", "Feature 2"],
            "covers_pollution_illness": true,
            "waiting_period_months": 3,
            "best_for": "Who this plan suits"
        }}
    ],
    "recommended_add_ons": [
        {{
            "name": "Critical Illness Cover",
            "why_needed": "For pollution-exposed individuals",
            "approximate_cost_inr": 2000
        }}
    ],
    "annual_out_of_pocket_estimate_inr": 5000,
    "tips": [
        "Buy insurance before age 35 for lower premiums",
        "Choose plans with low co-pay for respiratory treatments"
    ]
}}

Search for REAL insurance plans from providers like:
- Star Health, HDFC Ergo, ICICI Lombard, Max Bupa, Care Health
Include actual plan names and realistic premium estimates for 2024-2026."""

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config={
                    "tools": [{"google_search": {}}],  # Enable web search
                }
            )
            
            result_text = response.text.strip()
            if result_text.startswith("```"):
                result_text = result_text.split("```")[1]
                if result_text.startswith("json"):
                    result_text = result_text[4:]
            
            return json.loads(result_text.strip())
        except Exception as e:
            return self._get_default_insurance_recommendations(aqi, user_age, family_size)
    
    def _get_default_insurance_recommendations(self, aqi, user_age, family_size):
        """Fallback when API fails."""
        base_coverage = 5 if aqi < 150 else 10 if aqi < 250 else 15
        base_premium = 8000 if user_age < 35 else 12000 if user_age < 50 else 18000
        
        return {
            "recommended_coverage_amount_lakhs": base_coverage,
            "coverage_justification": f"Based on AQI {aqi} and age {user_age}",
            "insurance_plans": [
                {
                    "provider": "Star Health",
                    "plan_name": "Comprehensive Health Insurance",
                    "coverage_lakhs": base_coverage,
                    "annual_premium_inr": base_premium * family_size,
                    "key_features": ["Cashless hospitalization", "Pre & post hospitalization", "Day care procedures"],
                    "covers_pollution_illness": True,
                    "waiting_period_months": 3,
                    "best_for": "Families in polluted cities"
                }
            ],
            "recommended_add_ons": [
                {"name": "Critical Illness Cover", "why_needed": "Extra protection for severe conditions", "approximate_cost_inr": 3000}
            ],
            "annual_out_of_pocket_estimate_inr": 5000,
            "tips": ["Compare plans from multiple providers", "Check network hospitals in your area"]
        }
    
    def get_nearby_hospitals(self, city, specialty="pulmonology"):
        """
        Get nearby hospitals with specified specialty.
        """
        prompt = f"""Search for top hospitals in {city}, India that specialize in {specialty} and respiratory care.

OUTPUT FORMAT (JSON):
{{
    "city": "{city}",
    "specialty": "{specialty}",
    "hospitals": [
        {{
            "name": "Hospital Name",
            "type": "Government/Private",
            "specialties": ["Pulmonology", "Cardiology"],
            "rating": 4.5,
            "address": "Area/Landmark",
            "emergency_available": true,
            "estimated_consultation_fee_inr": 500
        }}
    ],
    "emergency_helpline": "108"
}}

Include 5-6 well-known hospitals with actual names."""

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config={
                    "tools": [{"google_search": {}}],
                }
            )
            
            result_text = response.text.strip()
            if result_text.startswith("```"):
                result_text = result_text.split("```")[1]
                if result_text.startswith("json"):
                    result_text = result_text[4:]
            
            return json.loads(result_text.strip())
        except Exception as e:
            return {"city": city, "hospitals": [], "error": str(e)}


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    key = os.getenv("GEMINI_API_KEY")
    agent = InsurancePlannerAgent(key)
    
    print("=" * 70)
    print("  INSURANCE PLANNING AGENT TEST")
    print("=" * 70)
    
    # Test 1: Health Risk
    print("\n📊 Health Risk Assessment (Delhi, AQI 320):")
    risk = agent.get_pollution_health_risk("Delhi", 320, user_age=35, conditions=["asthma"])
    print(f"   Risk Score: {risk['risk_score']}/100 ({risk['risk_level']})")
    
    # Test 2: Checkup Recommendations
    print("\n💊 Health Checkup Recommendations:")
    checkups = agent.get_health_checkup_recommendations("Delhi", 320)
    for test in checkups.get('annual_checkups', [])[:3]:
        print(f"   • {test.get('test_name')} - ₹{test.get('approximate_cost_inr')}")
    
    # Test 3: Insurance Recommendations
    print("\n🛡️ Insurance Recommendations:")
    insurance = agent.get_insurance_recommendations("Delhi", 320, user_age=35, family_size=4)
    print(f"   Recommended Coverage: ₹{insurance.get('recommended_coverage_amount_lakhs', 0)} Lakhs")
    for plan in insurance.get('insurance_plans', [])[:2]:
        print(f"   • {plan.get('provider')} - {plan.get('plan_name')}")
        print(f"     Premium: ₹{plan.get('annual_premium_inr')}/year")
