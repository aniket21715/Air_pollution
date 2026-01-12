"""
Policy Researcher Agent
Dynamically fetches and analyzes recent (2-year) air pollution policies from the web.

Role: The Researcher
Goal: Find and summarize recent government policies on air pollution control.
"""

from google import genai
import os
import json
from datetime import datetime, timedelta
import hashlib

# Simple file-based cache
_CACHE_FILE = os.path.join(os.path.dirname(__file__), ".policy_cache.json")
_CACHE_DURATION_HOURS = 24


class PolicyResearcherAgent:
    """
    Searches the web for recent air pollution policies and extracts structured information.
    Uses Gemini's grounding capability to access current web data.
    """
    
    def __init__(self, api_key):
        self.client = genai.Client(api_key=api_key)
        self.model_name = 'gemini-2.5-flash-lite'
    
    def _get_cache_key(self, city, timeframe):
        """Generate a unique cache key."""
        key_str = f"{city}_{timeframe}_{datetime.now().strftime('%Y-%m-%d')}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _load_cache(self):
        """Load cached results."""
        try:
            if os.path.exists(_CACHE_FILE):
                with open(_CACHE_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception:
            pass
        return {}
    
    def _save_cache(self, cache_data):
        """Save results to cache."""
        try:
            with open(_CACHE_FILE, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2)
        except Exception:
            pass
    
    def _is_cache_valid(self, cache_entry):
        """Check if cache entry is still valid."""
        if not cache_entry:
            return False
        cached_time = datetime.fromisoformat(cache_entry.get('timestamp', '2000-01-01'))
        return datetime.now() - cached_time < timedelta(hours=_CACHE_DURATION_HOURS)
    
    def research_recent_policies(self, city=None, timeframe="2 years"):
        """
        Research recent air pollution policies from the web.
        
        Args:
            city: Specific city to focus on (e.g., "Delhi"). If None, searches all-India policies.
            timeframe: How far back to search (default "2 years")
        
        Returns:
            Dictionary with policies list and metadata
        """
        # Check cache first
        cache_key = self._get_cache_key(city or "India", timeframe)
        cache = self._load_cache()
        
        if cache_key in cache and self._is_cache_valid(cache[cache_key]):
            print("📦 Using cached policy data...")
            return cache[cache_key]['data']
        
        # Build the search prompt
        city_context = f"for {city}" if city else "across India"
        current_year = datetime.now().year
        
        prompt = f"""You are a policy research specialist. Search for and analyze recent air pollution control policies {city_context} from the last {timeframe}.

Focus on:
1. Government policies (Central and State level)
2. CPCB (Central Pollution Control Board) directives  
3. NGT (National Green Tribunal) orders
4. GRAP updates (Graded Response Action Plan)
5. City-specific initiatives (Odd-Even, construction bans, etc.)

For each policy found, extract:
- Policy name
- Implementation date (month/year)
- Implementing authority
- Key provisions
- Expected AQI impact
- Affected cities/regions
- Current status (active/completed/ongoing)

OUTPUT FORMAT (JSON only, no markdown):
{{
    "search_date": "{datetime.now().strftime('%Y-%m-%d')}",
    "region": "{city or 'India'}",
    "timeframe": "{timeframe}",
    "policies": [
        {{
            "name": "Policy Name",
            "date": "Month Year",
            "authority": "Implementing Body",
            "description": "Brief description",
            "key_provisions": ["provision 1", "provision 2"],
            "expected_impact": "Expected AQI reduction %",
            "affected_regions": ["City1", "City2"],
            "status": "active/completed/ongoing",
            "category": "vehicular/industrial/construction/emergency/general"
        }}
    ],
    "summary": "A 2-sentence summary of the policy landscape"
}}

Provide at least 5-8 relevant policies from {current_year-1}-{current_year}. Be factual and cite real policies only."""

        try:
            # Use Gemini with search grounding for real-time web data
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config={
                    "tools": [{"google_search": {}}],  # Enable web search grounding
                }
            )
            
            # Parse the response
            result_text = response.text.strip()
            
            # Clean markdown code blocks if present
            if result_text.startswith("```"):
                result_text = result_text.split("```")[1]
                if result_text.startswith("json"):
                    result_text = result_text[4:]
            
            result = json.loads(result_text.strip())
            
            # Cache the result
            cache[cache_key] = {
                'timestamp': datetime.now().isoformat(),
                'data': result
            }
            self._save_cache(cache)
            
            return result
            
        except json.JSONDecodeError:
            # If JSON parsing fails, return a structured error with the raw response
            return {
                "search_date": datetime.now().strftime('%Y-%m-%d'),
                "region": city or "India",
                "timeframe": timeframe,
                "policies": [],
                "error": "Failed to parse policy data",
                "raw_response": response.text if 'response' in dir() else "No response"
            }
        except Exception as e:
            return {
                "search_date": datetime.now().strftime('%Y-%m-%d'),
                "region": city or "India",
                "timeframe": timeframe,
                "policies": [],
                "error": str(e)
            }
    
    def analyze_policy_effectiveness(self, policy_name, current_aqi, historical_avg_aqi):
        """
        Use AI to analyze how effective a specific policy might be.
        
        Args:
            policy_name: Name of the policy
            current_aqi: Current AQI value
            historical_avg_aqi: Historical average AQI
        
        Returns:
            AI analysis of policy effectiveness
        """
        prompt = f"""Analyze the effectiveness of the policy: "{policy_name}"

Current Situation:
- Current AQI: {current_aqi}
- Historical Average AQI: {historical_avg_aqi}
- Change: {((current_aqi - historical_avg_aqi) / historical_avg_aqi * 100):.1f}%

Provide a brief (3-4 sentences) analysis of:
1. Whether this policy is likely contributing to AQI improvement
2. Challenges in implementation
3. Recommendations for better enforcement

Be concise and data-driven."""

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            return response.text
        except Exception as e:
            return f"Analysis unavailable: {str(e)}"
    
    def get_policy_recommendations(self, city, current_aqi):
        """
        Get AI recommendations for new policies based on current conditions.
        
        Args:
            city: City name
            current_aqi: Current AQI value
        
        Returns:
            List of recommended policy interventions
        """
        severity = "severe" if current_aqi > 300 else "high" if current_aqi > 200 else "moderate"
        
        prompt = f"""Based on {city}'s current AQI of {current_aqi} ({severity} pollution level), recommend 3-4 specific policy interventions.

For each recommendation:
1. Policy name
2. Implementation timeframe (immediate/short-term/long-term)
3. Expected impact
4. Reference to similar successful policies in other cities

OUTPUT FORMAT (JSON only):
{{
    "city": "{city}",
    "current_aqi": {current_aqi},
    "severity": "{severity}",
    "recommendations": [
        {{
            "policy": "Policy Name",
            "timeframe": "immediate/short-term/long-term",
            "expected_impact": "X% reduction in AQI",
            "reference": "Similar policy in City X achieved Y results"
        }}
    ]
}}"""

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
            return {"error": str(e), "recommendations": []}


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    key = os.getenv("GEMINI_API_KEY")
    agent = PolicyResearcherAgent(key)
    
    print("=" * 70)
    print("  POLICY RESEARCHER AGENT TEST")
    print("=" * 70)
    
    # Test 1: Research recent policies for Delhi
    print("\n🔍 Researching recent policies for Delhi...")
    policies = agent.research_recent_policies(city="Delhi", timeframe="2 years")
    
    if "error" not in policies:
        print(f"\n📋 Found {len(policies.get('policies', []))} policies:")
        for p in policies.get('policies', [])[:5]:
            print(f"\n   📌 {p.get('name', 'Unknown')}")
            print(f"      Date: {p.get('date', 'N/A')}")
            print(f"      Status: {p.get('status', 'N/A')}")
            print(f"      Impact: {p.get('expected_impact', 'N/A')}")
        
        print(f"\n📝 Summary: {policies.get('summary', 'N/A')}")
    else:
        print(f"❌ Error: {policies.get('error')}")
    
    # Test 2: Get policy recommendations
    print("\n" + "=" * 70)
    print("🎯 Getting policy recommendations for Delhi (AQI 320)...")
    recommendations = agent.get_policy_recommendations("Delhi", 320)
    
    if "error" not in recommendations:
        for rec in recommendations.get('recommendations', []):
            print(f"\n   💡 {rec.get('policy', 'Unknown')}")
            print(f"      Timeframe: {rec.get('timeframe', 'N/A')}")
            print(f"      Expected: {rec.get('expected_impact', 'N/A')}")
    else:
        print(f"❌ Error: {recommendations.get('error')}")
