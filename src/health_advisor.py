"""
Health Advisor Module
Provides personalized, actionable health recommendations based on:
- Current and forecasted AQI levels
- User profile (age, health conditions, activities)
- Activity type and timing
"""

from indian_cities_config import get_aqi_category

class HealthAdvisor:
    """
    Generates personalized health and activity recommendations
    based on air quality levels and user profiles.
    """
    
    # User profile definitions
    PROFILES = {
        "general": {
            "name": "General Population",
            "description": "Healthy adults with no specific conditions",
            "sensitivity": "normal"
        },
        "elderly": {
            "name": "Elderly (65+ years)",
            "description": "Senior citizens with potential reduced lung capacity",
            "sensitivity": "high"
        },
        "children": {
            "name": "Children (0-12 years)",
            "description": "Developing respiratory systems, more vulnerable",
            "sensitivity": "high"
        },
        "asthma": {
            "name": "Asthma & Respiratory Conditions",
            "description": "People with asthma, COPD, or other lung diseases",
            "sensitivity": "very_high"
        },
        "pregnant": {
            "name": "Pregnant Women",
            "description": "Risk to both mother and fetal development",
            "sensitivity": "very_high"
        },
        "athlete": {
            "name": "Athletes & Active Individuals",
            "description": "People engaging in regular outdoor exercise",
            "sensitivity": "medium"
        },
        "heart_disease": {
            "name": "Heart Disease Patients",
            "description": "People with cardiovascular conditions",
            "sensitivity": "very_high"
        }
    }
    
    # Activity recommendations by AQI range and profile
    RECOMMENDATIONS = {
        "Good": {  # AQI 0-50
            "general": {
                "icon": "✅",
                "message": "Air quality is excellent! Safe for all outdoor activities.",
                "actions": [
                    "Perfect day for outdoor exercise and sports",
                    "Open windows for natural ventilation",
                    "Children can play outside freely",
                    "Ideal for long walks and running"
                ]
            },
            "all": {
                "icon": "✅",
                "message": "Air quality is excellent for everyone!",
                "actions": ["Enjoy outdoor activities without concerns"]
            }
        },
        "Satisfactory": {  # AQI 51-100
            "general": {
                "icon": "🟢",
                "message": "Air quality is acceptable for most people.",
                "actions": [
                    "Safe for outdoor activities",
                    "Sensitive individuals should monitor symptoms",
                    "Good time for exercise and recreation"
                ]
            },
            "asthma": {
                "icon": "⚠️",
                "message": "MODERATE RISK: Monitor your breathing closely.",
                "actions": [
                    "Have your rescue inhaler readily available",
                    "Consider shorter outdoor sessions",
                    "Watch for symptoms like wheezing or chest tightness"
                ]
            },
            "pregnant": {
                "icon": "⚠️",
                "message": "Generally safe but limit prolonged exposure.",
                "actions": [
                    "Moderate outdoor activities are acceptable",
                    "Avoid strenuous exercise outdoors"
                ]
            }
        },
        "Moderate": {  # AQI 101-200
            "general": {
                "icon": "🟠",
                "message": "Sensitive groups may experience health effects.",
                "actions": [
                    "Limit prolonged outdoor exertion",
                    "Close windows during peak pollution hours",
                    "Consider indoor exercise alternatives"
                ]
            },
            "asthma": {
                "icon": "🚨",
                "message": "HIGH RISK: Avoid outdoor activities if possible.",
                "actions": [
                    "Use preventive inhaler before going out",
                    "Stay indoors as much as possible",
                    "If you must go out, wear an N95 mask",
                    "   Keep indoor activities","Have peak flow meter ready",
                    "Avoid exercise entirely—opt for indoor yoga or stretching"
                ]
            },
            "elderly": {
                "icon": "⚠️",
                "message": "Reduce outdoor exposure, especially strenuous activities.",
                "actions": [
                    "Limit time outdoors to essential activities only",
                    "Avoid morning and evening hours (higher pollution)",
                    "Use air purifiers indoors"
                ]
            },
            "children": {
                "icon": "⚠️",
                "message": "Reduce outdoor play time and avoid strenuous games.",
                "actions": [
                    "Keep outdoor play to minimum",
                    "Schools should cancel outdoor sports",
                    "Indoor activities recommended"
                ]
            },
            "athlete": {
                "icon": "🏃‍♂️",
                "message": "Switch to indoor training or reschedule workouts.",
                "actions": [
                    "⛔ AVOID outdoor running/cycling",
                    "Switch to gym or indoor training",
                    "If you must train outdoors, choose early morning (6-7 AM)",
                    "Reduce intensity by 30-40%"
                ]
            }
        },
        "Poor": {  # AQI 201-300
            "general": {
                "icon": "🔴",
                "message": "Everyone may experience health effects.",
                "actions": [
                    "⛔ Avoid prolonged outdoor activities",
                    "Wear N95 mask if you must go outside",
                    "Close all windows and use air purifiers",
                    "Postpone or cancel outdoor events",
                    "Work from home if possible"
                ]
            },
            "asthma": {
                "icon": "🚨",
                "message": "SEVERE RISK: Stay indoors. Medical attention may be needed.",
                "actions": [
                    "🏠 STAY INDOORS - do not go outside",
                    "Use your preventive medication even if you feel okay",
                    "Keep rescue inhaler within arm's reach at all times",
                    "Monitor oxygen saturation if you have a pulse oximeter",
                    "Call doctor if you experience breathlessness",
                    "Run air purifiers on high setting"
                ]
            },
            "elderly": {
                "icon": "🚨",
                "message": "STAY INDOORS: High risk of respiratory complications.",
                "actions": [
                    "Do not venture outside unless absolutely necessary",
                    "Cancel non-essential appointments",
                    "Monitor for chest discomfort or breathing difficulty",
                    "Keep emergency contacts ready"
                ]
            },
            "children": {
                "icon": "🚨",
                "message": "SCHOOL CLOSURE RECOMMENDED: Air is unsafe for children.",
                "actions": [
                    "Schools should declare half-day or full holiday",
                    "Zero outdoor exposure",
                    "Indoor activities only",
                    "Parents: keep children's health monitored"
                ]
            },
            "pregnant": {
                "icon": "🚨",
                "message": "CRITICAL: Avoid all outdoor exposure for fetal safety.",
                "actions": [
                    "Complete indoor rest recommended",
                    "Use N95 mask even for brief outdoor moments",
                    "Consult obstetrician if experiencing any breathing discomfort"
                ]
            },
            "athlete": {
                "icon": "⛔",
                "message": "ALL OUTDOOR TRAINING CANCELLED: Indoor only.",
                "actions": [
                    "⛔ Complete outdoor exercise ban",
                    "Switch to indoor alternatives (treadmill, stationary bike)",
                    "Reduce training intensity significantly",
                    "Rest day recommended"
                ]
            }
        },
        "Very Poor": {  # AQI 301-400
            "general": {
                "icon": "🔴",
                "message": "Health alert: Everyone is at risk. Minimize outdoor exposure.",
                "actions": [
                    "🏠 STAY INDOORS COMPLETELY",
                    "Wear N95 mask even for brief outdoor exposure",
                    "Seal windows and doors",
                    "Use high-quality air purifiers (HEPA filters)",
                    "Work from home mandatory if possible"
                ]
            },
            "all": {
                "icon": "  🚨",
                "message": "HAZARDOUS CONDITIONS: Medical emergency risk for all groups.",
                "actions": [
                    "Complete indoor isolation",
                    "Even healthy people should not go outside",
                    "Stock up on essentials to avoid going out",
                    "Monitor health symptoms closely",
                    "Seek medical attention for any respiratory distress"
                ]
            }
        },
        "Severe": {  # AQI 401-500
            "all": {
                "icon": "☠️",
                "message": "EMERGENCY: Hazardous to all. Immediate health impacts expected.",
                "actions": [
                    "☠️ PUBLIC HEALTH EMERGENCY",
                    "🏠 ABSOLUTE INDOOR LOCKDOWN",
                    "Do not go outside under ANY circumstances",
                    "Even indoors, use HEPA air purifiers",
                    "Tape windows and door gaps if possible",
                    "Seek immediate medical help for ANY respiratory symptoms",
                    "Government should declare emergency measures"
                ]
            }
        }
    }
    
    @staticmethod
    def get_recommendation(aqi_value, user_profile="general"):
        """
        Get personalized recommendation based on AQI and user profile.
        
        Args:
            aqi_value: Current or forecasted AQI
            user_profile: User profile key (general, asthma, elderly, etc.)
        
        Returns:
            Dict with icon, message, and action list
        """
        # Get AQI category
        category, color, health_impact = get_aqi_category(aqi_value)
        
        # Get recommendations for this category
        category_recs = HealthAdvisor.RECOMMENDATIONS.get(category, {})
        
        # Try to get profile-specific recommendation
        if user_profile in category_recs:
            rec = category_recs[user_profile]
        # Fall back to 'all' profiles
        elif "all" in category_recs:
            rec = category_recs["all"]
        # Fall back to general
        elif "general" in category_recs:
            rec = category_recs["general"]
        else:
            rec = {
                "icon": "ℹ️",
                "message": f"AQI {aqi_value:.0f} - {health_impact}",
                "actions": ["Monitor air quality updates"]
            }
        
        return {
            **rec,
            "aqi": aqi_value,
            "category": category,
            "color": color,
            "health_impact": health_impact
        }
    
    @staticmethod
    def get_activity_guidance(aqi_forecast, activity_type, user_profile="general"):
        """
        Get specific activity guidance based on multi-day forecast.
        
        Args:
            aqi_forecast: List/dict of AQI values for next N days
            activity_type: Type of activity (jog, commute, outdoor_event, etc.)
            user_profile: User profile
        
        Returns:
            Activity-specific recommendations
        """
        if isinstance(aqi_forecast, dict):
            aqi_values = list(aqi_forecast.values())
        else:
            aqi_values = aqi_forecast
        
        min_aqi = min(aqi_values)
        max_aqi = max(aqi_values)
        avg_aqi = sum(aqi_values) / len(aqi_values)
        
        # Activity-specific thresholds
        thresholds = {
            "jog": {"safe": 100, "risky": 150, "dangerous": 200},
            "cycling": {"safe": 100, "risky": 150, "dangerous": 200},
            "outdoor_event": {"safe": 150, "risky": 200, "dangerous": 250},
            "commute": {"safe": 200, "risky": 300, "dangerous": 350},
            "children_play": {"safe": 50, "risky": 100, "dangerous": 150},
            "window_open": {"safe": 100, "risky": 150, "dangerous": 200},
        }
        
        threshold = thresholds.get(activity_type, {"safe": 100, "risky": 200, "dangerous": 300})
        
        # Find best days
        safe_days = [i for i, aqi in enumerate(aqi_values) if aqi <= threshold["safe"]]
        
        result = {
            "activity": activity_type,
            "min_aqi": min_aqi,
            "max_aqi": max_aqi,
            "avg_aqi": avg_aqi,
            "safe_days": safe_days,
        }
        
        # Overall recommendation
        if min_aqi > threshold["dangerous"]:
            result["recommendation"] = "❌ NOT RECOMMENDED - Postpone or move indoors"
            result["status"] = "dangerous"
        elif avg_aqi > threshold["risky"]:
            result["recommendation"] = "⚠️ RISKY - Consider alternatives or reschedule"
            result["status"] = "risky"
        elif len(safe_days) > 0:
            result["recommendation"] = f"✅ SAFE on day(s): {', '.join([f'Day {d+1}' for d in safe_days])}"
            result["status"] = "safe"
        else:
            result["recommendation"] = "⚠️ CAUTION - Proceed with protective measures (N95 mask)"
            result["status"] = "caution"
        
        return result


# Example usage and testing
if __name__ == "__main__":
    print("=" * 60)
    print("  HEALTH ADVISOR MODULE DEMO")
    print("=" * 60)
    
    # Test different scenarios
    test_scenarios = [
        {"aqi": 45, "profile": "general", "desc": "Good day, general population"},
        {"aqi": 120, "profile": "asthma", "desc": "Moderate day, asthma patient"},
        {"aqi": 250, "profile": "children", "desc": "Poor day, children"},
        {"aqi": 350, "profile": "athlete", "desc": "Very poor day, athlete"},
        {"aqi": 450, "profile": "general", "desc": "Severe day, everyone"},
    ]
    
    for scenario in test_scenarios:
        print(f"\n{'─'*60}")
        print(f"Scenario: {scenario['desc']}")
        print(f"AQI: {scenario['aqi']}, Profile: {scenario['profile']}")
        print(f"{'─'*60}")
        
        rec = HealthAdvisor.get_recommendation(scenario['aqi'], scenario['profile'])
        
        print(f"{rec['icon']}  {rec['message']}")
        print(f"\nRecommended Actions:")
        for action in rec['actions']:
            print(f"   • {action}")
    
    # Test activity guidance
    print(f"\n{'='*60}")
    print("ACTIVITY GUIDANCE DEMO")
    print(f"{'='*60}")
    
    forecast = [120, 180, 220, 95, 110, 160, 200]  # 7-day forecast
    activity = "jog"
    
    guidance = HealthAdvisor.get_activity_guidance(forecast, activity, "athlete")
    print(f"\nActivity: {guidance['activity']}")
    print(f"7-Day AQI Range: {guidance['min_aqi']:.0f} - {guidance['max_aqi']:.0f}")
    print(f"Average AQI: {guidance['avg_aqi']:.1f}")
    print(f"\n{guidance['recommendation']}")
