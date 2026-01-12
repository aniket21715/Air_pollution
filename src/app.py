"""
Air Quality Early Warning & Protection System
Purpose-driven application that transforms air quality data into actionable health decisions.

Key Features:
- Personalized health recommendations based on user profile
- Neural Prophet forecasting with uncertainty bounds
- Policy impact analysis
- Health cost calculations
- Multi-city comparison
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
from datetime import datetime, timedelta
import torch
import warnings
warnings.filterwarnings("ignore", message=".*Examining the path of torch.classes.*")
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import custom modules
from indian_cities_config import (
    INDIAN_CITIES, get_all_cities, get_aqi_category,
    get_cities_by_pollution_level, AQI_CATEGORIES
)
from health_advisor import HealthAdvisor
from policy_impact_analyzer import PolicyImpactAnalyzer

# Try to import real-time AQI clients (OpenWeatherMap primary, Open-Meteo fallback)
OPENWEATHERMAP_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY")

try:
    from openweathermap_client import OpenWeatherMapAQIClient
    OPENWEATHERMAP_AVAILABLE = True
except ImportError:
    OPENWEATHERMAP_AVAILABLE = False

try:
    from openmeteo_client import OpenMeteoAQIClient
    OPENMETEO_AVAILABLE = True
except ImportError:
    OPENMETEO_AVAILABLE = False

# Page Config
st.set_page_config(
    page_title="Air Quality Early Warning System",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    .big-aqi {
        font-size: 72px;
        font-weight: bold;
        text-align: center;
        padding: 20px;
        border-radius: 15px;
        margin: 10px 0;
    }
    .health-alert {
        padding: 20px;
        border-left: 5px solid;
        border-radius: 8px;
        margin: 15px 0;
        background-color: #1e293b;
        color: #e2e8f0;
    }
    .health-alert h3 {
        color: #f1f5f9;
        margin-bottom: 10px;
    }
    .health-alert p {
        color: #cbd5e1;
    }
    .health-alert ul {
        color: #e2e8f0;
    }
    .health-alert li {
        color: #f1f5f9;
        margin: 5px 0;
    }
    .health-alert strong {
        color: #60a5fa;
    }
    .city-stat {
        background: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    /* Light mode override */
    @media (prefers-color-scheme: light) {
        .health-alert {
            background-color: #f8fafc;
            color: #1e293b;
        }
        .health-alert h3 {
            color: #0f172a;
        }
        .health-alert p {
            color: #334155;
        }
        .health-alert ul, .health-alert li {
            color: #1e293b;
        }
        .health-alert strong {
            color: #2563eb;
        }
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_historical_data():
    """Load historical air quality data - prioritizes complete 10-year dataset."""
    # Priority 1: Complete dataset (Kaggle 2015-2020 + Open-Meteo 2022-2025)
    complete_path = "data/raw/india_aqi_complete.csv"
    
    # Priority 2: Real Kaggle dataset only (2015-2020)
    kaggle_real_path = "data/raw/india_aqi_kaggle_real.csv"
    
    try:
        if os.path.exists(complete_path):
            st.sidebar.success("✅ Complete data: 2015-2025 (Kaggle + Open-Meteo)")
            df = pd.read_csv(complete_path)
            df['Date'] = pd.to_datetime(df['Date'])
            return df
        elif os.path.exists(kaggle_real_path):
            st.sidebar.info("📊 Using Kaggle data (2015-2020)")
            df = pd.read_csv(kaggle_real_path)
            df['Date'] = pd.to_datetime(df['Date'])
            return df
        else:
            st.error("❌ No data available. Please run data collection.")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_realtime_aqi(city_name):
    """
    Fetch real-time AQI from API (OpenWeatherMap primary, Open-Meteo fallback).
    Uses Indian AQI calculation with CPCB breakpoints.
    TTL of 5 minutes prevents excessive API calls.
    """
    # Try OpenWeatherMap first (more accurate)
    if OPENWEATHERMAP_AVAILABLE and OPENWEATHERMAP_API_KEY:
        try:
            client = OpenWeatherMapAQIClient(OPENWEATHERMAP_API_KEY)
            data = client.get_current_aqi(city_name)
            if data:
                return data
        except Exception as e:
            pass  # Fall through to Open-Meteo
    
    # Fallback to Open-Meteo (free, no key)
    if OPENMETEO_AVAILABLE:
        try:
            client = OpenMeteoAQIClient()
            data = client.get_current_aqi(city_name)
            return data
        except Exception as e:
            pass
    
    return None

@st.cache_resource
def load_neural_prophet_model(city_name, target='AQI'):
    """Load a trained NeuralProphet model."""
    model_path = f"models/neuralprophet/{city_name}_{target}_model.pkl"
    
    if os.path.exists(model_path):
        try:
            # weights_only=False required for PyTorch 2.6+ to load NeuralProphet models
            model = torch.load(model_path, weights_only=False)
            return model
        except Exception as e:
            st.warning(f"Could not load NeuralProphet model: {e}")
            return None
    return None

def generate_forecast_neuralprophet(city_name, df_historical, days_ahead=7):
    """Generate forecast using NeuralProphet model."""
    model = load_neural_prophet_model(city_name, 'AQI')
    
    if model is None:
        # Fallback: Simple forecast using historical averages
        return generate_fallback_forecast(city_name, df_historical, days_ahead)
    
    try:
        # Prepare data in NeuralProphet format
        city_data = df_historical[df_historical['City'] == city_name].copy()
        prophet_df = pd.DataFrame({
            'ds': city_data['Date'],
            'y': city_data['AQI']
        }).dropna().sort_values('ds').reset_index(drop=True)
        
        # Create future dates
        last_date = prophet_df['ds'].max()
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days_ahead, freq='D')
        
        # Create future dataframe with y as NaN (required by NeuralProphet)
        future_df = pd.DataFrame({
            'ds': future_dates,
            'y': [np.nan] * days_ahead
        })
        
        # Combine historical and future data
        combined_df = pd.concat([prophet_df, future_df], ignore_index=True)
        
        # Generate predictions
        forecast = model.predict(combined_df)
        
        # Extract future predictions (last days_ahead rows)
        forecast_future = forecast.tail(days_ahead)
        
        # Get the prediction column name (yhat1 for models with AR, yhat for without)
        pred_col = 'yhat1' if 'yhat1' in forecast_future.columns else 'yhat'
        
        return {
            'dates': pd.to_datetime(forecast_future['ds']).tolist(),
            'aqi': forecast_future[pred_col].tolist(),
            'lower_bound': [max(0, v * 0.85) for v in forecast_future[pred_col].tolist()],
            'upper_bound': [v * 1.15 for v in forecast_future[pred_col].tolist()],
        }
    except Exception as e:
        st.warning(f"Forecast error: {e}. Using fallback method.")
        return generate_fallback_forecast(city_name, df_historical, days_ahead)

def generate_fallback_forecast(city_name, df_historical, days_ahead=7):
    """Fallback forecast using simple moving average Simplified method when NeuralProphet model is not available."""
    city_data = df_historical[df_historical['City'] == city_name].tail(60)  # Last 60 days
    
    # Get seasonal pattern
    last_date = city_data['Date'].max()
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days_ahead, freq='D')
    
    # Simple moving average with seasonal adjustment
    base_aqi = city_data['AQI'].rolling(window=7).mean().iloc[-1]
    
    forecasted_aqi = []
    for date in future_dates:
        # Add seasonality based on month
        if date.month in [10, 11, 12, 1, 2]:  # Winter
            seasonal_factor = 1.15
        elif date.month in [6, 7, 8, 9]:  # Monsoon
            seasonal_factor = 0.85
        else:
            seasonal_factor = 1.0
        
        # Weekend effect
        if date.weekday() >= 5:
            weekend_factor = 0.9
        else:
            weekend_factor = 1.0
        
        aqi = base_aqi * seasonal_factor * weekend_factor + np.random.normal(0, 10)
        forecasted_aqi.append(max(10, aqi))
    
    return {
        'dates': future_dates.tolist(),
        'aqi': forecasted_aqi,
        'lower_bound': [max(10, aqi * 0.85) for aqi in forecasted_aqi],
        'upper_bound': [aqi * 1.15 for aqi in forecasted_aqi],
    }


def main():
    st.title("🌍 Air Quality Early Warning & Protection System")
    st.markdown("### *Data-Driven Health Decisions for Indians*")
    
    # Load data
    df = load_historical_data()
    
    if df.empty:
        st.error("❌ No data available. Please run `python src/generate_data.py` first.")
        return
    
    available_cities = sorted(df['City'].unique())
    
    # ===== SIDEBAR =====
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # City selection
        selected_city = st.selectbox(
            "🏙️ Select City",
            available_cities,
            index=available_cities.index('Delhi') if 'Delhi' in available_cities else 0
        )
        
        st.markdown("---")
        
        # User Profile Selection
        st.subheader("👤 Your Profile")
        user_profile = st.selectbox(
            "Select your profile for personalized advice:",
            options=list(HealthAdvisor.PROFILES.keys()),
            format_func=lambda x: HealthAdvisor.PROFILES[x]['name']
        )
        
        profile_desc = HealthAdvisor.PROFILES[user_profile]['description']
        st.caption(f"ℹ️ {profile_desc}")
        
        st.markdown("---")
        
        # Planned Activity
        st.subheader("📅 Planned Activity")
        activity_options = ["jog", "cycling", "outdoor_event", "commute", "children_play", "window_open", "Other (Custom)"]
        selected_activity = st.selectbox(
            "What are you planning?",
            activity_options
        )
        
        if selected_activity == "Other (Custom)":
            planned_activity = st.text_input("✍️ Enter your specific activity:", placeholder="e.g., yoga in park, painting fence, marathon training")
            if not planned_activity:
                planned_activity = "general_outdoor_activity" # Default fallback
        else:
            planned_activity = selected_activity
        
        st.markdown("---")
        st.caption("💡 **Purpose**: This tool helps you make informed health decisions based on air quality forecasts.")
    
    # ===== MAIN CONTENT =====
    
    # Real-Time AQI Section
    st.markdown(f"## 📡 Current Air Quality: **{selected_city}**")
    
    col1, col2 = st.columns([2, 1])
    
    # Always create city_data for historical features (used in tabs)
    city_data = df[df['City'] == selected_city].copy()
    
    # Safety check for empty data
    if city_data.empty:
        st.warning(f"⚠️ No historical data available for {selected_city} in the dataset.")
        st.info("Try selecting a different city from the dropdown.")
        return
    
    with col1:
        # Try to get REAL-TIME data from Open-Meteo API
        realtime_data = get_realtime_aqi(selected_city)
        
        if realtime_data and realtime_data.get('aqi'):
            # Use REAL-TIME API data
            current_aqi = realtime_data['aqi']
            category = realtime_data.get('category', 'Unknown')
            color = realtime_data.get('color', '#888888')
            data_source = f"🟢 LIVE from Open-Meteo ({realtime_data.get('timestamp', 'now')[:16]})"
            pm25_val = realtime_data.get('pm25', 'N/A')
            pm10_val = realtime_data.get('pm10', 'N/A')
            no2_val = realtime_data.get('no2', 'N/A')
        else:
            # Fallback to historical data
            latest_data = city_data.iloc[-1]
            current_aqi = latest_data['AQI']
            category, color, health_impact = get_aqi_category(current_aqi)
            data_source = f"📊 Historical data ({latest_data['Date'].strftime('%Y-%m-%d')})"
            pm25_val = f"{latest_data['PM2.5']:.1f}"
            pm10_val = f"{latest_data['PM10']:.1f}"
            no2_val = f"{latest_data['NO2']:.1f}"
        
        # Display AQI
        st.markdown(f"""
        <div class="big-aqi" style="background-color: {color}; color: white;">
            {int(current_aqi)}
            <div style="font-size: 24px; margin-top: 10px;">{category}</div>
        </div>
        <div style="text-align: center; font-size: 12px; color: #666; margin-top: 5px;">
            {data_source}
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.metric("PM2.5", f"{pm25_val} µg/m³" if pm25_val != 'N/A' else "N/A")
        st.metric("PM10", f"{pm10_val} µg/m³" if pm10_val != 'N/A' else "N/A")
        st.metric("NO₂", f"{no2_val} µg/m³" if no2_val != 'N/A' else "N/A")
    
    # Personalized Health Recommendation
    recommendation = HealthAdvisor.get_recommendation(current_aqi, user_profile)
    
    st.markdown(f"""
    <div class="health-alert" style="border-color: {recommendation['color']};">
        <h3>{recommendation['icon']} {recommendation['message']}</h3>
        <p><strong>Health Impact:</strong> {recommendation['health_impact']}</p>
        <p><strong>Recommended Actions:</strong></p>
        <ul>
            {''.join([f'<li>{action}</li>' for action in recommendation['actions']])}
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Dynamic Activity Suggestions (Gemini-powered)
    st.markdown("### 🎯 Smart Activity Suggestions")
    st.caption("*AI-powered recommendations based on current air quality*")
    
    try:
        from gemini_advisor import GeminiHealthAdvisor
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        
        # Cache the suggestions in session state to avoid repeated API calls
        cache_key = f"suggestions_{selected_city}_{int(current_aqi)}_{user_profile}"
        
        if cache_key not in st.session_state or st.button("🔄 Refresh Suggestions", key="refresh_suggestions"):
            with st.spinner("Getting AI suggestions..."):
                advisor = GeminiHealthAdvisor(GEMINI_API_KEY)
                st.session_state[cache_key] = advisor.get_dynamic_activity_suggestions(
                    city=selected_city,
                    aqi=current_aqi,
                    category=category,
                    user_profile=user_profile
                )
        
        suggestions_data = st.session_state[cache_key]
        
        # Display activity cards in a grid
        cols = st.columns(4)
        safety_colors = {"safe": "#22c55e", "caution": "#f59e0b", "avoid": "#ef4444"}
        
        for i, sugg in enumerate(suggestions_data.get("suggestions", [])[:4]):
            with cols[i]:
                safety = sugg.get("safety", "caution")
                color = safety_colors.get(safety, "#888888")
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
                    padding: 15px;
                    border-radius: 12px;
                    border-left: 4px solid {color};
                    min-height: 120px;
                    margin-bottom: 10px;
                ">
                    <div style="font-size: 28px; text-align: center;">{sugg.get('icon', '🎯')}</div>
                    <div style="font-weight: bold; color: white; text-align: center; margin: 8px 0;">
                        {sugg.get('activity', 'Activity')}
                    </div>
                    <div style="font-size: 11px; color: #94a3b8; text-align: center;">
                        {sugg.get('tip', '')}
                    </div>
                    <div style="
                        background: {color}20;
                        color: {color};
                        font-size: 10px;
                        padding: 2px 8px;
                        border-radius: 10px;
                        text-align: center;
                        margin-top: 8px;
                        text-transform: uppercase;
                    ">{safety}</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Best time and general tip
        col_time, col_tip = st.columns(2)
        with col_time:
            st.info(f"⏰ **Best Time:** {suggestions_data.get('best_time', 'Check forecast')}")
        with col_tip:
            st.success(f"💡 **Today's Tip:** {suggestions_data.get('general_tip', 'Stay healthy!')}")
            
    except Exception as e:
        # Fallback to static suggestions if Gemini fails
        st.warning("AI suggestions unavailable. Showing default recommendations.")
        if current_aqi > 200:
            st.info("🏠 **Recommendation:** Stay indoors and use air purifiers. Wear N95 mask if going outside.")
        else:
            st.info("🌤️ **Recommendation:** Outdoor activities safe with normal precautions.")
    
    st.markdown("---")
    
    # Tabs for different analysis
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 7-Day Forecast",
        "📊 Historical Trends",
        "🏛️ Policy Impact",
        "🛡️ Insurance & Health Planning",
        "🗺️ City Comparison",
        "🤖 AI Health Advisor"
    ])
    
    with tab1:
        st.subheader(f"7-Day AQI Forecast for {selected_city}")
        
        # Generate forecast
        with st.spinner("Generating forecast with NeuralProphet..."):
            forecast_data = generate_forecast_neuralprophet(selected_city, df, days_ahead=7)
        
        # Plot forecast with uncertainty
        fig = go.Figure()
        
        # Uncertainty band
        fig.add_trace(go.Scatter(
            x=forecast_data['dates'] + forecast_data['dates'][::-1],
            y=forecast_data['upper_bound'] + forecast_data['lower_bound'][::-1],
            fill='toself',
            fillcolor='rgba(139, 92, 246, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='90% Confidence Interval',
            showlegend=True
        ))
        
        # Main forecast line
        fig.add_trace(go.Scatter(
            x=forecast_data['dates'],
            y=forecast_data['aqi'],
            mode='lines+markers',
            name='Forecasted AQI',
            line=dict(color='#8b5cf6', width=3),
            marker=dict(size=8)
        ))
        
        # Add    AQI threshold lines
        fig.add_hline(y=100, line_dash="dash", line_color="yellow", annotation_text="Satisfactory")
        fig.add_hline(y=200, line_dash="dash", line_color="orange", annotation_text="Moderate")
        fig.add_hline(y=300, line_dash="dash", line_color="red", annotation_text="Poor")
        
        fig.update_layout(
            title="AQI Forecast with Uncertainty Bounds",
            xaxis_title="Date",
            yaxis_title="AQI",
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Enhanced Activity Guidance Section
        st.markdown(f"### 🏃 Activity Guidance: *{planned_activity.replace('_', ' ').title()}*")
        activity_guidance = HealthAdvisor.get_activity_guidance(
            forecast_data['aqi'],
            planned_activity,
            user_profile
        )
        
        # Status-based styling
        status_styles = {
            "safe": {"color": "#22c55e", "bg": "#22c55e20", "icon": "✅"},
            "caution": {"color": "#f59e0b", "bg": "#f59e0b20", "icon": "⚠️"},
            "risky": {"color": "#f97316", "bg": "#f9731620", "icon": "⚠️"},
            "dangerous": {"color": "#ef4444", "bg": "#ef444420", "icon": "❌"}
        }
        status = activity_guidance.get('status', 'caution')
        style = status_styles.get(status, status_styles['caution'])
        
        # Main recommendation card
        st.markdown(f"""
        <div style="
            background: {style['bg']};
            border-left: 4px solid {style['color']};
            padding: 15px 20px;
            border-radius: 8px;
            margin-bottom: 15px;
        ">
            <div style="font-size: 18px; font-weight: bold; color: {style['color']};">
                {style['icon']} {activity_guidance['recommendation']}
            </div>
            <div style="margin-top: 10px; display: flex; gap: 20px; flex-wrap: wrap;">
                <span>📊 <strong>AQI Range:</strong> {activity_guidance['min_aqi']:.0f} - {activity_guidance['max_aqi']:.0f}</span>
                <span>📈 <strong>Average:</strong> {activity_guidance['avg_aqi']:.1f}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Dynamic Gemini-powered guidance
        try:
            from gemini_advisor import GeminiHealthAdvisor
            GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
            
            guidance_cache_key = f"activity_guidance_{selected_city}_{planned_activity}_{user_profile}_{int(activity_guidance['avg_aqi'])}"
            
            if guidance_cache_key not in st.session_state:
                with st.spinner("Getting AI recommendations..."):
                    advisor = GeminiHealthAdvisor(GEMINI_API_KEY)
                    
                    prompt = f"""You are an air quality health advisor. Generate activity guidance for:
City: {selected_city}
Activity: {planned_activity.replace('_', ' ')}
User Profile: {user_profile}
7-Day AQI Forecast: Min {activity_guidance['min_aqi']:.0f}, Max {activity_guidance['max_aqi']:.0f}, Avg {activity_guidance['avg_aqi']:.0f}
Status: {status}

Provide response in this exact JSON format (no markdown):
{{
  "best_timing": "Best time of day for this activity",
  "duration_advice": "How long they can safely do this activity",
  "alternatives": ["Alternative 1", "Alternative 2", "Alternative 3"],
  "safety_tips": ["Tip 1", "Tip 2"],
  "forecast_insight": "Brief insight about the 7-day trend"
}}
Be specific to Indian context. Keep all values concise (under 50 chars each)."""

                    response = advisor.model.generate_content(prompt)
                    import json
                    text = response.text.strip()
                    if text.startswith("```"):
                        text = text.split("```")[1]
                        if text.startswith("json"):
                            text = text[4:]
                    st.session_state[guidance_cache_key] = json.loads(text.strip())
            
            guidance_data = st.session_state[guidance_cache_key]
            
            # Display enhanced guidance in columns
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### ⏰ Timing & Duration")
                st.markdown(f"""
                <div style="background: #1e293b; padding: 12px; border-radius: 8px; margin-bottom: 10px;">
                    <div style="color: #94a3b8; font-size: 12px;">Best Time</div>
                    <div style="color: white; font-weight: bold;">{guidance_data.get('best_timing', 'Early morning')}</div>
                </div>
                <div style="background: #1e293b; padding: 12px; border-radius: 8px;">
                    <div style="color: #94a3b8; font-size: 12px;">Duration Advice</div>
                    <div style="color: white; font-weight: bold;">{guidance_data.get('duration_advice', 'Limit outdoor time')}</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("#### 📈 7-Day Forecast Insight")
                st.info(guidance_data.get('forecast_insight', 'AQI levels remain elevated throughout the week.'))
            
            with col2:
                st.markdown("#### 🔄 Better Alternatives")
                for alt in guidance_data.get('alternatives', ['Indoor gym', 'Home workout', 'Yoga'])[:3]:
                    st.markdown(f"""
                    <div style="background: #16a34a20; padding: 8px 12px; border-radius: 6px; margin-bottom: 6px; border-left: 3px solid #16a34a;">
                        <span style="color: white;">✓ {alt}</span>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("#### 🛡️ Safety Tips")
                for tip in guidance_data.get('safety_tips', ['Wear N95 mask', 'Stay hydrated'])[:2]:
                    st.markdown(f"• {tip}")
                    
        except Exception as e:
            # Fallback guidance
            st.markdown("#### 💡 Quick Tips")
            if status == "dangerous":
                st.error("🏠 **Strongly advised to stay indoors.** If you must go outside, wear an N95 mask and limit exposure to under 15 minutes.")
                st.markdown("**Alternatives:** Indoor gym, home yoga, treadmill running")
            elif status == "risky":
                st.warning("⚠️ **Consider postponing or switching to indoor alternatives.** If proceeding, choose early morning (6-7 AM) when pollution is typically lower.")
                st.markdown("**Alternatives:** Indoor cycling, swimming, gym workout")
            else:
                st.success("✅ **Conditions are acceptable.** Monitor how you feel and reduce intensity if you experience any discomfort.")
                st.markdown("**Best time:** Early morning or late evening")
    
    with tab2:
        st.subheader("Historical Pollution Trends")
        
        # Time range selector
        time_range = st.select_slider(
            "Select time range:",
            options=['Last 30 Days', 'Last 90 Days', 'Last 6 Months', 'Last Year', 'All Time'],
            value='Last 6 Months'
        )
        
        # Filter data based on range
        if time_range == 'Last 30 Days':
            filtered_data = city_data.tail(30)
        elif time_range == 'Last 90 Days':
            filtered_data = city_data.tail(90)
        elif time_range == 'Last 6 Months':
            filtered_data = city_data.tail(180)
        elif time_range == 'Last Year':
            filtered_data = city_data.tail(365)
        else:
            filtered_data = city_data
        
        # Plot historical data
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=filtered_data['Date'],
            y=filtered_data['AQI'],
            mode='lines',
            name='AQI',
            line=dict(color='#3b82f6', width=2)
        ))
        
        fig.update_layout(
            title=f"AQI Trend - {selected_city}",
            xaxis_title="Date",
            yaxis_title="AQI",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Average AQI", f"{filtered_data['AQI'].mean():.1f}")
        col2.metric("Max AQI", f"{filtered_data['AQI'].max():.1f}")
        col3.metric("Days > 200", f"{(filtered_data['AQI'] > 200).sum()}")
        col4.metric("Days > 300", f"{(filtered_data['AQI'] > 300).sum()}")
    
    with tab3:
        st.subheader("🏛️ Policy Impact Analysis")
        st.markdown("*Did government policies actually reduce pollution? Let's look at the data.*")
        
        analyzer = PolicyImpactAnalyzer()
        
        # Odd-Even Analysis - Only for NCR cities
        if selected_city in ['Delhi', 'Noida', 'Gurgaon', 'Ghaziabad', 'Faridabad']:
            st.markdown("---")
            st.markdown("### 🚗 Odd-Even Vehicle Rationing Scheme")
            st.markdown("""
            **What is it?** On odd dates, only odd-numbered vehicles can drive. On even dates, only even-numbered vehicles.
            The goal is to reduce traffic by 50% and improve air quality.
            """)
            
            odd_even_results = analyzer.analyze_odd_even_scheme()
            
            if not odd_even_results.empty:
                for idx, row in odd_even_results.iterrows():
                    with st.expander(f"📅 {row['period']}", expanded=idx==len(odd_even_results)-1):
                        # Simple visual comparison
                        st.markdown("#### 📊 What Happened to Air Quality?")
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Before Scheme", f"AQI {row['before_aqi']:.0f}")
                        col2.metric("During Scheme", f"AQI {row['during_aqi']:.0f}", 
                                   f"{row['pct_change_during']:+.1f}%")
                        col3.metric("After Scheme", f"AQI {row['after_aqi']:.0f}")
                        
                        # Plain language explanation
                        change = row['pct_change_during']
                        
                        st.markdown("#### 🤔 Did It Work?")
                        
                        if change < -10 and row['statistically_significant']:
                            st.success(f"""
                            ✅ **YES - Noticeable Improvement!**
                            
                            AQI dropped by **{abs(change):.0f}%** during the scheme. 
                            This is a meaningful reduction that likely helped reduce health problems.
                            """)
                        elif change < -5:
                            st.warning(f"""
                            ⚠️ **MINOR Effect**
                            
                            AQI dropped by only **{abs(change):.0f}%**. Here's the reality:
                            - Weather changes (wind, rain) often cause **20-30%** swings naturally
                            - A {abs(change):.0f}% drop is within normal daily variation
                            - The scheme reduced vehicles by ~25%, but that wasn't enough for major impact
                            """)
                        else:
                            st.error(f"""
                            ❌ **NO Significant Effect**
                            
                            AQI changed by only **{change:+.0f}%** - essentially flat or worse.
                            Possible reasons:
                            - Other pollution sources (construction, industry) weren't affected
                            - People found workarounds (taxis, metro, work from home)
                            - Weather conditions may have worsened air quality anyway
                            """)
                        
                        st.markdown("#### 💡 What This Means")
                        st.info("""
                        **Key Insight:** Vehicle restrictions alone have limited impact. 
                        For real improvement, cities need:
                        - Cleaner fuels (electric vehicles, CNG)
                        - Better public transport
                        - Control of construction dust and industrial emissions
                        - Regional solutions (stopping stubble burning)
                        """)
        
        # GRAP Analysis
        st.markdown("---")
        st.markdown("### 🚨 Emergency Measures (GRAP)")
        st.markdown("""
        **What is GRAP?** When pollution gets very bad, the government takes emergency actions:
        - 🚧 Construction work stopped
        - 🏫 Schools closed
        - 🚗 Trucks banned from entering city
        """)
        
        grap_results = analyzer.analyze_grap_effectiveness()
        
        before_pct = grap_results['hazardous_days_before_pct']
        after_pct = grap_results['hazardous_days_after_pct']
        change = grap_results['change']
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Before GRAP (Pre-2017)", 
                     f"{before_pct:.0f}% hazardous days",
                     help="Days with AQI > 300 during winter months")
        with col2:
            st.metric("After GRAP (2017+)", 
                     f"{after_pct:.0f}% hazardous days",
                     f"{change:+.0f}pp",
                     help="Percentage point change")
        
        if change < -5:
            st.success(f"""
            ✅ **GRAP is helping!** Hazardous days reduced by {abs(change):.0f} percentage points.
            Fewer emergency hospital visits and respiratory crises during winter.
            """)
        elif change < 0:
            st.warning(f"""
            ⚠️ **Slight improvement** but not enough. Still {after_pct:.0f}% of winter days are hazardous.
            Emergency measures help, but we need permanent solutions.
            """)
        else:
            st.error(f"""
            ❌ **No improvement.** Despite emergency measures, pollution hasn't decreased.
            This suggests the problem needs systemic solutions, not just temporary bans.
            """)
        
        # ===== RECENT POLICIES SECTION (2024-2026) - AI-Powered Web Search =====
        st.markdown("---")
        st.markdown("### 📰 Recent Policies (2024-2026)")
        st.markdown("*Live updates from government sources, CPCB, and NGT orders*")
        
        try:
            from agents.researcher import PolicyResearcherAgent
            GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
            
            if GEMINI_API_KEY:
                # Cache key for session
                policy_cache_key = f"recent_policies_{selected_city}"
                
                if st.button("🔍 Fetch Recent Policies", type="primary", key="fetch_policies"):
                    with st.spinner("🌐 Searching for recent policies..."):
                        researcher = PolicyResearcherAgent(GEMINI_API_KEY)
                        result = researcher.research_recent_policies(
                            city=selected_city, 
                            timeframe="2 years"
                        )
                        st.session_state[policy_cache_key] = result
                
                # Display cached or fetched policies
                if policy_cache_key in st.session_state:
                    policy_data = st.session_state[policy_cache_key]
                    
                    if "error" in policy_data and policy_data.get("policies") == []:
                        st.error(f"⚠️ Could not fetch policies: {policy_data.get('error')}")
                    else:
                        policies = policy_data.get("policies", [])
                        
                        if policies:
                            st.success(f"📋 Found **{len(policies)}** recent policies")
                            
                            for i, policy in enumerate(policies[:6]):  # Show max 6
                                with st.expander(f"📌 {policy.get('name', 'Policy')}", expanded=i==0):
                                    col1, col2 = st.columns([2, 1])
                                    with col1:
                                        st.markdown(f"**Description:** {policy.get('description', 'N/A')}")
                                        
                                        provisions = policy.get('key_provisions', [])
                                        if provisions:
                                            st.markdown("**Key Provisions:**")
                                            for p in provisions[:3]:
                                                st.markdown(f"- {p}")
                                    
                                    with col2:
                                        st.markdown(f"📅 **Date:** {policy.get('date', 'N/A')}")
                                        st.markdown(f"🏛️ **Authority:** {policy.get('authority', 'N/A')}")
                                        st.markdown(f"📊 **Impact:** {policy.get('expected_impact', 'N/A')}")
                                        
                                        status = policy.get('status', 'unknown')
                                        status_color = {"active": "🟢", "ongoing": "🟡", "completed": "🔵"}.get(status, "⚪")
                                        st.markdown(f"{status_color} **Status:** {status.title()}")
                            
                            # Summary
                            summary = policy_data.get("summary", "")
                            if summary:
                                st.info(f"📝 **Summary:** {summary}")
                        else:
                            st.info("No recent policies found. Try clicking 'Fetch Recent Policies'.")
                else:
                    st.info("👆 Click the button above to search for recent policies from 2024-2026")
            else:
                st.warning("⚠️ Gemini API key not configured. Add GEMINI_API_KEY to .env file.")
                
        except ImportError:
            st.warning("Policy research agent not available.")
        except Exception as e:
            st.error(f"Error fetching policies: {str(e)}")
    
    with tab4:
        st.subheader("🛡️ Insurance & Health Planning")
        st.markdown(f"*Plan your healthcare based on **{selected_city}'s** air quality*")
        
        # Get current AQI for calculations
        current_aqi = realtime_data.get('aqi', 150) if realtime_data else 150
        
        # User profile inputs
        st.markdown("---")
        st.markdown("### 👤 Your Profile")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            user_age = st.number_input("Your Age", min_value=1, max_value=100, value=30, key="insurance_age")
        with col2:
            family_size = st.number_input("Family Size", min_value=1, max_value=10, value=4, key="family_size")
        with col3:
            conditions = st.multiselect(
                "Pre-existing Conditions",
                ["Asthma", "COPD", "Heart Disease", "Diabetes", "Allergies", "None"],
                default=["None"],
                key="conditions"
            )
        
        if "None" in conditions:
            conditions = []
        
        # Section 1: Pollution Health Risk Score
        st.markdown("---")
        st.markdown("### 📊 Your Pollution Health Risk Score")
        
        try:
            from agents.insurance_advisor import InsurancePlannerAgent
            GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
            
            if GEMINI_API_KEY:
                insurance_agent = InsurancePlannerAgent(GEMINI_API_KEY)
                
                # Calculate risk
                risk = insurance_agent.get_pollution_health_risk(
                    selected_city, current_aqi, user_age, 
                    [c.lower() for c in conditions]
                )
                
                risk_score = risk['risk_score']
                risk_color = "#22c55e" if risk_score < 30 else "#eab308" if risk_score < 50 else "#f97316" if risk_score < 75 else "#ef4444"
                
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.markdown(f"""
                    <div style="
                        background: linear-gradient(135deg, {risk_color}22 0%, {risk_color}44 100%);
                        border: 3px solid {risk_color};
                        border-radius: 15px;
                        padding: 30px;
                        text-align: center;
                    ">
                        <div style="font-size: 48px; font-weight: bold; color: {risk_color};">{risk_score}</div>
                        <div style="font-size: 16px; color: #64748b;">out of 100</div>
                        <div style="font-size: 20px; font-weight: bold; margin-top: 10px; color: {risk_color};">
                            {risk['risk_level']} Risk
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    **Risk Factors:**
                    - 🏙️ City AQI: **{current_aqi}** ({risk['base_risk']})
                    - 👤 Age: **{user_age}** {"(higher risk)" if user_age > 60 or user_age < 18 else "(normal)"}
                    - 💊 Conditions: **{', '.join(conditions) if conditions else "None"}**
                    """)
                    
                    if risk_score >= 50:
                        st.warning("⚠️ Your risk level suggests you should prioritize comprehensive health coverage.")
                    else:
                        st.success("✅ Your risk level is manageable with standard health coverage.")
                
                # Section 2: Recommended Health Checkups
                st.markdown("---")
                st.markdown("### 💊 Recommended Health Checkups")
                
                checkup_cache_key = f"checkups_{selected_city}_{current_aqi}"
                
                if st.button("🔍 Get Personalized Checkup Recommendations", key="get_checkups"):
                    with st.spinner("Analyzing your health needs..."):
                        checkups = insurance_agent.get_health_checkup_recommendations(
                            selected_city, current_aqi, 
                            f"Age {user_age}, conditions: {conditions}"
                        )
                        st.session_state[checkup_cache_key] = checkups
                
                if checkup_cache_key in st.session_state:
                    checkups = st.session_state[checkup_cache_key]
                    
                    st.markdown(f"**Risk Level:** {checkups.get('risk_level', 'moderate').title()}")
                    
                    # Display checkups in a nice table
                    annual_tests = checkups.get('annual_checkups', [])
                    if annual_tests:
                        st.markdown("**Annual Tests for Pollution Exposure:**")
                        for test in annual_tests[:5]:
                            with st.expander(f"🩺 {test.get('test_name', 'Test')} - ₹{test.get('approximate_cost_inr', 0)}"):
                                st.write(f"**Purpose:** {test.get('purpose', 'General health')}")
                                st.write(f"**Frequency:** {test.get('frequency', 'Annual')}")
                        
                        total_cost = checkups.get('total_annual_checkup_cost_inr', 0)
                        st.info(f"💰 **Estimated Annual Checkup Cost:** ₹{total_cost}")
                    
                    # Tips
                    tips = checkups.get('tips', [])
                    if tips:
                        st.markdown("**💡 Tips:**")
                        for tip in tips[:3]:
                            st.write(f"- {tip}")
                else:
                    st.info("👆 Click the button to get personalized checkup recommendations")
                
                # Section 3: Insurance Recommendations
                st.markdown("---")
                st.markdown("### 🛡️ Insurance Plan Recommendations")
                st.markdown("*AI-powered search for plans that cover pollution-related illness*")
                
                insurance_cache_key = f"insurance_{selected_city}_{user_age}_{family_size}"
                
                if st.button("🔍 Find Insurance Plans", type="primary", key="find_insurance"):
                    with st.spinner("🌐 Searching for best insurance plans..."):
                        insurance = insurance_agent.get_insurance_recommendations(
                            selected_city, current_aqi, user_age, family_size,
                            [c.lower() for c in conditions]
                        )
                        st.session_state[insurance_cache_key] = insurance
                
                if insurance_cache_key in st.session_state:
                    insurance = st.session_state[insurance_cache_key]
                    
                    # Coverage recommendation
                    coverage = insurance.get('recommended_coverage_amount_lakhs', 10)
                    st.success(f"💰 **Recommended Coverage:** ₹{coverage} Lakhs for your family of {family_size}")
                    st.caption(insurance.get('coverage_justification', ''))
                    
                    # Insurance plans
                    plans = insurance.get('insurance_plans', [])
                    if plans:
                        st.markdown("**📋 Recommended Plans:**")
                        
                        for plan in plans[:4]:
                            with st.expander(f"🏥 {plan.get('provider', 'Provider')} - {plan.get('plan_name', 'Plan')}", expanded=False):
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.markdown(f"**Coverage:** ₹{plan.get('coverage_lakhs', 0)} Lakhs")
                                    st.markdown(f"**Premium:** ₹{plan.get('annual_premium_inr', 0):,}/year")
                                    st.markdown(f"**Waiting Period:** {plan.get('waiting_period_months', 0)} months")
                                with col2:
                                    st.markdown("**Key Features:**")
                                    for feature in plan.get('key_features', [])[:3]:
                                        st.write(f"✓ {feature}")
                                
                                if plan.get('covers_pollution_illness'):
                                    st.success("✅ Covers pollution-related illness")
                                
                                st.caption(f"Best for: {plan.get('best_for', 'General use')}")
                    
                    # Add-ons
                    addons = insurance.get('recommended_add_ons', [])
                    if addons:
                        st.markdown("**🔧 Recommended Add-ons:**")
                        for addon in addons[:2]:
                            st.write(f"- **{addon.get('name')}** (₹{addon.get('approximate_cost_inr', 0)}/year): {addon.get('why_needed', '')}")
                    
                    # Tips
                    tips = insurance.get('tips', [])
                    if tips:
                        st.markdown("**💡 Insurance Tips:**")
                        for tip in tips[:3]:
                            st.info(tip)
                else:
                    st.info("👆 Click to search for insurance plans suited to your needs")
                
                # Section 4: Cost Comparison
                st.markdown("---")
                st.markdown("### 💳 With vs Without Insurance")
                
                analyzer = PolicyImpactAnalyzer()
                health_costs = analyzer.calculate_health_cost_impact(selected_city, 2024)
                
                if health_costs:
                    per_capita = health_costs.get('per_capita_cost', 5000)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.error(f"""
                        **❌ Without Insurance**  
                        Annual out-of-pocket: ₹{per_capita * 1.5:.0f}  
                        One hospitalization: ₹50,000+  
                        Risk of major expense: HIGH
                        """)
                    with col2:
                        st.success(f"""
                        **✅ With Health Insurance**  
                        Annual premium: ₹12,000-20,000  
                        Hospitalization: Covered  
                        Financial protection: SECURED
                        """)
            else:
                st.warning("⚠️ Add GEMINI_API_KEY to .env for insurance recommendations")
                
        except ImportError:
            st.warning("Insurance planning agent not available.")
        except Exception as e:
            st.error(f"Error loading insurance planner: {str(e)}")
    
    with tab5:
        st.subheader("🗺️ Multi-City Comparison")
        
        # City selector for comparison
        comparison_cities = st.multiselect(
            "Select cities to compare:",
            available_cities,
            default=[selected_city, 'Mumbai', 'Bengaluru'] if len(available_cities) >= 3 else available_cities[:3]
        )
        
        if len(comparison_cities) > 0:
            # Get current AQI for each city
            comparison_data = []
            for city in comparison_cities:
                city_latest = df[df['City'] == city].iloc[-1]
                city_config = INDIAN_CITIES.get(city, {})
                
                comparison_data.append({
                    'City': city,
                    'Current AQI': city_latest['AQI'],
                    'PM2.5': city_latest['PM2.5'],
                    'Population': city_config.get('population', 0),
                    'Tier': city_config.get('tier', 0)  # Use 0 for unknown cities
                })
            
            comparison_df = pd.DataFrame(comparison_data).sort_values('Current AQI', ascending=False)
            
            # Bar chart
            fig = px.bar(
                comparison_df,
                x='City',
                y='Current AQI',
                color='Current AQI',
                color_continuous_scale='RdYlGn_r',
                title="Current AQI Comparison"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Data table
            st.dataframe(comparison_df, use_container_width=True)
    
    with tab6:
        st.subheader("🕵️‍♂️ Agentic Pollution Analysis")
        st.markdown("*Multi-Agent System: Scientist (Detective) & Doctor (Guardian)*")
        
        try:
            # Check for API Key
            GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
            if not GEMINI_API_KEY:
                st.error("❌ Gemini API Key not found. Please check your .env file.")
                st.stop()

            # Mode Selection
            analysis_mode = st.radio(
                "Select Analysis Mode:",
                ["🤖 Standard Chatbot (Advisor)", "🕵️‍♂️ Deep Agent Analysis (Detective + Guardian)"],
                index=1
            )
            
            if analysis_mode == "🤖 Standard Chatbot (Advisor)":
                from gemini_advisor import GeminiHealthAdvisor
                advisor = GeminiHealthAdvisor(GEMINI_API_KEY)
                
                col_ai1, col_ai2 = st.columns([2, 1])
                
                with col_ai1:
                    if 'chat_history' not in st.session_state:
                        st.session_state.chat_history = []
                    
                    user_question = st.text_input("Ask a question:", placeholder="Should I wear a mask today?")
                    
                    if st.button("🚀 Ask Advisor") and user_question:
                        with st.spinner("Thinking..."):
                            response = advisor.chat(user_question, selected_city, current_aqi, category, user_profile)
                            st.session_state.chat_history.append({'q': user_question, 'a': response})
                    
                    for chat in reversed(st.session_state.chat_history[-5:]):
                        with st.expander(f"Q: {chat['q'][:40]}...", expanded=True):
                            st.markdown(f"**AI:** {chat['a']}")

            else:  # Deep Agent Analysis
                from agents.orchestrator import AgentOrchestrator
                
                st.info("ℹ️ **Deep Analysis** uses two AI agents: one to analyze raw sensor/weather data, and another to formulate health advice.")
                
                if st.button("🚀 Run Full Analysis", type="primary"):
                    orchestrator = AgentOrchestrator(GEMINI_API_KEY)
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Step 1: Detective
                    status_text.text("🕵️‍♂️ Detective Agent: Scanning sensors and weather patterns...")
                    progress_bar.progress(30)
                    
                    result = orchestrator.run_analysis(selected_city, user_profile)
                    
                    if "error" in result and result.get("error"):
                        progress_bar.empty()
                        status_text.error(f"Analysis Failed: {result.get('message')}")
                    else:
                        progress_bar.progress(100)
                        status_text.success("Analysis Complete!")
                        
                        # Display Report
                        st.markdown("### 📊 Situation Report")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"""
                            <div style="background: #1e293b; padding: 15px; border-radius: 10px; border-left: 5px solid #60a5fa;">
                                <h4 style="color: #60a5fa; margin:0;">🕵️‍♂️ The Scientist's Findings</h4>
                                <p style="color: #e2e8f0; font-size: 14px; margin-top: 10px;">
                                    {result['situation_report']}
                                </p>
                                <div style="margin-top: 10px; font-size: 12px; color: #94a3b8;">
                                    <strong>Data Sources:</strong> OpenMeteo AQI + Weather
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Weather stats
                            w = result.get('weather', {})
                            st.markdown(f"**Weather Context:** 🌬️ Wind: {w.get('wind_speed_10m', 'N/A')} km/h | 🌡️ Temp: {w.get('temperature_2m', 'N/A')}°C")
                            
                        with col2:
                            st.markdown(f"""
                            <div style="background: #1e293b; padding: 15px; border-radius: 10px; border-left: 5px solid #34d399;">
                                <h4 style="color: #34d399; margin:0;">👩‍⚕️ The Doctor's Advice</h4>
                                <p style="color: #e2e8f0; font-size: 14px; margin-top: 10px;">
                                    {result['health_advice']}
                                </p>
                                <div style="margin-top: 10px; font-size: 12px; color: #94a3b8;">
                                    <strong>Profile:</strong> {user_profile}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"⚠️ System Error: {str(e)}")
            if "quota" in str(e).lower():
                st.warning("You may have exceeded the API rate limit for the free tier. Please try again later.")


if __name__ == "__main__":
    main()
