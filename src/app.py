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
from location_detector import find_nearest_city, CITY_COORDINATES

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
    page_title="Air Quality Monitor",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Material Design CSS
st.markdown("""
    <style>
    /* Import Google Fonts - Roboto (Material Design default) */
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
    
    /* Base styling */
    .main {
        background: #fafafa;
        font-family: 'Roboto', sans-serif;
    }
    
    /* Material Design Cards */
    .md-card {
        background: #ffffff;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1), 0 1px 2px rgba(0,0,0,0.06);
        padding: 24px;
        margin: 16px 0;
        transition: box-shadow 0.2s ease;
    }
    
    .md-card:hover {
        box-shadow: 0 4px 8px rgba(0,0,0,0.12), 0 2px 4px rgba(0,0,0,0.08);
    }
    
    /* AQI Display - Clean Material Style */
    .aqi-display {
        font-size: 64px;
        font-weight: 300;
        text-align: center;
        padding: 32px;
        border-radius: 8px;
        margin: 16px 0;
        font-family: 'Roboto', sans-serif;
    }
    
    .aqi-label {
        font-size: 14px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 8px;
    }
    
    /* Health Alert - Material Design */
    .health-card {
        padding: 24px;
        border-radius: 8px;
        margin: 16px 0;
        background: #ffffff;
        border-left: 4px solid;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    
    .health-card h3 {
        font-weight: 500;
        font-size: 18px;
        margin-bottom: 12px;
        color: #212121;
    }
    
    .health-card p {
        color: #616161;
        font-size: 14px;
        line-height: 1.6;
    }
    
    .health-card ul {
        margin: 12px 0;
        padding-left: 20px;
    }
    
    .health-card li {
        color: #424242;
        margin: 8px 0;
        font-size: 14px;
    }
    
    /* Section headers */
    .section-title {
        font-size: 20px;
        font-weight: 500;
        color: #212121;
        margin: 24px 0 16px 0;
        padding-bottom: 8px;
        border-bottom: 1px solid #e0e0e0;
    }
    
    /* City cards for comparison */
    .city-card {
        background: #ffffff;
        border-radius: 8px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .city-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    .city-card.current {
        border: 2px solid #1976d2;
    }
    
    .city-name {
        font-size: 16px;
        font-weight: 500;
        color: #212121;
        margin-bottom: 8px;
    }
    
    .city-aqi {
        font-size: 36px;
        font-weight: 300;
    }
    
    .city-label {
        font-size: 12px;
        color: #757575;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Ranking list item */
    .rank-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 12px 16px;
        margin: 4px 0;
        background: #ffffff;
        border-radius: 4px;
        transition: background 0.15s ease;
    }
    
    .rank-item:hover {
        background: #f5f5f5;
    }
    
    .rank-item.current {
        background: #e3f2fd;
        border-left: 3px solid #1976d2;
    }
    
    /* Chips/Tags */
    .md-chip {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 12px;
        font-weight: 500;
        margin: 4px;
    }
    
    /* Status colors - Material palette */
    .status-good { background: #e8f5e9; color: #2e7d32; }
    .status-moderate { background: #fff8e1; color: #f57f17; }
    .status-poor { background: #fff3e0; color: #e65100; }
    .status-severe { background: #ffebee; color: #c62828; }
    
    /* Dark mode support */
    @media (prefers-color-scheme: dark) {
        .md-card, .city-card, .rank-item {
            background: #1e1e1e;
            color: #e0e0e0;
        }
        .health-card {
            background: #1e1e1e;
        }
        .health-card h3 {
            color: #ffffff;
        }
        .health-card p, .health-card li {
            color: #b0b0b0;
        }
        .section-title {
            color: #ffffff;
            border-bottom-color: #424242;
        }
        .city-name {
            color: #ffffff;
        }
        .city-label {
            color: #9e9e9e;
        }
        .rank-item:hover {
            background: #2c2c2c;
        }
        .rank-item.current {
            background: #1a237e;
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
    st.title("Air Quality Monitor")
    st.caption("Real-time air quality tracking and health recommendations for India")
    
    # Load data
    df = load_historical_data()
    
    if df.empty:
        st.error("No data available. Please run `python src/generate_data.py` first.")
        return
    
    available_cities = sorted(df['City'].unique())
    
    # ===== SIDEBAR =====
    with st.sidebar:
        st.header("Settings")
        
        # Auto-detect location on first load using IP geolocation
        if 'auto_detected_city' not in st.session_state:
            try:
                import requests
                # Use ipapi.co free API for IP-based location
                response = requests.get('https://ipapi.co/json/', timeout=5)
                if response.status_code == 200:
                    ip_data = response.json()
                    detected_city_name = ip_data.get('city', '')
                    
                    # Match to available cities
                    matched_city = None
                    for city in available_cities:
                        if city.lower() in detected_city_name.lower() or detected_city_name.lower() in city.lower():
                            matched_city = city
                            break
                    
                    # If no match, find nearest using lat/lon
                    if not matched_city and ip_data.get('latitude') and ip_data.get('longitude'):
                        matched_city, _ = find_nearest_city(
                            ip_data.get('latitude'), 
                            ip_data.get('longitude'), 
                            available_cities
                        )
                    
                    st.session_state.auto_detected_city = matched_city or 'Delhi'
                else:
                    st.session_state.auto_detected_city = 'Delhi'
            except:
                st.session_state.auto_detected_city = 'Delhi'
        
        # Location Selection - Simple dropdown with auto-detected default
        st.subheader("Location")
        
        selected_city = st.selectbox(
            "Your City",
            available_cities,
            index=available_cities.index(st.session_state.auto_detected_city) if st.session_state.auto_detected_city in available_cities else 0,
            key="city_selector",
            help="Auto-detected based on your location. Change if needed."
        )
        
        st.markdown("---")
        
        # User Profile Selection
        st.subheader("Profile")
        user_profile = st.selectbox(
            "Select your profile:",
            options=list(HealthAdvisor.PROFILES.keys()),
            format_func=lambda x: HealthAdvisor.PROFILES[x]['name']
        )
        
        profile_desc = HealthAdvisor.PROFILES[user_profile]['description']
        st.caption(profile_desc)
        
        st.markdown("---")
        
        # Planned Activity
        st.subheader("Activity")
        activity_options = ["jog", "cycling", "outdoor_event", "commute", "children_play", "window_open", "Other (Custom)"]
        selected_activity = st.selectbox(
            "What are you planning?",
            activity_options
        )
        
        if selected_activity == "Other (Custom)":
            planned_activity = st.text_input("Enter your activity:", placeholder="e.g., yoga in park")
            if not planned_activity:
                planned_activity = "general_outdoor_activity"
        else:
            planned_activity = selected_activity
        
        st.markdown("---")
        st.caption("Health decisions based on air quality forecasts")
    
    # ===== MAIN CONTENT =====
    
    # Real-Time AQI Section
    st.markdown(f"### Current Air Quality in {selected_city}")
    
    col1, col2 = st.columns([2, 1])
    
    # Always create city_data for historical features (used in tabs)
    city_data = df[df['City'] == selected_city].copy()
    
    # Safety check for empty data
    if city_data.empty:
        st.warning(f"No historical data available for {selected_city} in the dataset.")
        st.info("Try selecting a different city from the dropdown.")
        return
    
    # Try to get REAL-TIME data from Open-Meteo API
    realtime_data = get_realtime_aqi(selected_city)
    
    # Also fetch weather data
    weather_data = None
    if OPENWEATHERMAP_AVAILABLE and OPENWEATHERMAP_API_KEY:
        try:
            from openweathermap_client import OpenWeatherMapAQIClient
            weather_client = OpenWeatherMapAQIClient(OPENWEATHERMAP_API_KEY)
            weather_data = weather_client.get_current_weather(selected_city)
        except:
            pass
    
    if realtime_data and realtime_data.get('aqi'):
        # Use REAL-TIME API data
        current_aqi = realtime_data['aqi']
        category = realtime_data.get('category', 'Unknown')
        color = realtime_data.get('color', '#888888')
        # Show actual data source name
        api_source = realtime_data.get('source', 'OpenWeatherMap')
        timestamp = realtime_data.get('timestamp', 'now')[:16]
        data_source = f"Live from {api_source} • {timestamp}"
        pm25_val = realtime_data.get('pm25', 'N/A')
        pm10_val = realtime_data.get('pm10', 'N/A')
        no2_val = realtime_data.get('no2', 'N/A')
    else:
        # Fallback to historical data
        latest_data = city_data.iloc[-1]
        current_aqi = latest_data['AQI']
        category, color, health_impact = get_aqi_category(current_aqi)
        data_source = f"Historical data ({latest_data['Date'].strftime('%Y-%m-%d')})"
        pm25_val = f"{latest_data['PM2.5']:.1f}"
        pm10_val = f"{latest_data['PM10']:.1f}"
        no2_val = f"{latest_data['NO2']:.1f}"
    
    # Weather defaults if not available
    temp = weather_data.get('temperature', '--') if weather_data else '--'
    feels_like = weather_data.get('feels_like', '--') if weather_data else '--'
    humidity = weather_data.get('humidity', '--') if weather_data else '--'
    condition = weather_data.get('description', 'Loading...') if weather_data else 'N/A'
    weather_icon = weather_data.get('icon', '🌤️') if weather_data else '🌤️'
    wind_speed = weather_data.get('wind_speed', '--') if weather_data else '--'
    
    # ===== AQI CARD (Simple, reliable HTML) =====
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {color} 0%, {color}dd 100%); padding: 30px; border-radius: 16px; text-align: center; box-shadow: 0 4px 20px rgba(0,0,0,0.15); margin-bottom: 16px; color: white;">
        <div style="font-size: 14px; opacity: 0.8; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 8px;">Air Quality Index</div>
        <div style="font-size: 72px; font-weight: 700; line-height: 1; margin-bottom: 8px;">{int(current_aqi)}</div>
        <div style="display: inline-block; background: rgba(255,255,255,0.2); padding: 8px 20px; border-radius: 20px; font-size: 18px; font-weight: 600;">{category}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # ===== POLLUTANTS (Using Streamlit Native Metrics) =====
    pm_col1, pm_col2, pm_col3 = st.columns(3)
    with pm_col1:
        st.metric(
            label="PM2.5",
            value=f"{pm25_val} µg/m³",
            help="Fine particles (<2.5µm) - Enters lungs & blood. Main cause of respiratory issues."
        )
    with pm_col2:
        st.metric(
            label="PM10", 
            value=f"{pm10_val} µg/m³",
            help="Coarse particles (<10µm) - Dust, pollen, mold. Irritates airways."
        )
    with pm_col3:
        st.metric(
            label="NO₂",
            value=f"{no2_val} µg/m³",
            help="Nitrogen Dioxide - From vehicles & power plants. Causes breathing problems."
        )
    
    # ===== WEATHER STRIP (Compact Horizontal) =====
    st.markdown(f"""
    <div style="
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 16px 24px;
        border-radius: 12px;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 16px;
        color: white;
    ">
        <div style="display: flex; align-items: center; gap: 12px;">
            <span style="font-size: 36px;">{weather_icon}</span>
            <div>
                <div style="font-size: 28px; font-weight: 700;">{temp}°C</div>
                <div style="font-size: 12px; opacity: 0.8;">{condition}</div>
            </div>
        </div>
        <div style="display: flex; gap: 24px; font-size: 13px;">
            <div style="text-align: center;">
                <div style="opacity: 0.7;">Feels</div>
                <div style="font-weight: 600;">{feels_like}°C</div>
            </div>
            <div style="text-align: center;">
                <div style="opacity: 0.7;">Humidity</div>
                <div style="font-weight: 600;">{humidity}%</div>
            </div>
            <div style="text-align: center;">
                <div style="opacity: 0.7;">Wind</div>
                <div style="font-weight: 600;">{wind_speed} km/h</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Data source caption
    st.caption(f"📡 {data_source}")
    
    # AQI Scale Reference - Shows what's normal
    with st.expander("📊 AQI Scale Reference - What's Normal?", expanded=False):
        st.markdown("""
        <div style="display: flex; gap: 4px; margin: 10px 0;">
            <div style="flex: 1; background: #4caf50; color: white; padding: 10px 5px; text-align: center; border-radius: 4px 0 0 4px;">
                <div style="font-size: 11px;">Good</div>
                <div style="font-weight: 600;">0-50</div>
            </div>
            <div style="flex: 1; background: #8bc34a; color: white; padding: 10px 5px; text-align: center;">
                <div style="font-size: 11px;">Satisfactory</div>
                <div style="font-weight: 600;">51-100</div>
            </div>
            <div style="flex: 1; background: #ff9800; color: white; padding: 10px 5px; text-align: center;">
                <div style="font-size: 11px;">Moderate</div>
                <div style="font-weight: 600;">101-200</div>
            </div>
            <div style="flex: 1; background: #ff5722; color: white; padding: 10px 5px; text-align: center;">
                <div style="font-size: 11px;">Poor</div>
                <div style="font-weight: 600;">201-300</div>
            </div>
            <div style="flex: 1; background: #9c27b0; color: white; padding: 10px 5px; text-align: center;">
                <div style="font-size: 11px;">Very Poor</div>
                <div style="font-weight: 600;">301-400</div>
            </div>
            <div style="flex: 1; background: #c62828; color: white; padding: 10px 5px; text-align: center; border-radius: 0 4px 4px 0;">
                <div style="font-size: 11px;">Severe</div>
                <div style="font-weight: 600;">400+</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        | Level | AQI Range | PM2.5 (µg/m³) | PM10 (µg/m³) | Health Impact |
        |-------|-----------|---------------|--------------|---------------|
        | **Good** | 0-50 | 0-30 | 0-50 | ✅ No health risk |
        | **Satisfactory** | 51-100 | 31-60 | 51-100 | Minor issues for sensitive people |
        | **Moderate** | 101-200 | 61-90 | 101-250 | Breathing problems for some |
        | **Poor** | 201-300 | 91-120 | 251-350 | ⚠️ Affects everyone |
        | **Very Poor** | 301-400 | 121-250 | 351-430 | 🚨 Serious health effects |
        | **Severe** | 400+ | 250+ | 430+ | ☠️ Emergency conditions |
        
        *Based on Indian National Air Quality Index (NAQI) by CPCB*
        """)
    
    # ===== CREATIVE QUICK ACCESS NAVIGATION =====
    st.markdown("""
    <style>
        .quick-nav {
            display: flex;
            gap: 12px;
            justify-content: center;
            flex-wrap: wrap;
            padding: 20px 0;
            margin: 20px 0;
        }
        .nav-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 16px;
            padding: 16px 20px;
            min-width: 120px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        }
        .nav-card:hover {
            transform: translateY(-5px) scale(1.05);
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.5);
        }
        .nav-card.forecast { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); box-shadow: 0 4px 15px rgba(17, 153, 142, 0.3); }
        .nav-card.trends { background: linear-gradient(135deg, #ee0979 0%, #ff6a00 100%); box-shadow: 0 4px 15px rgba(238, 9, 121, 0.3); }
        .nav-card.policy { background: linear-gradient(135deg, #8E2DE2 0%, #4A00E0 100%); box-shadow: 0 4px 15px rgba(142, 45, 226, 0.3); }
        .nav-card.health { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); box-shadow: 0 4px 15px rgba(240, 147, 251, 0.3); }
        .nav-card.compare { background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); box-shadow: 0 4px 15px rgba(79, 172, 254, 0.3); }
        .nav-card.ai { background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); box-shadow: 0 4px 15px rgba(250, 112, 154, 0.3); }
        .nav-icon { font-size: 28px; margin-bottom: 6px; }
        .nav-label { color: white; font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(102, 126, 234, 0.7); }
            70% { box-shadow: 0 0 0 10px rgba(102, 126, 234, 0); }
            100% { box-shadow: 0 0 0 0 rgba(102, 126, 234, 0); }
        }
        .nav-hint {
            text-align: center;
            color: #888;
            font-size: 13px;
            margin-bottom: 10px;
            animation: fadeIn 1s ease;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-10px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
    
    <div class="nav-hint">
        ⚡ Quick Access — scroll down to explore
    </div>
    
    <div class="quick-nav">
        <div class="nav-card forecast">
            <div class="nav-icon">🌤️</div>
            <div class="nav-label">7-Day<br>Forecast</div>
        </div>
        <div class="nav-card trends">
            <div class="nav-icon">📈</div>
            <div class="nav-label">Historical<br>Trends</div>
        </div>
        <div class="nav-card policy">
            <div class="nav-icon">🏛️</div>
            <div class="nav-label">Policy<br>Impact</div>
        </div>
        <div class="nav-card health">
            <div class="nav-icon">❤️</div>
            <div class="nav-label">Health<br>Planning</div>
        </div>
        <div class="nav-card compare">
            <div class="nav-icon">🏙️</div>
            <div class="nav-label">Compare<br>Cities</div>
        </div>
        <div class="nav-card ai">
            <div class="nav-icon">🤖</div>
            <div class="nav-label">AI<br>Advisor</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Personalized Health Recommendation
    recommendation = HealthAdvisor.get_recommendation(current_aqi, user_profile)
    
    # Remove emoji from recommendation message
    rec_message = recommendation['message']
    
    st.markdown(f"""
    <div class="health-card" style="border-color: {recommendation['color']};">
        <h3>{rec_message}</h3>
        <p><strong>Health Impact:</strong> {recommendation['health_impact']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Dynamic Activity Suggestions (Gemini-powered)
    st.markdown("### Activity Suggestions")
    st.caption("AI-powered recommendations based on current air quality")
    
    try:
        from gemini_advisor import GeminiHealthAdvisor
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        
        # Cache the suggestions in session state to avoid repeated API calls
        cache_key = f"suggestions_{selected_city}_{int(current_aqi)}_{user_profile}"
        
        if cache_key not in st.session_state or st.button("Refresh", key="refresh_suggestions"):
            with st.spinner("Getting AI suggestions..."):
                advisor = GeminiHealthAdvisor(GEMINI_API_KEY)
                st.session_state[cache_key] = advisor.get_dynamic_activity_suggestions(
                    city=selected_city,
                    aqi=current_aqi,
                    category=category,
                    user_profile=user_profile
                )
        
        suggestions_data = st.session_state[cache_key]
        
        # Display activity cards in a grid with Material Design
        cols = st.columns(4)
        safety_colors = {"safe": "#4caf50", "caution": "#ff9800", "avoid": "#f44336"}
        
        for i, sugg in enumerate(suggestions_data.get("suggestions", [])[:4]):
            with cols[i]:
                safety = sugg.get("safety", "caution")
                color = safety_colors.get(safety, "#757575")
                st.markdown(f"""
                <div class="city-card" style="border-left: 4px solid {color};">
                    <div class="city-name">{sugg.get('activity', 'Activity')}</div>
                    <div style="font-size: 13px; color: #616161; margin: 8px 0;">
                        {sugg.get('tip', '')}
                    </div>
                    <div class="md-chip" style="background: {color}20; color: {color};">
                        {safety.upper()}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Best time and general tip
        col_time, col_tip = st.columns(2)
        with col_time:
            st.info(f"**Best Time:** {suggestions_data.get('best_time', 'Check forecast')}")
        with col_tip:
            st.success(f"**Tip:** {suggestions_data.get('general_tip', 'Stay healthy!')}")
            
    except Exception as e:
        # Fallback to static suggestions if Gemini fails
        st.warning("AI suggestions unavailable. Showing default recommendations.")
        if current_aqi > 200:
            st.info("**Recommendation:** Stay indoors and use air purifiers.")
        else:
            st.info("**Recommendation:** Outdoor activities safe with normal precautions.")
    
    # ===== PROTECTION IMPACT SECTION =====
    # Show real-world effect of precautions to educate users
    if current_aqi > 100:
        st.markdown("### Protection Impact")
        st.caption("How precautions reduce your actual pollution exposure")
        
        # Calculate effective indoor AQI with air purifier
        # HEPA air purifiers reduce PM2.5 by 70-90%
        purifier_reduction = 0.80  # 80% reduction
        indoor_with_purifier = max(current_aqi * (1 - purifier_reduction), 20)
        
        # N95 mask reduces PM2.5 by 95% when properly fitted
        mask_reduction = 0.95
        effective_with_mask = max(current_aqi * (1 - mask_reduction), 10)
        
        # Get categories for comparison
        def get_simple_category(aqi):
            if aqi <= 50: return "Good", "#4caf50"
            elif aqi <= 100: return "Satisfactory", "#8bc34a"
            elif aqi <= 200: return "Moderate", "#ff9800"
            elif aqi <= 300: return "Poor", "#ff5722"
            else: return "Severe", "#f44336"
        
        outdoor_cat, outdoor_color = get_simple_category(current_aqi)
        purifier_cat, purifier_color = get_simple_category(indoor_with_purifier)
        mask_cat, mask_color = get_simple_category(effective_with_mask)
        
        # Display comparison cards
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div style="background: #ffebee; padding: 16px; border-radius: 8px; text-align: center; border: 2px solid {outdoor_color};">
                <div style="font-size: 12px; color: #757575; margin-bottom: 4px;">OUTDOOR (No Protection)</div>
                <div style="font-size: 32px; font-weight: bold; color: {outdoor_color};">{int(current_aqi)}</div>
                <div style="font-size: 14px; color: {outdoor_color}; font-weight: 500;">{outdoor_cat}</div>
                <div style="font-size: 11px; color: #9e9e9e; margin-top: 8px;">Direct exposure to pollution</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="background: #e8f5e9; padding: 16px; border-radius: 8px; text-align: center; border: 2px solid {purifier_color};">
                <div style="font-size: 12px; color: #757575; margin-bottom: 4px;">INDOOR + AIR PURIFIER</div>
                <div style="font-size: 32px; font-weight: bold; color: {purifier_color};">{int(indoor_with_purifier)}</div>
                <div style="font-size: 14px; color: {purifier_color}; font-weight: 500;">{purifier_cat}</div>
                <div style="font-size: 11px; color: #4caf50; margin-top: 8px;">80% reduction with HEPA filter</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div style="background: #e3f2fd; padding: 16px; border-radius: 8px; text-align: center; border: 2px solid {mask_color};">
                <div style="font-size: 12px; color: #757575; margin-bottom: 4px;">OUTDOOR + N95 MASK</div>
                <div style="font-size: 32px; font-weight: bold; color: {mask_color};">{int(effective_with_mask)}</div>
                <div style="font-size: 14px; color: {mask_color}; font-weight: 500;">{mask_cat}</div>
                <div style="font-size: 11px; color: #1976d2; margin-top: 8px;">95% filtered with proper fit</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Additional context
        st.markdown("""
        <div style="background: #f5f5f5; padding: 12px 16px; border-radius: 8px; margin-top: 12px;">
            <div style="font-weight: 500; color: #424242; margin-bottom: 8px;">What This Means For You</div>
            <div style="font-size: 13px; color: #616161; line-height: 1.6;">
                <strong>HEPA Air Purifier:</strong> Running a quality air purifier at home can reduce indoor PM2.5 by 70-90%. 
                Keep doors/windows closed for best effect. Clean or replace filters regularly.<br><br>
                <strong>N95 Mask:</strong> When properly fitted (no gaps around nose/chin), N95 masks filter 95% of fine particles. 
                Essential when going outside during high pollution days. Replace daily during severe pollution.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Tabs for different analysis
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Forecast",
        "Trends",
        "Policy Impact",
        "Health Planning",
        "Compare Cities",
        "AI Advisor"
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
        st.markdown(f"### Activity Guidance: {planned_activity.replace('_', ' ').title()}")
        activity_guidance = HealthAdvisor.get_activity_guidance(
            forecast_data['aqi'],
            planned_activity,
            user_profile
        )
        
        # Status-based styling - Material Design colors, no emojis
        status_styles = {
            "safe": {"color": "#4caf50", "bg": "#e8f5e9", "label": "SAFE"},
            "caution": {"color": "#ff9800", "bg": "#fff3e0", "label": "CAUTION"},
            "risky": {"color": "#ff5722", "bg": "#fbe9e7", "label": "RISKY"},
            "dangerous": {"color": "#f44336", "bg": "#ffebee", "label": "NOT RECOMMENDED"}
        }
        status = activity_guidance.get('status', 'caution')
        style = status_styles.get(status, status_styles['caution'])
        
        # Main recommendation card - Material Design
        st.markdown(f"""
        <div style="
            background: {style['bg']};
            border-left: 4px solid {style['color']};
            padding: 16px 20px;
            border-radius: 8px;
            margin-bottom: 16px;
        ">
            <div style="
                display: inline-block;
                background: {style['color']};
                color: white;
                font-size: 11px;
                font-weight: 500;
                padding: 4px 10px;
                border-radius: 4px;
                margin-bottom: 8px;
            ">{style['label']}</div>
            <div style="font-size: 16px; font-weight: 500; color: #212121; margin-bottom: 12px;">
                {activity_guidance['recommendation']}
            </div>
            <div style="display: flex; gap: 24px; flex-wrap: wrap; font-size: 13px; color: #616161;">
                <span><strong>AQI Range:</strong> {activity_guidance['min_aqi']:.0f} - {activity_guidance['max_aqi']:.0f}</span>
                <span><strong>Average:</strong> {activity_guidance['avg_aqi']:.1f}</span>
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

                    response = advisor.client.models.generate_content(
                        model=advisor.model_name,
                        contents=prompt
                    )
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
                st.markdown("#### Timing & Duration")
                st.markdown(f"""
                <div style="background: #f5f5f5; padding: 12px; border-radius: 8px; margin-bottom: 10px;">
                    <div style="color: #757575; font-size: 12px;">Best Time</div>
                    <div style="color: #212121; font-weight: 500;">{guidance_data.get('best_timing', 'Early morning')}</div>
                </div>
                <div style="background: #f5f5f5; padding: 12px; border-radius: 8px;">
                    <div style="color: #757575; font-size: 12px;">Duration Advice</div>
                    <div style="color: #212121; font-weight: 500;">{guidance_data.get('duration_advice', 'Limit outdoor time')}</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("#### Forecast Insight")
                st.info(guidance_data.get('forecast_insight', 'AQI levels remain elevated throughout the week.'))
            
            with col2:
                st.markdown("#### Better Alternatives")
                for alt in guidance_data.get('alternatives', ['Indoor gym', 'Home workout', 'Yoga'])[:3]:
                    st.markdown(f"""
                    <div style="background: #e8f5e9; padding: 8px 12px; border-radius: 6px; margin-bottom: 6px; border-left: 3px solid #4caf50;">
                        <span style="color: #2e7d32;">✓ {alt}</span>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("#### Safety Tips")
                for tip in guidance_data.get('safety_tips', ['Wear N95 mask', 'Stay hydrated'])[:2]:
                    st.markdown(f"- {tip}")
                    
        except Exception as e:
            # Activity-specific fallback guidance
            activity_alternatives = {
                "jog": ["Treadmill running at gym", "Indoor cycling", "Jump rope at home"],
                "cycling": ["Stationary bike", "Indoor spinning class", "Swimming"],
                "outdoor_event": ["Move event indoors", "Virtual event option", "Reschedule to better AQI day"],
                "commute": ["Work from home if possible", "Use AC vehicle with recirculation", "Avoid peak traffic hours"],
                "children_play": ["Indoor play area", "Board games at home", "Kids fitness videos"],
                "window_open": ["Use air purifier instead", "Ventilate during low-traffic hours", "Use exhaust fans in kitchen/bathroom"]
            }
            
            activity_tips = {
                "jog": ["Hydrate well before and after", "Monitor breathing difficulty"],
                "cycling": ["Avoid main roads with heavy traffic", "Wear fitted N95 mask"],
                "outdoor_event": ["Provide masks for attendees", "Set up air purifiers if indoors"],
                "commute": ["Keep car windows up, AC on recirculate", "Change route to avoid congestion"],
                "children_play": ["Children are more vulnerable - prioritize indoor play", "Watch for coughing or wheezing"],
                "window_open": ["Seal gaps around windows and doors", "Run purifier on high when cooking"]
            }
            
            alternatives = activity_alternatives.get(planned_activity, ["Indoor gym", "Home yoga", "Swimming"])
            tips = activity_tips.get(planned_activity, ["Wear N95 mask outdoors", "Stay hydrated"])
            
            st.markdown("#### Quick Tips")
            if status == "dangerous":
                st.error(f"**Strongly advised to stay indoors.** If you must go outside, wear an N95 mask and limit exposure to under 15 minutes.")
            elif status == "risky":
                st.warning(f"**Consider postponing or switching to indoor alternatives.** If proceeding, choose early morning (6-7 AM) when pollution is typically lower.")
            else:
                st.success("**Conditions are acceptable.** Monitor how you feel and reduce intensity if you experience any discomfort.")
            
            st.markdown(f"**Alternatives for {planned_activity.replace('_', ' ')}:** {', '.join(alternatives)}")
            st.markdown("**Safety Tips:**")
            for tip in tips:
                st.markdown(f"- {tip}")
    
    with tab2:
        st.subheader("Historical Pollution Trends")
        
        # Controls row
        col_time, col_metric = st.columns([2, 1])
        
        with col_time:
            time_range = st.select_slider(
                "Time Range",
                options=['Last 30 Days', 'Last 90 Days', 'Last 6 Months', 'Last Year', 'All Time'],
                value='Last 6 Months'
            )
        
        with col_metric:
            # Pollutant selector
            available_metrics = {
                'AQI': {'color': '#1976d2', 'unit': ''},
                'PM2.5': {'color': '#f44336', 'unit': 'µg/m³'},
                'PM10': {'color': '#ff9800', 'unit': 'µg/m³'},
                'NO2': {'color': '#9c27b0', 'unit': 'µg/m³'},
                'SO2': {'color': '#4caf50', 'unit': 'µg/m³'},
                'CO': {'color': '#795548', 'unit': 'mg/m³'}
            }
            
            selected_metrics = st.multiselect(
                "Pollutants to Display",
                options=list(available_metrics.keys()),
                default=['AQI'],
                help="Select one or more pollutants to compare"
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
        
        # Plot historical data with selected metrics
        if selected_metrics:
            fig = go.Figure()
            
            for metric in selected_metrics:
                if metric in filtered_data.columns:
                    fig.add_trace(go.Scatter(
                        x=filtered_data['Date'],
                        y=filtered_data[metric],
                        mode='lines',
                        name=metric,
                        line=dict(color=available_metrics[metric]['color'], width=2)
                    ))
            
            # Update layout
            y_axis_title = "Value"
            if len(selected_metrics) == 1:
                metric = selected_metrics[0]
                unit = available_metrics[metric]['unit']
                y_axis_title = f"{metric} ({unit})" if unit else metric
            
            fig.update_layout(
                title=f"Pollution Trends - {selected_city}",
                xaxis_title="Date",
                yaxis_title=y_axis_title,
                height=400,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                hovermode='x unified'
            )
            
            # Add AQI threshold lines if AQI is selected
            if 'AQI' in selected_metrics and len(selected_metrics) == 1:
                fig.add_hline(y=100, line_dash="dash", line_color="#4caf50", annotation_text="Satisfactory")
                fig.add_hline(y=200, line_dash="dash", line_color="#ff9800", annotation_text="Moderate")
                fig.add_hline(y=300, line_dash="dash", line_color="#f44336", annotation_text="Poor")
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistics for selected metrics
            st.markdown("### Statistics")
            stat_cols = st.columns(len(selected_metrics))
            
            for i, metric in enumerate(selected_metrics):
                if metric in filtered_data.columns:
                    with stat_cols[i]:
                        unit = available_metrics[metric]['unit']
                        avg_val = filtered_data[metric].mean()
                        max_val = filtered_data[metric].max()
                        min_val = filtered_data[metric].min()
                        
                        st.markdown(f"""
                        <div style="background: #f5f5f5; padding: 16px; border-radius: 8px; border-left: 4px solid {available_metrics[metric]['color']};">
                            <div style="font-size: 14px; font-weight: 500; color: #424242; margin-bottom: 8px;">{metric}</div>
                            <div style="display: flex; justify-content: space-between; font-size: 13px; color: #616161;">
                                <span>Avg: <strong>{avg_val:.1f}</strong></span>
                                <span>Max: <strong>{max_val:.1f}</strong></span>
                                <span>Min: <strong>{min_val:.1f}</strong></span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Additional insights for AQI
            if 'AQI' in selected_metrics:
                st.markdown("### Air Quality Summary")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Average AQI", f"{filtered_data['AQI'].mean():.1f}")
                col2.metric("Peak AQI", f"{filtered_data['AQI'].max():.1f}")
                col3.metric("Days > 200 (Poor)", f"{(filtered_data['AQI'] > 200).sum()}")
                col4.metric("Days > 300 (Very Poor)", f"{(filtered_data['AQI'] > 300).sum()}")
        else:
            st.info("Select at least one pollutant to view the trend.")
    
    with tab3:
        st.subheader("Policy Impact Analysis")
        st.markdown("*Did government policies actually reduce pollution? Let's look at the data.*")
        
        analyzer = PolicyImpactAnalyzer()
        
        # Odd-Even Analysis - Only for NCR cities
        if selected_city in ['Delhi', 'Noida', 'Gurgaon', 'Ghaziabad', 'Faridabad']:
            st.markdown("---")
            st.markdown("### Odd-Even Vehicle Rationing Scheme")
            st.markdown("""
            **What is it?** On odd dates, only odd-numbered vehicles can drive. On even dates, only even-numbered vehicles.
            The goal is to reduce traffic by 50% and improve air quality.
            """)
            
            odd_even_results = analyzer.analyze_odd_even_scheme()
            
            if not odd_even_results.empty:
                for idx, row in odd_even_results.iterrows():
                    with st.expander(f"{row['period']}", expanded=idx==len(odd_even_results)-1):
                        # Simple visual comparison
                        st.markdown("#### Results")
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Before Scheme", f"AQI {row['before_aqi']:.0f}")
                        col2.metric("During Scheme", f"AQI {row['during_aqi']:.0f}", 
                                   f"{row['pct_change_during']:+.1f}%")
                        col3.metric("After Scheme", f"AQI {row['after_aqi']:.0f}")
                        
                        # Plain language explanation
                        change = row['pct_change_during']
                        
                        st.markdown("#### Analysis")
                        
                        if change < -10 and row['statistically_significant']:
                            st.success(f"""
                            **Noticeable Improvement**
                            
                            AQI dropped by **{abs(change):.0f}%** during the scheme. 
                            This is a meaningful reduction that likely helped reduce health problems.
                            """)
                        elif change < -5:
                            st.warning(f"""
                            **Minor Effect**
                            
                            AQI dropped by only **{abs(change):.0f}%**. Here's the reality:
                            - Weather changes (wind, rain) often cause **20-30%** swings naturally
                            - A {abs(change):.0f}% drop is within normal daily variation
                            - The scheme reduced vehicles by ~25%, but that wasn't enough for major impact
                            """)
                        else:
                            st.error(f"""
                            **No Significant Effect**
                            
                            AQI changed by only **{change:+.0f}%** - essentially flat or worse.
                            Possible reasons:
                            - Other pollution sources (construction, industry) weren't affected
                            - People found workarounds (taxis, metro, work from home)
                            - Weather conditions may have worsened air quality anyway
                            """)
                        
                        st.markdown("#### Key Insight")
                        st.info("""
                        **Vehicle restrictions alone have limited impact.** 
                        For real improvement, cities need:
                        - Cleaner fuels (electric vehicles, CNG)
                        - Better public transport
                        - Control of construction dust and industrial emissions
                        - Regional solutions (stopping stubble burning)
                        """)
        
        # GRAP Analysis
        st.markdown("---")
        st.markdown("### Emergency Measures (GRAP)")
        st.markdown("""
        **What is GRAP?** When pollution gets very bad, the government takes emergency actions:
        - Construction work stopped
        - Schools closed
        - Trucks banned from entering city
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
            **GRAP is helping!** Hazardous days reduced by {abs(change):.0f} percentage points.
            Fewer emergency hospital visits and respiratory crises during winter.
            """)
        elif change < 0:
            st.warning(f"""
            **Slight improvement** but not enough. Still {after_pct:.0f}% of winter days are hazardous.
            Emergency measures help, but we need permanent solutions.
            """)
        else:
            st.error(f"""
            **No improvement.** Despite emergency measures, pollution hasn't decreased.
            This suggests the problem needs systemic solutions, not just temporary bans.
            """)
        
        # ===== RECENT POLICIES SECTION (2024-2026) - AI-Powered Web Search =====
        st.markdown("---")
        st.markdown("### Recent Policies (2024-2026)")
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
        st.subheader("Insurance & Health Planning")
        st.markdown(f"*Plan your healthcare based on **{selected_city}'s** air quality*")
        
        # Get current AQI for calculations
        current_aqi = realtime_data.get('aqi', 150) if realtime_data else 150
        
        # User profile inputs
        st.markdown("---")
        st.markdown("### Your Profile")
        
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
        st.markdown("### Your Pollution Health Risk Score")
        
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
                    - City AQI: **{current_aqi}** ({risk['base_risk']})
                    - Age: **{user_age}** {"(higher risk)" if user_age > 60 or user_age < 18 else "(normal)"}
                    - Conditions: **{', '.join(conditions) if conditions else "None"}**
                    """)
                    
                    if risk_score >= 50:
                        st.warning("Your risk level suggests you should prioritize comprehensive health coverage.")
                    else:
                        st.success("Your risk level is manageable with standard health coverage.")
                
                # Section 2: Recommended Health Checkups
                st.markdown("---")
                st.markdown("### Recommended Health Checkups")
                
                checkup_cache_key = f"checkups_{selected_city}_{current_aqi}"
                
                if st.button("Get Personalized Checkup Recommendations", key="get_checkups"):
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
                            with st.expander(f"{test.get('test_name', 'Test')} - Rs.{test.get('approximate_cost_inr', 0)}"):
                                st.write(f"**Purpose:** {test.get('purpose', 'General health')}")
                                st.write(f"**Frequency:** {test.get('frequency', 'Annual')}")
                        
                        total_cost = checkups.get('total_annual_checkup_cost_inr', 0)
                        st.info(f"**Estimated Annual Checkup Cost:** Rs.{total_cost}")
                    
                    # Tips
                    tips = checkups.get('tips', [])
                    if tips:
                        st.markdown("**Tips:**")
                        for tip in tips[:3]:
                            st.write(f"- {tip}")
                else:
                    st.info("Click the button to get personalized checkup recommendations")
                
                # Section 3: Insurance Recommendations
                st.markdown("---")
                st.markdown("### Insurance Plan Recommendations")
                st.markdown("*AI-powered search for plans that cover pollution-related illness*")
                
                insurance_cache_key = f"insurance_{selected_city}_{user_age}_{family_size}"
                
                if st.button("Find Insurance Plans", type="primary", key="find_insurance"):
                    with st.spinner("Searching for best insurance plans..."):
                        insurance = insurance_agent.get_insurance_recommendations(
                            selected_city, current_aqi, user_age, family_size,
                            [c.lower() for c in conditions]
                        )
                        st.session_state[insurance_cache_key] = insurance
                
                if insurance_cache_key in st.session_state:
                    insurance = st.session_state[insurance_cache_key]
                    
                    # Coverage recommendation
                    coverage = insurance.get('recommended_coverage_amount_lakhs', 10)
                    st.success(f"**Recommended Coverage:** Rs.{coverage} Lakhs for your family of {family_size}")
                    st.caption(insurance.get('coverage_justification', ''))
                    
                    # Insurance plans
                    plans = insurance.get('insurance_plans', [])
                    if plans:
                        st.markdown("**Recommended Plans:**")
                        
                        for plan in plans[:4]:
                            with st.expander(f"{plan.get('provider', 'Provider')} - {plan.get('plan_name', 'Plan')}", expanded=False):
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.markdown(f"**Coverage:** Rs.{plan.get('coverage_lakhs', 0)} Lakhs")
                                    st.markdown(f"**Premium:** Rs.{plan.get('annual_premium_inr', 0):,}/year")
                                    st.markdown(f"**Waiting Period:** {plan.get('waiting_period_months', 0)} months")
                                with col2:
                                    st.markdown("**Key Features:**")
                                    for feature in plan.get('key_features', [])[:3]:
                                        st.write(f"- {feature}")
                                
                                if plan.get('covers_pollution_illness'):
                                    st.success("Covers pollution-related illness")
                                
                                st.caption(f"Best for: {plan.get('best_for', 'General use')}")
                    
                    # Add-ons
                    addons = insurance.get('recommended_add_ons', [])
                    if addons:
                        st.markdown("**Recommended Add-ons:**")
                        for addon in addons[:2]:
                            st.write(f"- **{addon.get('name')}** (Rs.{addon.get('approximate_cost_inr', 0)}/year): {addon.get('why_needed', '')}")
                    
                    # Tips
                    tips = insurance.get('tips', [])
                    if tips:
                        st.markdown("**Insurance Tips:**")
                        for tip in tips[:3]:
                            st.info(tip)
                else:
                    st.info("Click to search for insurance plans suited to your needs")
                
                # Section 4: Cost Comparison
                st.markdown("---")
                st.markdown("### With vs Without Insurance")
                
                analyzer = PolicyImpactAnalyzer()
                health_costs = analyzer.calculate_health_cost_impact(selected_city, 2024)
                
                if health_costs:
                    per_capita = health_costs.get('per_capita_cost', 5000)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.error(f"""
                        **Without Insurance**  
                        Annual out-of-pocket: Rs.{per_capita * 1.5:.0f}  
                        One hospitalization: Rs.50,000+  
                        Risk of major expense: HIGH
                        """)
                    with col2:
                        st.success(f"""
                        **With Health Insurance**  
                        Annual premium: Rs.12,000-20,000  
                        Hospitalization: Covered  
                        Financial protection: SECURED
                        """)
            else:
                st.warning("Add GEMINI_API_KEY to .env for insurance recommendations")
                
        except ImportError:
            st.warning("Insurance planning agent not available.")
        except Exception as e:
            st.error(f"Error loading insurance planner: {str(e)}")
    
    with tab5:
        st.subheader("City Comparison")
        st.markdown(f"**Your Location:** {selected_city}")
        
        # Comparison mode selection
        compare_mode = st.radio(
            "Comparison Type:",
            ["Your City vs Others", "Multi-City Ranking", "Before & After (Time)"],
            horizontal=True,
            key="compare_mode"
        )
        
        if compare_mode == "Your City vs Others":
            st.markdown("### Compare your city with others")
            
            # Find nearby cities based on distance
            def get_nearby_cities(city, all_cities, n=3):
                """Get n closest cities to the given city"""
                if city not in CITY_COORDINATES:
                    return [c for c in all_cities if c != city][:n]
                
                city_lat, city_lon = CITY_COORDINATES[city]
                distances = []
                
                for other_city in all_cities:
                    if other_city != city and other_city in CITY_COORDINATES:
                        other_lat, other_lon = CITY_COORDINATES[other_city]
                        # Simple distance calculation
                        dist = ((city_lat - other_lat)**2 + (city_lon - other_lon)**2)**0.5
                        distances.append((other_city, dist))
                
                distances.sort(key=lambda x: x[1])
                return [c[0] for c in distances[:n]]
            
            nearby_cities = get_nearby_cities(selected_city, available_cities, 3)
            
            # Select cities to compare
            compare_cities = st.multiselect(
                "Select cities to compare with:",
                [c for c in available_cities if c != selected_city],
                default=nearby_cities,
                max_selections=5,
                key="compare_cities",
                help=f"Showing nearby cities to {selected_city} by default"
            )
            
            if compare_cities:
                all_cities = [selected_city] + compare_cities
                
                # Fetch data for all cities
                comparison_data = []
                
                with st.spinner("Fetching air quality data..."):
                    for city in all_cities:
                        realtime = get_realtime_aqi(city)
                        
                        if realtime and realtime.get('aqi'):
                            aqi = realtime['aqi']
                            pm25 = realtime.get('pm25', 'N/A')
                            source = "Live"
                        else:
                            # Fallback to historical
                            city_df = df[df['City'] == city]
                            if not city_df.empty:
                                latest = city_df.iloc[-1]
                                aqi = latest['AQI']
                                pm25 = f"{latest['PM2.5']:.1f}"
                                source = "Historical"
                            else:
                                continue
                        
                        comparison_data.append({
                            'City': city,
                            'AQI': aqi,
                            'PM2.5': pm25,
                            'Source': source,
                            'Is Current': city == selected_city
                        })
                
                if comparison_data:
                    # Sort by AQI (best to worst)
                    comparison_data = sorted(comparison_data, key=lambda x: x['AQI'])
                    
                    # Visual comparison cards
                    st.markdown("### Air Quality Comparison")
                    
                    cols = st.columns(len(comparison_data))
                    
                    for i, data in enumerate(comparison_data):
                        with cols[i]:
                            aqi = data['AQI']
                            
                            # Handle NaN values
                            import math
                            if math.isnan(aqi) if isinstance(aqi, float) else False:
                                aqi = 0
                            
                            # Color based on AQI - Material palette
                            if aqi <= 50:
                                color = "#4caf50"
                            elif aqi <= 100:
                                color = "#8bc34a"
                            elif aqi <= 200:
                                color = "#ff9800"
                            elif aqi <= 300:
                                color = "#ff5722"
                            else:
                                color = "#f44336"
                            
                            highlight = "border: 2px solid #1976d2;" if data['Is Current'] else ""
                            best_label = "BEST" if i == 0 else ""
                            
                            st.markdown(f"""
                            <div class="city-card" style="{highlight}">
                                <div class="city-label">
                                    {'Your Location' if data['Is Current'] else 'City'} {best_label}
                                </div>
                                <div class="city-name">{data['City']}</div>
                                <div class="city-aqi" style="color: {color};">{int(aqi)}</div>
                                <div class="city-label">AQI</div>
                                <div style="font-size: 12px; color: #757575; margin-top: 8px;">
                                    PM2.5: {data['PM2.5']} µg/m³
                                </div>
                                <div style="font-size: 11px; color: #9e9e9e; margin-top: 4px;">
                                    {data['Source']}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Summary
                    st.markdown("---")
                    current_rank = next((i+1 for i, d in enumerate(comparison_data) if d['Is Current']), None)
                    current_aqi = next((d['AQI'] for d in comparison_data if d['Is Current']), None)
                    best_city = comparison_data[0]
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if current_rank == 1:
                            st.success(f"**{selected_city}** has the best air quality!")
                        elif current_rank == len(comparison_data):
                            st.error(f"**{selected_city}** has the worst air quality")
                        else:
                            st.info(f"**{selected_city}** ranks #{current_rank} of {len(comparison_data)}")
                    
                    with col2:
                        st.metric("Best City", best_city['City'], f"AQI: {int(best_city['AQI'])}")
                    
                    with col3:
                        if current_aqi and best_city['AQI']:
                            diff = current_aqi - best_city['AQI']
                            if diff > 0:
                                st.warning(f"{int(diff)} points worse than {best_city['City']}")
                            else:
                                st.success("You're in the cleanest city!")
            else:
                st.info("Select cities above to compare")
        
        elif compare_mode == "Multi-City Ranking":
            st.markdown("### All Cities Ranked by Air Quality")
            
            num_cities = st.slider("Number of cities:", 5, min(20, len(available_cities)), 10)
            
            ranking_data = []
            for city in available_cities[:num_cities]:
                city_df = df[df['City'] == city]
                if not city_df.empty:
                    latest = city_df.iloc[-1]
                    ranking_data.append({
                        'City': city,
                        'AQI': latest['AQI'],
                        'Is Current': city == selected_city
                    })
            
            ranking_data = sorted(ranking_data, key=lambda x: x['AQI'])
            
            for i, data in enumerate(ranking_data):
                rank_label = "1st" if i == 0 else "2nd" if i == 1 else "3rd" if i == 2 else f"{i+1}th"
                your_loc = " (You)" if data['Is Current'] else ""
                aqi = data['AQI']
                
                # Handle NaN values
                import math
                if isinstance(aqi, float) and math.isnan(aqi):
                    aqi = 0
                
                aqi_color = "#4caf50" if aqi <= 100 else "#ff9800" if aqi <= 200 else "#f44336"
                current_class = "rank-item current" if data['Is Current'] else "rank-item"
                
                st.markdown(f"""
                <div class="{current_class}">
                    <span><strong>{rank_label}</strong> {data['City']}<span style="color: #1976d2; font-weight: 500;">{your_loc}</span></span>
                    <span style="color: {aqi_color}; font-weight: 500;">AQI {int(aqi)}</span>
                </div>
                """, unsafe_allow_html=True)
        
        else:  # Time comparison
            st.markdown(f"### {selected_city}: Air Quality Over Time")
            
            col1, col2 = st.columns(2)
            with col1:
                period1 = st.selectbox("Recent Period:", ["Last 7 days", "Last 30 days", "Last 90 days"])
            with col2:
                period2 = st.selectbox("Compare To:", ["Previous 7 days", "Previous 30 days", "Same period last year"])
            
            period_map = {"Last 7 days": 7, "Last 30 days": 30, "Last 90 days": 90}
            days1 = period_map.get(period1, 7)
            
            recent_data = city_data.tail(days1)
            recent_avg = recent_data['AQI'].mean()
            
            if period2 == "Same period last year":
                older_data = city_data.tail(365 + days1).head(days1)
            else:
                prev_days = int(period2.split()[1])
                older_data = city_data.tail(days1 + prev_days).head(prev_days)
            
            older_avg = older_data['AQI'].mean() if not older_data.empty else recent_avg
            change = ((recent_avg - older_avg) / older_avg * 100) if older_avg > 0 else 0
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(period1, f"{recent_avg:.0f} AQI")
            with col2:
                st.metric(period2, f"{older_avg:.0f} AQI")
            with col3:
                st.metric("Change", f"{change:+.1f}%", delta=f"{change:+.1f}%", delta_color="inverse")
            
            if change < -10:
                st.success("Air quality has improved significantly!")
            elif change > 10:
                st.error("Air quality has worsened")
            else:
                st.info("Air quality is stable")
    
    with tab6:
        st.subheader("AI Health Advisor")
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
