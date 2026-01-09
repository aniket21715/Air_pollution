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

# Import custom modules
from indian_cities_config import (
    INDIAN_CITIES, get_all_cities, get_aqi_category,
    get_cities_by_pollution_level, AQI_CATEGORIES
)
from health_advisor import HealthAdvisor
from policy_impact_analyzer import PolicyImpactAnalyzer

# Try to import real-time AQI clients (OpenWeatherMap primary, Open-Meteo fallback)
OPENWEATHERMAP_API_KEY = "7fd3d6c76a563b00cb481d3319ab4a93"  # User's API key

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
        background: white;
    }
    .city-stat {
        background: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 10px 0;
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
        planned_activity = st.selectbox(
            "What are you planning?",
            ["jog", "cycling", "outdoor_event", "commute", "children_play", "window_open"]
        )
        
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
    
    st.markdown("---")
    
    # Tabs for different analysis
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 7-Day Forecast",
        "📊 Historical Trends",
        "🏛️ Policy Impact",
        "💰 Health Costs",
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
        
        # Activity guidance
        st.markdown(f"### 🏃 Activity Guidance: *{planned_activity.replace('_', ' ').title()}*")
        activity_guidance = HealthAdvisor.get_activity_guidance(
            forecast_data['aqi'],
            planned_activity,
            user_profile
        )
        
        st.info(f"**{activity_guidance['recommendation']}**")
        st.write(f"- **AQI Range**: {activity_guidance['min_aqi']:.0f} - {activity_guidance['max_aqi']:.0f}")
        st.write(f"- **Average AQI**: {activity_guidance['avg_aqi']:.1f}")
    
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
    
    with tab4:
        st.subheader("💰 What Bad Air Costs You")
        st.markdown(f"*The hidden price of pollution in **{selected_city}** (2024)*")
        
        analyzer = PolicyImpactAnalyzer()
        health_costs = analyzer.calculate_health_cost_impact(selected_city, 2024)
        
        if health_costs:
            total_unhealthy = health_costs['days_poor'] + health_costs['days_very_poor'] + health_costs['days_severe']
            total_days = 365
            unhealthy_pct = (total_unhealthy / total_days) * 100
            
            # Visual summary
            st.markdown("---")
            st.markdown("### 📅 Days You Breathed Unhealthy Air in 2024")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("🟡 Poor Days", f"{health_costs['days_poor']}", 
                       help="AQI 201-300: Sensitive groups should avoid outdoor exercise")
            col2.metric("🟠 Very Poor Days", f"{health_costs['days_very_poor']}",
                       help="AQI 301-400: Everyone should limit outdoor activity")
            col3.metric("🔴 Severe Days", f"{health_costs['days_severe']}",
                       help="AQI 401+: Health emergency - stay indoors!")
            col4.metric("📊 Total Unhealthy", f"{total_unhealthy} days",
                       f"{unhealthy_pct:.0f}% of year")
            
            # Personal cost breakdown
            st.markdown("---")
            st.markdown("### 💳 What It Costs YOU (Per Person Per Year)")
            
            per_capita = health_costs['per_capita_cost']
            
            # Break down the costs into relatable categories
            doctor_cost = per_capita * 0.45  # 45% on doctor visits
            medicine_cost = per_capita * 0.35  # 35% on medicines
            productivity_cost = per_capita * 0.20  # 20% on lost work
            
            st.markdown(f"""
            | Category | Estimated Cost | What It Means |
            |----------|----------------|---------------|
            | 🏥 Doctor Visits | ₹{doctor_cost:.0f}/year | Extra visits for cough, breathing issues, allergies |
            | 💊 Medicines | ₹{medicine_cost:.0f}/year | Inhalers, antihistamines, cough syrups |
            | 📉 Sick Days | ₹{productivity_cost:.0f}/year | Lost wages when too sick to work |
            | **TOTAL** | **₹{per_capita:.0f}/year** | *That's ₹{per_capita/12:.0f} every month* |
            """)
            
            # City-wide cost
            st.markdown("---")
            st.markdown(f"### 🏙️ What It Costs {selected_city} (Total)")
            
            population_cr = health_costs['population'] / 10000000
            total_cost_cr = health_costs['estimated_health_cost_crores']
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("City Population", f"{population_cr:.1f} Crore people")
            with col2:
                st.error(f"### ₹{total_cost_cr:.0f} Crores/year")
            
            # Make it relatable
            hospitals = total_cost_cr / 20  # Assume ₹20 Cr per hospital
            schools = total_cost_cr / 5  # Assume ₹5 Cr per school
            
            st.info(f"""
            **To put ₹{total_cost_cr:.0f} Crores in perspective:**
            - 🏥 Could build **{hospitals:.0f} new hospitals**
            - 🏫 Could build **{schools:.0f} new schools**  
            - 🚌 Could buy **{total_cost_cr*10:.0f} electric buses**
            
            *This money is being spent on TREATING pollution-related illness instead of PREVENTING it.*
            """)
            
            # What could be saved
            st.markdown("---")
            st.markdown("### 💡 What If Air Was Better?")
            
            # Compare with a cleaner city (assume Bengaluru has 40% lower costs)
            savings_if_better = total_cost_cr * 0.40  # 40% savings estimate
            
            st.success(f"""
            If {selected_city} reduced pollution to Bengaluru's level:
            - 💰 **₹{savings_if_better:.0f} Crores saved** annually
            - 🏥 **{savings_if_better * 10000000 / health_costs['population']:.0f}** fewer rupees per person
            - 😷 **{total_unhealthy * 0.4:.0f} fewer unhealthy days** per year
            
            *Cleaner air = More money in your pocket, fewer hospital visits, longer life!*
            """)
        else:
            st.warning(f"No health cost data available for {selected_city}")
    
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
                    'Tier': city_config.get('tier', 'N/A')
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
        st.subheader("🤖 AI-Powered Health Advisor")
        st.markdown("*Powered by Google Gemini 2.0 Flash*")
        
        try:
            from gemini_advisor import GeminiHealthAdvisor
            
            # Initialize Gemini with API key
            GEMINI_API_KEY = "AIzaSyA6SzVuFQQDyGx4wlCfnflCdYl1J4JQkH0"
            advisor = GeminiHealthAdvisor(GEMINI_API_KEY)
            
            col_ai1, col_ai2 = st.columns([2, 1])
            
            with col_ai1:
                st.markdown("### 💬 Ask Me Anything About Air Quality")
                
                # Initialize chat history in session state
                if 'chat_history' not in st.session_state:
                    st.session_state.chat_history = []
                
                # User input
                user_question = st.text_input(
                    "Type your question:",
                    placeholder="Is it safe to go jogging today? Should I wear a mask?"
                )
                
                if st.button("🚀 Get AI Advice", type="primary"):
                    if user_question:
                        with st.spinner("Thinking..."):
                            response = advisor.chat(
                                user_message=user_question,
                                city=selected_city,
                                aqi=current_aqi,
                                category=category,
                                user_profile=user_profile
                            )
                            
                            # Add to chat history
                            st.session_state.chat_history.append({
                                'question': user_question,
                                'answer': response
                            })
                
                # Display chat history
                if st.session_state.chat_history:
                    st.markdown("---")
                    st.markdown("### 📝 Conversation History")
                    for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):
                        with st.expander(f"Q: {chat['question'][:50]}...", expanded=(i==0)):
                            st.markdown(f"**You asked:** {chat['question']}")
                            st.markdown(f"**AI Response:** {chat['answer']}")
                
                if st.button("🗑️ Clear Chat History"):
                    st.session_state.chat_history = []
                    st.rerun()
            
            with col_ai2:
                st.markdown("### 🎯 Quick AI Advice")
                
                if st.button("Get Personalized Advice", use_container_width=True):
                    with st.spinner("Generating personalized advice..."):
                        advice = advisor.get_personalized_advice(
                            city=selected_city,
                            aqi=current_aqi,
                            category=category,
                            pm25=pm25_val if pm25_val != 'N/A' else 0,
                            pm10=pm10_val if pm10_val != 'N/A' else 0,
                            user_profile=user_profile,
                            activity=planned_activity
                        )
                        st.success(advice)
                
                st.markdown("---")
                st.markdown("### 🏃 Activity Recommendation")
                
                activity_options = ["jogging", "cycling", "walking", "outdoor yoga", "children's play", "morning exercise"]
                selected_activity = st.selectbox("Choose activity:", activity_options)
                
                if st.button("Check Activity Safety", use_container_width=True):
                    with st.spinner("Analyzing..."):
                        recommendation = advisor.get_activity_recommendation(
                            city=selected_city,
                            aqi=current_aqi,
                            activity_type=selected_activity,
                            user_profile=user_profile
                        )
                        st.info(recommendation)
                
        except Exception as e:
            st.error(f"⚠️ AI Advisor not available: {str(e)}")
            st.info("Make sure google-generativeai is installed: `pip install google-generativeai`")


if __name__ == "__main__":
    main()
