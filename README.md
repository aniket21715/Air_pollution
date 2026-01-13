# Air Quality Monitor & Health Advisory System

> **AI-Powered Air Quality Intelligence for India** — Real-time monitoring, forecasting, and personalized health recommendations

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)](https://streamlit.io)
[![Live Demo](https://img.shields.io/badge/demo-live-brightgreen.svg)](https://air-pollution-monitor.streamlit.app)

---

## Features

### Real-Time Air Quality Monitoring
- Live AQI data for 50+ Indian cities via OpenWeatherMap API
- PM2.5, PM10, NO2, SO2, CO concentration tracking
- Auto-detect user location with IP geolocation

### AI-Powered Health Advisor (Gemini 2.5 Flash)
- Personalized recommendations based on user profile (asthma, elderly, children, athletes)
- Dynamic activity suggestions that change based on AQI and time of day
- Interactive chat for air quality questions

### Protection Impact Visualization
- Shows real-world effect of precautions (air purifier: 80% reduction, N95 mask: 95% filtered)
- Educates users on why precautions matter with before/after AQI comparison

### 7-Day AQI Forecasting (NeuralProphet)
- Machine learning predictions with 90% confidence intervals
- Activity-specific guidance (jog, cycling, commute, outdoor events)
- Seasonal pattern recognition

### Historical Trends Analysis
- Multi-pollutant charts (AQI, PM2.5, PM10, NO2, SO2, CO)
- Time range filters (30 days to all-time)
- Statistical summaries with threshold markers

### Policy Impact Analysis
- Odd-Even scheme effectiveness with statistical testing
- GRAP (emergency measures) evaluation
- Real policy updates from government sources

### Insurance & Health Planning
- Pollution health risk score calculation
- AI-powered insurance plan recommendations
- Cost comparison (with vs without insurance)

### City Comparison
- Compare your city with nearby cities (auto-detected by distance)
- Multi-city ranking by AQI
- Before/after time comparison

---

## Tech Stack

| Category | Technologies |
|----------|--------------|
| **Frontend** | Streamlit, Material Design CSS |
| **AI/ML** | Google Gemini 2.5 Flash, NeuralProphet |
| **Data** | Pandas, NumPy, Plotly |
| **APIs** | OpenWeatherMap AQI, ipapi.co geolocation |
| **Deployment** | Streamlit Cloud, GitHub |

---

## Quick Start

```bash
# Clone repository
git clone https://github.com/aniket21715/Air_pollution.git
cd Air_pollution

# Install dependencies
pip install -r requirements.txt

# Set environment variables
echo "GEMINI_API_KEY=your_key" > .env
echo "OPENWEATHERMAP_API_KEY=your_key" >> .env

# Run application
streamlit run src/app.py
```

---

## Project Structure

```
Air_pollution/
├── src/
│   ├── app.py                    # Main Streamlit application
│   ├── gemini_advisor.py         # AI health advisor (Gemini API)
│   ├── health_advisor.py         # Rule-based health recommendations
│   ├── neural_prophet_trainer.py # ML forecasting model
│   ├── policy_impact_analyzer.py # Policy evaluation & health costs
│   ├── location_detector.py      # City coordinates & distance calc
│   ├── openweathermap_client.py  # Real-time AQI API
│   └── indian_cities_config.py   # 50+ cities metadata
├── data/
│   └── processed/                # Historical AQI data
├── agents/
│   ├── researcher.py             # Policy research agent
│   └── insurance_advisor.py      # Insurance planning agent
└── requirements.txt
```

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | Yes | Google AI Studio API key for Gemini |
| `OPENWEATHERMAP_API_KEY` | Optional | For real-time AQI (has fallback) |

---

## Screenshots

- **Dashboard**: Real-time AQI with health recommendations
- **Protection Impact**: Air purifier and mask effectiveness visualization
- **Historical Trends**: Multi-pollutant analysis charts
- **City Comparison**: Nearby cities AQI comparison

---

## Key Achievements

- **50+ Indian cities** covered with real-time and historical data
- **7 user profiles** for personalized health advice
- **5 AI agents** for specialized recommendations
- **6 pollutants** tracked with historical trends
- **Material Design UI** for professional, clean interface

---

## Author

**Aniket**  
GitHub: [@aniket21715](https://github.com/aniket21715)

---

## License

MIT License — Free for personal and educational use.
