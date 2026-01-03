# Air Quality Analytics Dashboard

A comprehensive end-to-end Air Pollution Monitoring System that combines historical analysis, Prophet-based forecasting, and real-time API integration.

## 🚀 Features
- **Multi-City Support**: Monitors 10+ major global cities including Delhi, New York, London, and Tokyo.
- **Historical Analysis**: Interactive visualization of AQI trends from 2020-2025.
- **Forecasting**: Facebook Prophet model integration to predict air quality for the next 7 days.
- **Real-Time Data**: Live integration with OpenAQ API for up-to-the-minute updates.
- **Health Insights**: Actionable health recommendations based on current pollution levels.

## 🛠️ Tech Stack
- **Data Engineering**: Pandas, NumPy
- **Machine Learning**: Facebook Prophet, Scikit-learn
- **API Integration**: OpenAQ API, Requests
- **Visualization**: Plotly Interactive Charts
- **Frontend**: Streamlit

## 📂 Project Structure
```
.
├── data/               # Data storage
│   ├── raw/            # Raw CSVs and synthetic data
│   └── processed/      # Cleaned data for ML models
├── models/             # Serialized Prophet models
├── src/                # Source code
│   ├── generate_data.py # Synthetic data generator
│   ├── openaq_client.py # API client
│   └── ...
└── requirements.txt    # Python dependencies
```

## 🏁 Getting Started

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Generate Data**
   ```bash
   python src/generate_data.py
   ```

3. **Run Dashboard (Coming Soon)**
   ```bash
   streamlit run src/app.py
   ```
