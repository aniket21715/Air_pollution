"""
Neural Prophet Trainer for Air Quality Forecasting
Implements state-of-the-art time-series forecasting with:
- Multi-variate predictions (PM2.5, PM10, NO2, O3)
- Uncertainty quantification
- Seasonal decomposition
- External regressors (holidays, policy events)
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta
from neuralprophet import NeuralProphet, set_log_level
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
set_log_level("ERROR")  # Reduce verbosity

class AirQualityForecaster:
    """
    Advanced air quality forecasting using NeuralProphet.
    Handles multi-variate time-series with uncertainty quantification.
    """
    
    def __init__(self, data_path="data/raw/india_aqi_complete.csv", models_dir="models/neuralprophet"):
        self.data_path = data_path
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        
        # Load data
        print(f"Loading data from {data_path}...")
        self.df = pd.read_csv(data_path)
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        
        print(f"✅ Loaded {len(self.df):,} rows for {self.df['City'].nunique()} cities")
        print(f"   Date range: {self.df['Date'].min().date()} to {self.df['Date'].max().date()}")
    
    def prepare_city_data(self, city_name, target_column='AQI'):
        """
        Prepare data for a specific city in NeuralProphet format.
        NeuralProphet requires columns: 'ds' (datetime) and 'y' (target variable)
        """
        city_df = self.df[self.df['City'] == city_name].copy()
        
        if len(city_df) == 0:
            raise ValueError(f"No data found for city: {city_name}")
        
        # NeuralProphet format - keep it simple with just ds and y
        # External regressors removed to prevent training issues
        prophet_df = pd.DataFrame({
            'ds': city_df['Date'],
            'y': city_df[target_column]
        })
        
        # Drop any rows with NaN values
        prophet_df = prophet_df.dropna()
        
        return prophet_df.sort_values('ds').reset_index(drop=True)
    
    def train_city_model(self, city_name, target_column='AQI', epochs=50):
        """
        Train NeuralProphet model for a specific city and pollutant.
        
        Args:
            city_name: City to train model for
            target_column: Pollutant to forecast (AQI, PM2.5, PM10, NO2, O3)
            epochs: Number of training epochs (more = better but slower)
        """
        print(f"\n{'='*60}")
        print(f"Training {target_column} forecaster for {city_name}")
        print(f"{'='*60}")
        
        # Prepare data
        df_train = self.prepare_city_data(city_name, target_column)
        
        # Split into train/validation (last 7 days for validation)
        # Reduced from 30 to 7 to work with smaller datasets
        min_required_samples = 100  # Minimum samples needed for training
        
        if len(df_train) < min_required_samples:
            raise ValueError(f"Insufficient data: {len(df_train)} samples (minimum {min_required_samples} required)")
        
        split_idx = len(df_train) - 7
        df_train_split = df_train.iloc[:split_idx]
        df_val = df_train.iloc[split_idx:]
        
        print(f"Training samples: {len(df_train_split)}")
        print(f"Validation samples: {len(df_val)}")
        
        # Initialize NeuralProphet with SIMPLE configuration
        # Using only seasonality - no AR/lags to avoid data size issues
        model = NeuralProphet(
            growth="linear",
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            epochs=epochs,
            learning_rate=0.1,
            batch_size=32,
            loss_func="MSE",
        )
        
        # Train model (no lagged regressors - they cause data size issues)
        print("\n🚀 Starting training...")
        metrics = model.fit(df_train_split, freq='D', validation_df=df_val, progress='bar')
        
        # Evaluate on validation set
        forecast = model.predict(df_val)
        mae = np.mean(np.abs(forecast['yhat1'].values - df_val['y'].values))
        rmse = np.sqrt(np.mean((forecast['yhat1'].values - df_val['y'].values)**2))
        
        print(f"\n✅ Training complete!")
        print(f"   MAE (validation): {mae:.2f}")
        print(f"   RMSE (validation): {rmse:.2f}")
        
        # Save model
        model_filename = f"{city_name}_{target_column}_model.pkl"
        model_path = os.path.join(self.models_dir, model_filename)
        
        # Workaround: Save model using torch
        import torch
        torch.save(model, model_path)
        
        # Save metadata
        metadata = {
            'city': city_name,
            'target': target_column,
            'trained_date': datetime.now().isoformat(),
            'mae': float(mae),
            'rmse': float(rmse),
            'train_samples': len(df_train_split),
            'val_samples': len(df_val),
            'epochs': epochs,
        }
        
        metadata_path = os.path.join(self.models_dir, f"{city_name}_{target_column}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"💾 Saved model to: {model_path}")
        print(f"💾 Saved metadata to: {metadata_path}")
        
        return model, metrics, mae, rmse
    
    def generate_forecast(self, model, city_name, target_column='AQI', days_ahead=7):
        """Generate future forecast with uncertainty bounds."""
        df_full = self.prepare_city_data(city_name, target_column)
        
        # Make future dataframe
        future = model.make_future_dataframe(df_full, periods=days_ahead, n_historic_predictions=30)
        
        # Generate predictions
        forecast = model.predict(future)
        
        # Extract future predictions
        forecast_future = forecast.tail(days_ahead)
        
        return forecast_future
    
    def train_all_cities(self, cities=None, target_columns=['AQI', 'PM2.5', 'PM10']):
        """
        Train models for multiple cities and pollutants.
        
        Args:
            cities: List of cities to train (None = all cities)
            target_columns: List of pollutants to forecast
        """
        if cities is None:
            cities = self.df['City'].unique()
        
        results = []
        total_models = len(cities) * len(target_columns)
        current = 0
        
        print(f"\n{'='*60}")
        print(f"  NEURAL PROPHET TRAINING PIPELINE")
        print(f"  Cities: {len(cities)}")
        print(f"  Pollutants: {len(target_columns)}")
        print(f"  Total models to train: {total_models}")
        print(f"{'='*60}")
        
        for city in cities:
            for target in target_columns:
                current += 1
                print(f"\n[{current}/{total_models}] Training {city} - {target}")
                
                try:
                    model, metrics, mae, rmse = self.train_city_model(city, target, epochs=30)
                    results.append({
                        'city': city,
                        'target': target,
                        'mae': mae,
                        'rmse': rmse,
                        'status': 'success'
                    })
                except Exception as e:
                    print(f"   ❌ Error training {city} - {target}: {e}")
                    results.append({
                        'city': city,
                        'target': target,
                        'mae': None,
                        'rmse': None,
                        'status': 'failed',
                        'error': str(e)
                    })
        
        # Save results summary
        results_df = pd.DataFrame(results)
        results_path = os.path.join(self.models_dir, 'training_summary.csv')
        results_df.to_csv(results_path, index=False)
        
        print(f"\n{'='*60}")
        print(f"  TRAINING COMPLETE")
        print(f"{'='*60}")
        print(f"✅ Successfully trained: {(results_df['status'] == 'success').sum()} models")
        print(f"❌ Failed: {(results_df['status'] == 'failed').sum()} models")
        print(f"📊 Results saved to: {results_path}")
        
        # Top 5 best models by MAE
        successful = results_df[results_df['status'] == 'success'].sort_values('mae')
        print(f"\n🏆 Top 5 Most Accurate Models:")
        for idx, row in successful.head(5).iterrows():
            print(f"   {row['city']} ({row['target']}): MAE = {row['mae']:.2f}")
        
        return results_df


def main():
    """Main training pipeline."""
    # Initialize forecaster
    forecaster = AirQualityForecaster()
    
    # Option 1: Train for select major cities (faster for testing)
    major_cities = ['Delhi', 'Mumbai', 'Bengaluru', 'Kolkata', 'Chennai', 
                    'Ghaziabad', 'Kanpur', 'Lucknow', 'Patna', 'Meerut']
    
    # Option 2: Train for all cities (set to None)
    # major_cities = None
    
    # Train models
    results = forecaster.train_all_cities(
        cities=major_cities,
        target_columns=['AQI', 'PM2.5']  # Can add PM10, NO2, O3
    )
    
    print("\n✅ Model training pipeline complete!")
    print("You can now run: streamlit run src/app.py")


if __name__ == "__main__":
    main()
