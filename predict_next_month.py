"""
Predict Next Month's Data - EEPD System
Simple script to generate next month (30 days) predictions

Usage:
    python predict_next_month.py your_data.csv
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from covid_trend_detection import COVIDTrendAnalysisSystem

# ========================================
# CONFIGURATION
# ========================================

# Path to your CSV file
DATA_FILE = 'your_data.csv'  # Change this to your actual file path

# Disease name
DISEASE_NAME = 'COVID-19'  # or 'Disease X', 'Disease Y', etc.

# Model parameters
WINDOW_SIZE = 14
EPOCHS = 50  # Can reduce to 30 for faster prediction
SENSITIVITY = 'medium'

# ========================================
# MAIN FUNCTION
# ========================================

def predict_next_month(data_file):
    """Generate next month (30 days) predictions"""
    
    print("\n" + "="*80)
    print(" NEXT MONTH PREDICTION - EEPD SYSTEM ".center(80, "="))
    print("="*80)
    
    print(f"\n📊 Loading data from: {data_file}")
    
    try:
        # Load data
        df = pd.read_csv(data_file)
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
        df = df.sort_values('Date').reset_index(drop=True)
        
        print(f"✅ Data loaded: {len(df)} days")
        print(f"   Date range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
        
        # Initialize system
        print(f"\n🤖 Initializing EEPD System for {DISEASE_NAME}...")
        system = COVIDTrendAnalysisSystem(
            df=df,
            window_size=WINDOW_SIZE,
            sensitivity=SENSITIVITY,
            disease_name=DISEASE_NAME
        )
        
        # Train models (required for prediction)
        print(f"\n🔧 Training LSTM model ({EPOCHS} epochs)...")
        print("   This may take a few minutes...")
        system.train_lstm_model(epochs=EPOCHS, batch_size=32)
        print("   ✅ LSTM model trained")
        
        # Generate next month forecast (30 days)
        print(f"\n🔮 Generating next month predictions (30 days)...")
        forecast = system.generate_forecast(days_ahead=30)
        
        if forecast is None:
            print("❌ Failed to generate forecast")
            return None
        
        # Display summary
        print("\n" + "="*80)
        print(" PREDICTION SUMMARY ".center(80, "="))
        print("="*80)
        
        last_date = df['Date'].max()
        first_pred_date = forecast['Date'].min()
        last_pred_date = forecast['Date'].max()
        
        print(f"\n📅 Prediction Period: {first_pred_date.strftime('%Y-%m-%d')} to {last_pred_date.strftime('%Y-%m-%d')}")
        print(f"   (Next 30 days from {last_date.strftime('%Y-%m-%d')})")
        
        # Key metrics summary
        print(f"\n📈 Key Predictions:")
        print(f"   New Cases:")
        print(f"     - First day: {forecast['New_Cases'].iloc[0]:.0f}")
        print(f"     - Last day: {forecast['New_Cases'].iloc[-1]:.0f}")
        print(f"     - Average: {forecast['New_Cases'].mean():.0f}")
        print(f"     - Peak: {forecast['New_Cases'].max():.0f} (day {forecast['New_Cases'].idxmax() + 1})")
        
        if 'ICU_Admission_Rate' in forecast.columns:
            print(f"\n   ICU Admission Rate:")
            print(f"     - Average: {forecast['ICU_Admission_Rate'].mean()*100:.2f}%")
            print(f"     - Peak: {forecast['ICU_Admission_Rate'].max()*100:.2f}%")
        
        if 'Ventilator_Utilization' in forecast.columns:
            print(f"\n   Ventilator Utilization:")
            print(f"     - Average: {forecast['Ventilator_Utilization'].mean()*100:.2f}%")
            print(f"     - Peak: {forecast['Ventilator_Utilization'].max()*100:.2f}%")
        
        # Save predictions
        output_file = 'next_month_predictions.csv'
        forecast.to_csv(output_file, index=False)
        print(f"\n💾 Predictions saved to: {output_file}")
        
        # Display first few days
        print(f"\n📋 First 5 Days of Predictions:")
        print(forecast[['Date', 'New_Cases', 'Test_Positivity_Rate', 
                       'Hospitalization_Rate', 'ICU_Admission_Rate']].head().to_string(index=False))
        
        # Display last few days
        print(f"\n📋 Last 5 Days of Predictions:")
        print(forecast[['Date', 'New_Cases', 'Test_Positivity_Rate', 
                       'Hospitalization_Rate', 'ICU_Admission_Rate']].tail().to_string(index=False))
        
        # Trend analysis
        print(f"\n📊 Trend Analysis:")
        case_trend = forecast['New_Cases'].iloc[-1] - forecast['New_Cases'].iloc[0]
        if case_trend > 0:
            print(f"   ⬆️  Cases predicted to INCREASE by {case_trend:.0f} over the month")
        elif case_trend < 0:
            print(f"   ⬇️  Cases predicted to DECREASE by {abs(case_trend):.0f} over the month")
        else:
            print(f"   ➡️  Cases predicted to remain STABLE")
        
        print("\n" + "="*80)
        print(" ✅ PREDICTION COMPLETE ".center(80, "="))
        print("="*80)
        
        return forecast
        
    except FileNotFoundError:
        print(f"\n❌ ERROR: File not found: {data_file}")
        print("Please check the file path and try again.")
        return None
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def show_usage():
    """Show usage instructions"""
    print("""
    ========================================
    Next Month Prediction - EEPD System
    ========================================

    Usage:
        python predict_next_month.py <data_file.csv>

    Or:
        1. Edit DATA_FILE variable in this script
        2. Run: python predict_next_month.py

    Your CSV file should have these columns:
        - Date (format: DD/MM/YYYY)
        - Test_Positivity_Rate
        - New_Cases
        - Case_Growth_Rate
        - Hospitalization_Rate
        - ICU_Admission_Rate
        - Ventilator_Utilization
        - Mortality_Rate

    Output:
        - next_month_predictions.csv: 30-day forecast

    Configuration (edit in script):
        - DISEASE_NAME: Disease name (default: 'COVID-19')
        - EPOCHS: Training epochs (default: 50, reduce to 30 for faster)
    """)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] in ['-h', '--help', 'help']:
            show_usage()
        else:
            data_file = sys.argv[1]
            forecast = predict_next_month(data_file)
    else:
        print("No command line argument provided.")
        print(f"Using DATA_FILE from configuration: {DATA_FILE}\n")
        
        if DATA_FILE == 'your_data.csv':
            print("⚠️  WARNING: Please update the DATA_FILE variable with your actual file path!")
            show_usage()
        else:
            forecast = predict_next_month(DATA_FILE)

