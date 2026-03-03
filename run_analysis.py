"""
Early Epidemic & Pandemic Detection System (EEPD) - Full Analysis Runner

This script runs the complete EEPD analysis including:
- LSTM-based forecasting
- Wave prediction
- ICU/Ventilator monitoring
- 4-tier risk classification
- Automated alerting

Usage:
    python run_analysis.py your_data.csv

Or modify the DATA_FILE variable below and run:
    python run_analysis.py
"""

import sys
import pandas as pd
from covid_trend_detection import COVIDTrendAnalysisSystem

# ========================================
# CONFIGURATION
# ========================================

# Path to your CSV file
DATA_FILE = 'your_data.csv'  # Change this to your actual file path

# Disease name (for multi-disease support)
DISEASE_NAME = 'COVID-19'  # Can be 'COVID-19', 'Disease X', 'Disease Y', etc.

# Model parameters
WINDOW_SIZE = 14  # Number of days to look back for predictions
EPOCHS = 50  # Training epochs (50-100 recommended, more = better but slower)
SENSITIVITY = 'medium'  # Alert sensitivity: 'low', 'medium', or 'high'

# Forecast parameters
FORECAST_DAYS = 30  # Number of days to forecast ahead (next month = 30 days)

# ========================================
# MAIN EXECUTION
# ========================================

def run_analysis(data_file):
    """Run the complete analysis"""

    print(f"\nLoading data from: {data_file}")

    try:
        # Load your data
        df = pd.read_csv(data_file)

        print(f"Data loaded successfully!")
        print(f"Shape: {df.shape}")
        print(f"\nFirst few rows:")
        print(df.head())

        # Initialize the system
        print(f"\nInitializing EEPD System for {DISEASE_NAME}...")
        print(f"  - Disease: {DISEASE_NAME}")
        print(f"  - Window size: {WINDOW_SIZE} days")
        print(f"  - Sensitivity: {SENSITIVITY}")
        print(f"  - Training epochs: {EPOCHS}")

        system = COVIDTrendAnalysisSystem(
            df=df,
            window_size=WINDOW_SIZE,
            sensitivity=SENSITIVITY,
            disease_name=DISEASE_NAME
        )

        # Run full analysis
        print("\nStarting full EEPD analysis pipeline...")
        print("This may take a few minutes depending on your hardware...")
        print("The analysis includes:")
        print("  - Training LSTM and Transformer models")
        print("  - Risk level classification (4-tier)")
        print("  - Wave prediction")
        print("  - ICU/Ventilator monitoring")
        print("  - Automated alerting\n")

        results = system.run_full_analysis(
            train_models=True,
            epochs=EPOCHS
        )

        # Save results
        if results and results.get('forecast') is not None:
            forecast = results['forecast']
            forecast.to_csv('forecast_results.csv', index=False)
            print(f"\n✅ Forecast saved to: forecast_results.csv")

        # Save detailed results with EEPD variables
        if 'risk_levels' in system.results:
            results_df = pd.DataFrame({
                'Date': system.results['dates'],
                'Trend': system.results['trends'],
                'Risk_Level': system.results['risk_levels'],
                'Risk_Score': system.results.get('risk_scores', [0] * len(system.results['dates'])),
                'Anomaly_Detected': system.results['anomalies'],
                'Alert_Triggered': system.results['alert_triggered'],
                'Alert_Message': system.results['alert_messages'],
                'ICU_Utilization': system.results.get('icu_utilization', [0] * len(system.results['dates'])),
                'Ventilator_Utilization': system.results.get('ventilator_utilization', [0] * len(system.results['dates']))
            })
        else:
            results_df = pd.DataFrame({
                'Date': system.results['dates'],
                'Trend': system.results['trends'],
                'Anomaly_Detected': system.results['anomalies'],
                'Alert_Triggered': system.results['alert_triggered'],
                'Alert_Message': system.results['alert_messages']
            })
        results_df.to_csv('detection_results.csv', index=False)
        print(f"✅ Detection results saved to: detection_results.csv")
        
        # Save detailed metrics matrix
        try:
            metrics_df = system.get_detailed_metrics_matrix()
            if metrics_df is not None:
                metrics_df.to_csv('detailed_metrics_matrix.csv', index=False)
                print(f"✅ Detailed metrics matrix saved to: detailed_metrics_matrix.csv")
        except Exception as e:
            print(f"⚠️ Could not save detailed metrics matrix: {e}")

        print("\n" + "="*80)
        print(" SUCCESS! Analysis Complete ".center(80, "="))
        print("="*80)

        return system, forecast

    except FileNotFoundError:
        print(f"\nERROR: File not found: {data_file}")
        print("Please check the file path and try again.")
        return None, None

    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None


def show_usage():
    """Show usage instructions"""
    print("""
    ========================================
    Early Epidemic & Pandemic Detection System (EEPD)
    ========================================

    Usage:
        python run_analysis.py <data_file.csv>

    Or:
        1. Edit DATA_FILE variable in this script
        2. Run: python run_analysis.py

    Your CSV file should have these columns:
        - Date (format: DD/MM/YYYY)
        - Test_Positivity_Rate
        - New_Cases
        - Case_Growth_Rate
        - Hospitalization_Rate
        - ICU_Admission_Rate
        - Ventilator_Utilization
        - Mortality_Rate

    Configuration Options (edit in script):
        - DISEASE_NAME: Name of disease (default: 'COVID-19')
        - WINDOW_SIZE: Look-back window (default: 14 days)
        - EPOCHS: Training epochs (default: 50)
        - SENSITIVITY: 'low', 'medium', or 'high' (default: 'medium')
        - FORECAST_DAYS: Days to forecast (default: 30)

    Features:
        ✅ Multi-disease outbreak detection
        ✅ LSTM-based forecasting
        ✅ Wave prediction
        ✅ ICU/Ventilator monitoring
        ✅ 4-tier risk classification (Normal, Elevated Risk, High Alert, Critical)
        ✅ Automated alerting

    Output Files:
        - forecast_results.csv: 30-day forecast
        - detection_results.csv: Complete detection results with risk levels
        - detailed_metrics_matrix.csv: Comprehensive metrics matrix
        - Multiple visualization plots
    """)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] in ['-h', '--help', 'help']:
            show_usage()
        else:
            # Use command line argument
            data_file = sys.argv[1]
            system, forecast = run_analysis(data_file)
    else:
        # Use configured DATA_FILE
        print("No command line argument provided.")
        print(f"Using DATA_FILE from configuration: {DATA_FILE}\n")

        if DATA_FILE == 'your_data.csv':
            print("WARNING: Please update the DATA_FILE variable with your actual file path!")
            show_usage()
        else:
            system, forecast = run_analysis(DATA_FILE)