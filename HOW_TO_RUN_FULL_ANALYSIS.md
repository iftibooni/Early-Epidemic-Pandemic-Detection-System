# How to Run Full EEPD Analysis

## Quick Start

### Method 1: Command Line (Recommended)

```bash
cd "/Users/apple/Desktop/AI project"
python run_analysis.py your_data.csv
```

Replace `your_data.csv` with your actual data file.

### Method 2: Edit Configuration

1. Open `run_analysis.py`
2. Edit the `DATA_FILE` variable:
   ```python
   DATA_FILE = 'sample_data.csv'  # Your file path
   ```
3. Optionally edit other settings:
   ```python
   DISEASE_NAME = 'COVID-19'  # or 'Disease X', 'Disease Y', etc.
   EPOCHS = 50  # More epochs = better but slower
   SENSITIVITY = 'medium'  # 'low', 'medium', or 'high'
   ```
4. Run:
   ```bash
   python run_analysis.py
   ```

### Method 3: Python Script

```python
import pandas as pd
from covid_trend_detection import COVIDTrendAnalysisSystem

# Load your data
df = pd.read_csv('your_data.csv')

# Initialize system
system = COVIDTrendAnalysisSystem(
    df=df,
    disease_name='COVID-19',  # or 'Disease X', etc.
    window_size=14,
    sensitivity='medium'
)

# Run full analysis
results = system.run_full_analysis(
    train_models=True,
    epochs=50
)

# Access results
forecast = results['forecast']
wave_predictions = results['wave_predictions']
icu_monitoring = results['icu_monitoring']
```

## What the Full Analysis Includes

1. **Model Training** (if `train_models=True`):
   - Transformer-based anomaly detection
   - LSTM forecasting model
   - Takes 5-15 minutes depending on data size

2. **Anomaly & Trend Detection**:
   - Detects anomalies using Transformer
   - Identifies trends (increasing/decreasing/stable)
   - Classifies risk levels (4-tier system)

3. **Forecasting**:
   - 30-day forecast using LSTM
   - Predicts all key metrics

4. **Wave Prediction**:
   - Detects past waves
   - Predicts upcoming waves
   - Estimates peak timing

5. **ICU/Ventilator Monitoring**:
   - Current utilization status
   - Forecasted capacity needs
   - Critical alerts

6. **Comprehensive Reporting**:
   - Risk level distribution
   - Alert details by severity
   - Current trends and status

7. **Visualizations**:
   - Multiple plots showing trends
   - Anomaly detection
   - Alert timeline
   - Multi-metric overview

## Output Files

After running, you'll get:

1. **forecast_results.csv**: 30-day forecast for all metrics
2. **detection_results.csv**: Complete detection results with:
   - Risk levels
   - Risk scores
   - Anomaly detection
   - Alert triggers
   - ICU/Ventilator utilization
3. **detailed_metrics_matrix.csv**: Comprehensive metrics matrix with all variables
4. **Visualization plots**: Displayed automatically

## Data Requirements

Your CSV file must have these columns:
- **Date** (format: DD/MM/YYYY)
- **Test_Positivity_Rate**
- **New_Cases**
- **Case_Growth_Rate**
- **Hospitalization_Rate**
- **ICU_Admission_Rate**
- **Ventilator_Utilization**
- **Mortality_Rate**

**Minimum**: 7 days of data  
**Recommended**: 30+ days for better predictions

## Example

```bash
# Using sample data
python run_analysis.py sample_data.csv

# Or with custom disease name
# Edit run_analysis.py:
#   DISEASE_NAME = 'Disease X'
python run_analysis.py your_data.csv
```

## Troubleshooting

**Error: File not found**
- Check the file path
- Use absolute path if needed: `/full/path/to/file.csv`

**Error: Missing columns**
- Ensure all required columns are present
- Check column names match exactly (case-sensitive)

**Slow performance**
- Reduce `EPOCHS` (e.g., 30 instead of 50)
- Use smaller `WINDOW_SIZE` (e.g., 7 instead of 14)

**Memory issues**
- Use smaller dataset
- Reduce `EPOCHS`
- Close other applications

## Quick Test

To test with sample data:

```python
python -c "
from covid_trend_detection import COVIDTrendAnalysisSystem
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Create sample data
dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
df = pd.DataFrame({
    'Date': dates.strftime('%d/%m/%Y'),
    'Test_Positivity_Rate': np.random.uniform(0.03, 0.08, 30),
    'New_Cases': np.random.randint(50, 200, 30),
    'Case_Growth_Rate': np.random.uniform(-0.1, 0.1, 30),
    'Hospitalization_Rate': np.random.uniform(0.1, 0.2, 30),
    'ICU_Admission_Rate': np.random.uniform(0.02, 0.05, 30),
    'Ventilator_Utilization': np.random.uniform(0.2, 0.4, 30),
    'Mortality_Rate': np.random.uniform(0.005, 0.015, 30)
})

system = COVIDTrendAnalysisSystem(df=df, disease_name='Test Disease')
results = system.run_full_analysis(train_models=True, epochs=20)
print('✅ Analysis complete!')
"
```

## Need Help?

- Check `EEPD_FEATURES.md` for feature details
- See `run_analysis.py` for configuration options
- Review error messages for specific issues

