# Early Epidemic & Pandemic Detection System (EEPD)

## 🚨 Overview

**EEPD** is an advanced machine learning system for detecting and predicting epidemic and pandemic outbreaks across multiple diseases. Using state-of-the-art deep learning techniques, the system analyzes disease trends, detects anomalies, forecasts future trends, and provides real-time alerting capabilities.

### Key Features

✅ **Transformer-Based Anomaly Detection** - Detects unusual disease patterns using attention mechanisms  
✅ **LSTM Time Series Forecasting** - Predicts disease trends up to 30 days ahead  
✅ **Wave Detection & Prediction** - Identifies past waves and predicts upcoming outbreaks  
✅ **ICU/Ventilator Monitoring** - Tracks and forecasts healthcare capacity needs  
✅ **Multi-Disease Support** - Configurable for COVID-19, influenza, or any disease  
✅ **Interactive Web Dashboard** - Real-time visualization with Streamlit  
✅ **Automated Alerts** - 4-tier severity alerting system (Low, Medium, High, Critical)  
✅ **Comprehensive Reporting** - Detailed analysis with visualizations and metrics  

## 📊 System Architecture

The system is built on three main pillars:

1. **Anomaly Detection Module** - Transformer-based detection of unusual patterns
2. **Forecasting Module** - LSTM networks for time series prediction
3. **Analysis Module** - Trend detection, wave prediction, and ICU monitoring

## 🛠️ Tech Stack

- **Deep Learning**: PyTorch (Transformer, LSTM)
- **Data Processing**: Pandas, NumPy, Scikit-learn
- **Statistical Analysis**: SciPy, Seaborn
- **Visualization**: Matplotlib, Seaborn
- **Web Framework**: Streamlit
- **ML Models**: Isolation Forest, Random Forest Classifier

## 📋 Requirements

```
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
torch>=2.0.0
scikit-learn>=1.0.0
scipy>=1.7.0
streamlit>=1.28.0
```

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd Final\ AI\ project
```

### 2. Create Virtual Environment

**Windows:**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS/Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

### Option 1: Command Line Analysis (Recommended)

```bash
python run_analysis.py your_data.csv
```

Replace `your_data.csv` with your actual COVID-19 or disease data file.

### Option 2: Interactive Web Dashboard

```bash
streamlit run streamlit_app.py
```

This launches an interactive dashboard where you can:
- Upload CSV data files
- Configure analysis parameters
- View real-time predictions and alerts
- Download comprehensive reports

### Option 3: Python API

```python
import pandas as pd
from covid_trend_detection import COVIDTrendAnalysisSystem

# Load your data
df = pd.read_csv('your_data.csv')

# Initialize system
system = COVIDTrendAnalysisSystem(
    df=df,
    disease_name='COVID-19',
    window_size=14,
    sensitivity='medium'
)

# Run full analysis
results = system.run_full_analysis(train_models=True, epochs=50)

# Access results
forecast = results['forecast']
wave_predictions = results['wave_predictions']
icu_monitoring = results['icu_monitoring']
alerts = results['alerts']
```

## 📁 Project Structure

```
├── covid_trend_detection.py      # Core analysis system
├── streamlit_app.py               # Interactive web dashboard
├── run_analysis.py                # Command-line interface
├── train_and_save_models.py       # Model training utilities
├── predict_next_month.py          # Forecasting module
├── HOW_TO_RUN_FULL_ANALYSIS.md   # Detailed usage guide
├── requirements.txt               # Python dependencies
├── Boo.csv                        # Sample data
├── saved_models/                  # Pre-trained models
│   ├── lstm_model.pth
│   └── transformer_model.pth
└── README.md                      # This file
```

## 📊 Data Format

Your data should be in CSV format with the following columns:

```
date,confirmed_cases,deaths,recovered,icu_patients,ventilators_in_use
2023-01-01,150,5,100,10,2
2023-01-02,175,6,120,12,3
...
```

**Required columns:**
- `date` - Date of record (YYYY-MM-DD format)
- `confirmed_cases` - Number of confirmed cases
- `deaths` - Cumulative deaths
- `recovered` - Cumulative recovered

**Optional columns:**
- `icu_patients` - ICU bed count
- `ventilators_in_use` - Ventilator count

## 🎯 Analysis Capabilities

### 1. Anomaly Detection
- Uses Transformer networks to identify unusual disease patterns
- Trains on historical data to establish baselines
- Detects sudden spikes, drops, or abnormal behavior

### 2. Time Series Forecasting
- LSTM models predict 30-day disease trends
- Forecasts all key metrics (cases, deaths, recovered, ICU needs)
- Includes confidence intervals

### 3. Wave Detection
- Identifies historical outbreak waves
- Predicts future waves based on patterns
- Estimates peak timing and severity

### 4. Risk Assessment
- 4-tier severity system: Low, Medium, High, Critical
- Trend classification: Increasing, Decreasing, Stable
- Actionable alerts for healthcare systems

### 5. ICU Monitoring
- Current utilization rates
- Forecasted capacity needs
- Critical alerts when capacity exceeded

## 📈 Output Reports

After analysis, the system generates:

- **Trend Analysis** - Current and forecasted trends
- **Anomaly Details** - Flagged unusual values with explanations
- **Wave Analysis** - Detected and predicted waves
- **ICU Projections** - Healthcare capacity forecasts
- **Risk Summary** - Alert details by severity level
- **Visualizations** - Multiple diagnostic plots

## 🔧 Configuration

### In `run_analysis.py`:

```python
DATA_FILE = 'Boo.csv'           # Your data file
DISEASE_NAME = 'COVID-19'        # Disease name
EPOCHS = 50                       # Training epochs
SENSITIVITY = 'medium'            # 'low', 'medium', or 'high'
```

### Model Training

Train new models with custom parameters:

```bash
python train_and_save_models.py
```

## 📚 Documentation

For detailed usage instructions, see [HOW_TO_RUN_FULL_ANALYSIS.md](HOW_TO_RUN_FULL_ANALYSIS.md)

## 🔬 Research Foundation

This system is based on cutting-edge research:

- **TranAD**: Deep Transformer Networks for Anomaly Detection
- **LSTM**: Advanced sequence-to-sequence modeling
- **Ensemble Methods**: Combining multiple ML approaches
- **Statistical Anomaly Detection**: Time series analysis techniques

## 🤝 Contributing

Contributions are welcome! Areas for enhancement:
- Additional disease models
- Improved forecasting algorithms
- Enhanced visualizations
- API endpoints
- Mobile app integration

## 📝 License

This project is provided as-is for educational and research purposes.

## 👨‍💻 Author

Developed by Iftikhar Ul Hassan 

## 📞 Support

For issues, questions, or suggestions, please open an issue in the repository or contact the development team.

---

**Note**: This system is designed to assist healthcare professionals and epidemiologists in outbreak detection and response planning. It should be used in conjunction with official health guidelines and expert analysis.
