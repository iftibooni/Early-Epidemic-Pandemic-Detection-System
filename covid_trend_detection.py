"""
COVID-19 Trend Detection and Early Warning System
Based on state-of-the-art research:
- Transformer-based Anomaly Detection (TranAD)
- LSTM Time Series Forecasting
- Ensemble Methods for Trend Classification
- Statistical Anomaly Detection

References:
- Deep Learning for Time Series Anomaly Detection (ACM Computing Surveys)
- TranAD: Deep Transformer Networks for Anomaly Detection
- Detection of COVID-19 epidemic outbreak using machine learning
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Deep Learning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Sklearn
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Statistical Models
from scipy import stats
from scipy.signal import find_peaks
import scipy.stats as st

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# =====================================================
# 1. TRANSFORMER-BASED ANOMALY DETECTION (TranAD)
# =====================================================

class TransformerEncoder(nn.Module):
    """Transformer Encoder for time series anomaly detection"""
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.output_projection = nn.Linear(d_model, input_dim)

    def forward(self, src):
        src = self.input_projection(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.output_projection(output)
        return output


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        # Use sequence length (x.size(1)), not batch size
        seq_len = x.size(1)
        x = x + self.pe[:seq_len].transpose(0, 1)
        return self.dropout(x)


class TranAD(nn.Module):
    """
    TranAD: Transformer-based Anomaly Detection
    Based on: "TranAD: Deep Transformer Networks for Anomaly Detection in Multivariate Time Series"
    """
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2):
        super(TranAD, self).__init__()
        self.encoder1 = TransformerEncoder(input_dim, d_model, nhead, num_layers)
        self.encoder2 = TransformerEncoder(input_dim, d_model, nhead, num_layers)

    def forward(self, x, phase='train'):
        # First pass
        z1 = self.encoder1(x)
        # Second pass
        z2 = self.encoder2(z1)

        if phase == 'train':
            return z1, z2
        else:
            return z2


# =====================================================
# 2. LSTM-BASED FORECASTING MODEL
# =====================================================

class LSTMForecaster(nn.Module):
    """
    LSTM-based time series forecasting
    Based on: "Deep Learning for Time Series Anomaly Detection: A Survey"
    """
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.2):
        super(LSTMForecaster, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        predictions = self.fc(lstm_out)
        return predictions


class WavePredictorLSTM(nn.Module):
    """
    LSTM-based wave prediction model for epidemic waves
    Detects and predicts upcoming waves using LSTM
    """
    def __init__(self, input_dim, hidden_dim=128, num_layers=3, dropout=0.2):
        super(WavePredictorLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Bidirectional LSTM for better pattern recognition
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Output layers for wave prediction
        self.fc_forecast = nn.Linear(hidden_dim * 2, input_dim)  # *2 for bidirectional
        self.fc_wave_prob = nn.Linear(hidden_dim * 2, 1)  # Probability of wave
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        
        # Forecast values
        forecast = self.fc_forecast(lstm_out)
        
        # Wave probability (sigmoid for 0-1 range)
        wave_prob = torch.sigmoid(self.fc_wave_prob(lstm_out))
        
        return forecast, wave_prob


# =====================================================
# 3. ATTENTION-BASED LSTM
# =====================================================

class AttentionLSTM(nn.Module):
    """LSTM with Attention Mechanism for better long-term dependencies"""
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.2):
        super(AttentionLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Attention mechanism
        self.attention = nn.Linear(hidden_dim, 1)
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)

        # Apply attention
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attention_weights * lstm_out, dim=1)

        # Prediction
        predictions = self.fc(context).unsqueeze(1)
        return predictions


# =====================================================
# 4. TIME SERIES DATASET
# =====================================================

class TimeSeriesDataset(Dataset):
    """Custom dataset for time series data"""
    def __init__(self, data, window_size=14):
        self.data = data
        self.window_size = window_size

    def __len__(self):
        return len(self.data) - self.window_size

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.window_size]
        y = self.data[idx + self.window_size]
        return torch.FloatTensor(x), torch.FloatTensor(y)


# =====================================================
# 5. EEPD ALERT SYSTEM WITH 4-TIER RISK CLASSIFICATION
# =====================================================

class EEPDAlertSystem:
    """
    Early Epidemic & Pandemic Detection Alert System
    Classifies risk levels: Normal, Elevated Risk, High Alert, Critical/Outbreak Likely
    """
    def __init__(self, sensitivity='medium'):
        self.sensitivity = sensitivity
        
        # Risk level thresholds based on growth rates and consecutive days
        self.risk_thresholds = {
            'Normal': {
                'growth_rate_max': 0.02,
                'consecutive_days_max': 2,
                'icu_utilization_max': 0.5,
                'ventilator_utilization_max': 0.4
            },
            'Elevated Risk': {
                'growth_rate_min': 0.02,
                'growth_rate_max': 0.05,
                'consecutive_days_min': 2,
                'consecutive_days_max': 4,
                'icu_utilization_min': 0.5,
                'icu_utilization_max': 0.7,
                'ventilator_utilization_min': 0.4,
                'ventilator_utilization_max': 0.6
            },
            'High Alert': {
                'growth_rate_min': 0.05,
                'growth_rate_max': 0.10,
                'consecutive_days_min': 4,
                'consecutive_days_max': 7,
                'icu_utilization_min': 0.7,
                'icu_utilization_max': 0.85,
                'ventilator_utilization_min': 0.6,
                'ventilator_utilization_max': 0.75
            },
            'Critical / Outbreak Likely': {
                'growth_rate_min': 0.10,
                'consecutive_days_min': 7,
                'icu_utilization_min': 0.85,
                'ventilator_utilization_min': 0.75
            }
        }

    def detect_trend(self, values, window=7):
        """Detect trend using multiple methods"""
        if len(values) < window:
            return 'insufficient_data', 0.0

        recent = values[-window:]

        # Method 1: Linear regression slope
        x = np.arange(len(recent))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, recent)

        # Method 2: Moving average comparison
        if len(values) >= window * 2:
            older_avg = np.mean(values[-(window*2):-window])
            recent_avg = np.mean(recent)
            pct_change = (recent_avg - older_avg) / (older_avg + 1e-8)
        else:
            pct_change = 0

        # Method 3: Exponential weighted moving average
        ewma = pd.Series(values).ewm(span=window).mean()
        ewma_trend = ewma.iloc[-1] - ewma.iloc[-window] if len(ewma) >= window else 0

        # Combined decision
        avg_growth = (abs(slope) + abs(pct_change)) / 2

        if slope > 0.01 or pct_change > 0.01:
            trend = 'increasing'
        elif slope < -0.01 or pct_change < -0.01:
            trend = 'decreasing'
        else:
            trend = 'stable'

        confidence = abs(r_value) if r_value else 0

        return trend, confidence, avg_growth

    def classify_risk_level(self, metrics_data, disease_name='Unknown'):
        """
        Classify risk level based on multiple indicators
        
        Args:
            metrics_data: dict with keys like 'growth_rate', 'consecutive_days', 
                         'icu_utilization', 'ventilator_utilization', 'anomaly_score'
            disease_name: Name of the disease being monitored
        
        Returns:
            dict with 'risk_level', 'confidence', 'indicators', 'message'
        """
        growth_rate = metrics_data.get('growth_rate', 0)
        consecutive_days = metrics_data.get('consecutive_days', 0)
        icu_util = metrics_data.get('icu_utilization', 0)
        vent_util = metrics_data.get('ventilator_utilization', 0)
        anomaly_score = metrics_data.get('anomaly_score', 0)
        predicted_cases = metrics_data.get('predicted_cases_increase', 0)
        
        # Calculate risk score (0-100)
        risk_score = 0
        
        # Growth rate contribution (0-40 points)
        if growth_rate >= 0.10:
            risk_score += 40
        elif growth_rate >= 0.05:
            risk_score += 30
        elif growth_rate >= 0.02:
            risk_score += 15
        
        # Consecutive days contribution (0-20 points)
        if consecutive_days >= 7:
            risk_score += 20
        elif consecutive_days >= 4:
            risk_score += 12
        elif consecutive_days >= 2:
            risk_score += 6
        
        # ICU utilization contribution (0-20 points)
        if icu_util >= 0.85:
            risk_score += 20
        elif icu_util >= 0.7:
            risk_score += 12
        elif icu_util >= 0.5:
            risk_score += 6
        
        # Ventilator utilization contribution (0-10 points)
        if vent_util >= 0.75:
            risk_score += 10
        elif vent_util >= 0.6:
            risk_score += 6
        elif vent_util >= 0.4:
            risk_score += 3
        
        # Anomaly score contribution (0-10 points)
        if anomaly_score > 0.8:
            risk_score += 10
        elif anomaly_score > 0.5:
            risk_score += 5
        
        # Classify based on risk score
        if risk_score >= 80:
            risk_level = 'Critical / Outbreak Likely'
            confidence = min(0.95, 0.7 + (risk_score - 80) / 100)
        elif risk_score >= 60:
            risk_level = 'High Alert'
            confidence = min(0.9, 0.6 + (risk_score - 60) / 100)
        elif risk_score >= 30:
            risk_level = 'Elevated Risk'
            confidence = min(0.8, 0.5 + (risk_score - 30) / 100)
        else:
            risk_level = 'Normal'
            confidence = max(0.5, 1.0 - risk_score / 100)
        
        # Generate message
        indicators = []
        if growth_rate > 0.05:
            indicators.append(f"High growth rate ({growth_rate*100:.1f}%)")
        if consecutive_days >= 4:
            indicators.append(f"{consecutive_days} consecutive days of increase")
        if icu_util > 0.7:
            indicators.append(f"ICU utilization at {icu_util*100:.1f}%")
        if vent_util > 0.6:
            indicators.append(f"Ventilator utilization at {vent_util*100:.1f}%")
        if anomaly_score > 0.5:
            indicators.append(f"Anomaly detected (score: {anomaly_score:.2f})")
        
        message = f"{disease_name}: {risk_level}"
        if indicators:
            message += f" - {'; '.join(indicators)}"
        
        return {
            'risk_level': risk_level,
            'risk_score': risk_score,
            'confidence': confidence,
            'indicators': indicators,
            'message': message,
            'growth_rate': growth_rate,
            'consecutive_days': consecutive_days,
            'icu_utilization': icu_util,
            'ventilator_utilization': vent_util
        }

    def check_alert(self, trends, metrics, disease_name='Unknown'):
        """Check if alert should be triggered and classify risk level"""
        consecutive = 0
        if len(trends) > 0:
            # Count consecutive increasing trends
            for i in range(len(trends) - 1, -1, -1):
                if trends[i] == 'increasing':
                    consecutive += 1
                else:
                    break
        
        # Calculate growth rate from metrics
        growth_rate = 0
        if 'growth_rate' in metrics:
            growth_rate = metrics['growth_rate']
        elif 'recent_growth' in metrics:
            growth_rate = metrics['recent_growth']
        
        # Get ICU and ventilator utilization
        icu_util = metrics.get('icu_utilization', 0)
        vent_util = metrics.get('ventilator_utilization', 0)
        anomaly_score = metrics.get('anomaly_score', 0)
        
        metrics_data = {
            'growth_rate': growth_rate,
            'consecutive_days': consecutive,
            'icu_utilization': icu_util,
            'ventilator_utilization': vent_util,
            'anomaly_score': anomaly_score
        }
        
        risk_assessment = self.classify_risk_level(metrics_data, disease_name)
        
        # Determine if alert should be triggered
        alert_triggered = risk_assessment['risk_level'] != 'Normal'
        
        return alert_triggered, risk_assessment


class TrendDetector:
    """
    Legacy trend detector (kept for backward compatibility)
    """
    def __init__(self, sensitivity='medium'):
        self.eepd_system = EEPDAlertSystem(sensitivity)
    
    def detect_trend(self, values, window=7):
        trend, confidence, _ = self.eepd_system.detect_trend(values, window)
        return trend, confidence
    
    def check_alert(self, trends, metrics):
        alert, assessment = self.eepd_system.check_alert(trends, metrics)
        return alert, assessment.get('message', '')


# =====================================================
# 6. ENSEMBLE EARLY WARNING SYSTEM
# =====================================================

class EnsembleEarlyWarning:
    """
    Ensemble-based early warning system combining multiple models
    Based on: "Detection of COVID-19 epidemic outbreak using machine learning"
    Uses Random Forest, XGBoost-style gradient boosting, and statistical methods
    """
    def __init__(self):
        self.rf_classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.isolation_forest = IsolationForest(
            contamination=0.1,
            random_state=42
        )
        self.scaler = StandardScaler()

    def extract_features(self, window_data):
        """Extract statistical features from time series window"""
        features = []

        for i in range(window_data.shape[1]):  # For each metric
            col = window_data[:, i]

            # Statistical features
            features.extend([
                np.mean(col),
                np.std(col),
                np.min(col),
                np.max(col),
                np.median(col),
                stats.skew(col),
                stats.kurtosis(col),
            ])

            # Trend features
            if len(col) > 1:
                slope, _ = np.polyfit(range(len(col)), col, 1)
                features.append(slope)
            else:
                features.append(0)

            # Change features
            if len(col) > 1:
                pct_change = (col[-1] - col[0]) / (col[0] + 1e-8)
                features.append(pct_change)
            else:
                features.append(0)

        return features

    def train_classifier(self, X_train, y_train):
        """Train the Random Forest classifier"""
        X_scaled = self.scaler.fit_transform(X_train)
        self.rf_classifier.fit(X_scaled, y_train)

    def predict_trend(self, X):
        """Predict trend class: 0=decrease, 1=stable, 2=increase"""
        X_scaled = self.scaler.transform(X)
        return self.rf_classifier.predict(X_scaled)

    def detect_anomalies(self, X):
        """Detect anomalies using Isolation Forest"""
        return self.isolation_forest.fit_predict(X)


# =====================================================
# 7. WAVE DETECTION AND PREDICTION
# =====================================================

class WaveDetector:
    """
    Detects and predicts epidemic waves using LSTM and statistical methods
    """
    def __init__(self, window_size=14):
        self.window_size = window_size
    
    def detect_waves(self, cases_data):
        """
        Detect past waves in the data using peak detection
        Returns list of wave periods with start, peak, end dates
        """
        if len(cases_data) < self.window_size * 2:
            return []
        
        # Find peaks using scipy
        peaks, properties = find_peaks(
            cases_data, 
            distance=self.window_size,
            prominence=np.std(cases_data) * 0.5
        )
        
        waves = []
        for peak_idx in peaks:
            # Find wave start (local minimum before peak)
            start_idx = max(0, peak_idx - self.window_size * 2)
            before_peak = cases_data[start_idx:peak_idx]
            if len(before_peak) > 0:
                min_before = np.argmin(before_peak) + start_idx
            else:
                min_before = start_idx
            
            # Find wave end (local minimum after peak)
            end_idx = min(len(cases_data), peak_idx + self.window_size * 2)
            after_peak = cases_data[peak_idx:end_idx]
            if len(after_peak) > 0:
                min_after = np.argmin(after_peak) + peak_idx
            else:
                min_after = end_idx
            
            waves.append({
                'start': min_before,
                'peak': peak_idx,
                'end': min_after,
                'peak_value': cases_data[peak_idx]
            })
        
        return waves
    
    def predict_next_wave(self, cases_data, forecast_data, model=None):
        """
        Predict if a new wave is coming based on current trends and LSTM forecast
        """
        if len(cases_data) < self.window_size:
            return {'wave_likely': False, 'confidence': 0.0, 'estimated_peak_days': None}
        
        # Analyze recent trend
        recent = cases_data[-self.window_size:]
        older = cases_data[-(self.window_size*2):-self.window_size] if len(cases_data) >= self.window_size*2 else recent
        
        recent_avg = np.mean(recent)
        older_avg = np.mean(older)
        growth_rate = (recent_avg - older_avg) / (older_avg + 1e-8)
        
        # Analyze forecast trend
        if forecast_data is not None and len(forecast_data) > 0:
            forecast_growth = (forecast_data[-1] - forecast_data[0]) / (forecast_data[0] + 1e-8)
        else:
            forecast_growth = 0
        
        # Calculate wave probability
        wave_prob = 0.0
        
        # Growth rate indicator
        if growth_rate > 0.1:
            wave_prob += 0.4
        elif growth_rate > 0.05:
            wave_prob += 0.25
        elif growth_rate > 0.02:
            wave_prob += 0.1
        
        # Forecast growth indicator
        if forecast_growth > 0.15:
            wave_prob += 0.4
        elif forecast_growth > 0.08:
            wave_prob += 0.25
        elif forecast_growth > 0.03:
            wave_prob += 0.1
        
        # Check if we're in a low period (between waves)
        current_level = cases_data[-1]
        historical_avg = np.mean(cases_data)
        if current_level < historical_avg * 0.7:
            wave_prob += 0.2  # More likely to start a new wave from low base
        
        wave_prob = min(1.0, wave_prob)
        
        # Estimate peak timing (rough estimate based on growth rate)
        estimated_peak_days = None
        if wave_prob > 0.5 and growth_rate > 0:
            # Rough estimate: days to reach peak assuming exponential growth
            target_multiplier = 2.0  # Assume wave peaks at 2x current level
            estimated_peak_days = int(np.log(target_multiplier) / np.log(1 + growth_rate)) if growth_rate > 0 else None
        
        return {
            'wave_likely': wave_prob > 0.5,
            'confidence': wave_prob,
            'estimated_peak_days': estimated_peak_days,
            'growth_rate': growth_rate,
            'forecast_growth': forecast_growth
        }


# =====================================================
# 8. COMPLETE EEPD SYSTEM (Multi-Disease Support)
# =====================================================

class EEPDSystem:
    """
    Early Epidemic & Pandemic Detection System (EEPD)
    Multi-disease outbreak detection with LSTM-based predictions
    Supports: COVID-19, Disease X, Disease Y, and other infectious diseases
    """
    def __init__(self, data_path=None, df=None, window_size=14, sensitivity='medium', disease_name='COVID-19'):
        """
        Initialize the EEPD system

        Args:
            data_path: Path to CSV file
            df: DataFrame (if already loaded)
            window_size: Window size for sequence prediction
            sensitivity: Alert sensitivity ('low', 'medium', 'high')
            disease_name: Name of the disease being monitored (e.g., 'COVID-19', 'Disease X', 'Disease Y')
        """
        self.window_size = window_size
        self.sensitivity = sensitivity
        self.disease_name = disease_name

        # Load data
        if df is not None:
            self.df = df
        elif data_path:
            self.df = pd.read_csv(data_path)
        else:
            raise ValueError("Either data_path or df must be provided")

        self.prepare_data()

        # Initialize models
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        print(f"Monitoring disease: {self.disease_name}")

        # Initialize all components
        self.eepd_alert_system = EEPDAlertSystem(sensitivity=sensitivity)
        self.trend_detector = TrendDetector(sensitivity=sensitivity)
        self.ensemble_system = EnsembleEarlyWarning()
        self.wave_detector = WaveDetector(window_size=window_size)

        self.models = {}
        self.scalers = {}
        self.results = {}
        self.wave_predictions = {}
        self.icu_monitoring = {}


class COVIDTrendAnalysisSystem:
    """Complete system for COVID-19 trend analysis and early warning (Legacy - use EEPDSystem for new projects)"""

    def __init__(self, data_path=None, df=None, window_size=14, sensitivity='medium', disease_name='COVID-19'):
        """
        Initialize the system

        Args:
            data_path: Path to CSV file
            df: DataFrame (if already loaded)
            window_size: Window size for sequence prediction
            sensitivity: Alert sensitivity ('low', 'medium', 'high')
            disease_name: Name of the disease (default: 'COVID-19')
        """
        self.window_size = window_size
        self.sensitivity = sensitivity
        self.disease_name = disease_name

        # Load data
        if df is not None:
            self.df = df
        elif data_path:
            self.df = pd.read_csv(data_path)
        else:
            raise ValueError("Either data_path or df must be provided")

        self.prepare_data()

        # Initialize models
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Initialize all components
        self.eepd_alert_system = EEPDAlertSystem(sensitivity=sensitivity)
        self.trend_detector = TrendDetector(sensitivity=sensitivity)
        self.ensemble_system = EnsembleEarlyWarning()
        self.wave_detector = WaveDetector(window_size=window_size)

        self.models = {}
        self.scalers = {}
        self.results = {}
        self.wave_predictions = {}
        self.icu_monitoring = {}
        
        # Try to load pre-trained models if they exist (silently)
        try:
            self.load_models()
        except:
            pass  # Models don't exist yet, that's okay

    def prepare_data(self):
        """Prepare and clean data"""
        # Convert date
        self.df['Date'] = pd.to_datetime(self.df['Date'], format='%d/%m/%Y')
        self.df = self.df.sort_values('Date').reset_index(drop=True)

        # Select numeric columns
        self.feature_cols = [
            'Test_Positivity_Rate', 'New_Cases', 'Case_Growth_Rate',
            'Hospitalization_Rate', 'ICU_Admission_Rate',
            'Ventilator_Utilization', 'Mortality_Rate'
        ]

        # Handle missing values
        self.df[self.feature_cols] = self.df[self.feature_cols].fillna(0)

        print(f"Data shape: {self.df.shape}")
        print(f"Date range: {self.df['Date'].min()} to {self.df['Date'].max()}")

    def train_transformer_model(self, epochs=50, batch_size=32, lr=0.001):
        """Train Transformer-based anomaly detection model"""
        print("\n" + "="*60)
        print("Training Transformer-based Anomaly Detection (TranAD)")
        print("="*60)

        # Prepare data
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(self.df[self.feature_cols].values)

        # Create dataset
        dataset = TimeSeriesDataset(data_scaled, self.window_size)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Initialize model
        input_dim = len(self.feature_cols)
        model = TranAD(input_dim=input_dim, d_model=64, nhead=4, num_layers=2)
        model = model.to(self.device)

        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        # Training loop
        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            model.train()
            train_loss = 0
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)

                optimizer.zero_grad()
                z1, z2 = model(batch_x, phase='train')

                loss1 = criterion(z1, batch_x)
                loss2 = criterion(z2, batch_x)
                loss = loss1 + loss2

                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)
            train_losses.append(train_loss)

            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(self.device)
                    output = model(batch_x, phase='test')
                    loss = criterion(output, batch_x)
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            val_losses.append(val_loss)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        self.models['transformer'] = model
        self.scalers['transformer'] = scaler

        return train_losses, val_losses

    def train_lstm_model(self, epochs=50, batch_size=32, lr=0.001):
        """Train LSTM forecasting model"""
        print("\n" + "="*60)
        print("Training LSTM Forecasting Model")
        print("="*60)

        # Prepare data
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(self.df[self.feature_cols].values)

        # Create dataset
        dataset = TimeSeriesDataset(data_scaled, self.window_size)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Initialize model
        input_dim = len(self.feature_cols)
        model = LSTMForecaster(input_dim=input_dim, hidden_dim=128, num_layers=2)
        model = model.to(self.device)

        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        # Training loop
        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            model.train()
            train_loss = 0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()
                output = model(batch_x)
                loss = criterion(output[:, -1, :], batch_y)

                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)
            train_losses.append(train_loss)

            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    output = model(batch_x)
                    loss = criterion(output[:, -1, :], batch_y)
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            val_losses.append(val_loss)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        self.models['lstm'] = model
        self.scalers['lstm'] = scaler
        
        # Store last forecast for evaluation
        self.last_forecast = None

        return train_losses, val_losses

    def calculate_evaluation_metrics(self):
        """
        Calculate comprehensive evaluation metrics for all models
        Returns a dictionary with evaluation metrics
        """
        eval_metrics = {}
        
        # Transformer Model Metrics
        if 'transformer' in self.models and 'transformer' in self.scalers:
            try:
                # Prepare validation data
                scaler = self.scalers['transformer']
                data_scaled = scaler.transform(self.df[self.feature_cols].values)
                
                dataset = TimeSeriesDataset(data_scaled, self.window_size)
                val_size = int(0.2 * len(dataset))
                train_size = len(dataset) - val_size
                _, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
                val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
                
                model = self.models['transformer']
                model.eval()
                
                total_mse = 0
                total_mae = 0
                count = 0
                
                with torch.no_grad():
                    for batch_x, _ in val_loader:
                        batch_x = batch_x.to(self.device)
                        output = model(batch_x, phase='test')
                        
                        mse = torch.nn.functional.mse_loss(output, batch_x)
                        mae = torch.nn.functional.l1_loss(output, batch_x)
                        
                        total_mse += mse.item()
                        total_mae += mae.item()
                        count += 1
                
                if count > 0:
                    eval_metrics['transformer'] = {
                        'MSE': total_mse / count,
                        'MAE': total_mae / count,
                        'RMSE': np.sqrt(total_mse / count),
                        'Model_Type': 'Transformer (TranAD)',
                        'Status': 'Trained'
                    }
            except Exception as e:
                eval_metrics['transformer'] = {
                    'MSE': 'N/A',
                    'MAE': 'N/A',
                    'RMSE': 'N/A',
                    'Model_Type': 'Transformer (TranAD)',
                    'Status': f'Error: {str(e)[:50]}'
                }
        else:
            eval_metrics['transformer'] = {
                'MSE': 'N/A',
                'MAE': 'N/A',
                'RMSE': 'N/A',
                'Model_Type': 'Transformer (TranAD)',
                'Status': 'Not Trained'
            }
        
        # LSTM Model Metrics
        if 'lstm' in self.models and 'lstm' in self.scalers:
            try:
                # Prepare validation data
                scaler = self.scalers['lstm']
                data_scaled = scaler.transform(self.df[self.feature_cols].values)
                
                dataset = TimeSeriesDataset(data_scaled, self.window_size)
                val_size = int(0.2 * len(dataset))
                train_size = len(dataset) - val_size
                _, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
                val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
                
                model = self.models['lstm']
                model.eval()
                
                total_mse = 0
                total_mae = 0
                total_mape = 0
                count = 0
                
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                        output = model(batch_x)
                        predictions = output[:, -1, :]
                        
                        mse = torch.nn.functional.mse_loss(predictions, batch_y)
                        mae = torch.nn.functional.l1_loss(predictions, batch_y)
                        
                        # MAPE (Mean Absolute Percentage Error)
                        mape = torch.mean(torch.abs((batch_y - predictions) / (batch_y + 1e-8))) * 100
                        
                        total_mse += mse.item()
                        total_mae += mae.item()
                        total_mape += mape.item()
                        count += 1
                
                if count > 0:
                    eval_metrics['lstm'] = {
                        'MSE': total_mse / count,
                        'MAE': total_mae / count,
                        'RMSE': np.sqrt(total_mse / count),
                        'MAPE': total_mape / count,
                        'Model_Type': 'LSTM Forecaster',
                        'Status': 'Trained'
                    }
            except Exception as e:
                eval_metrics['lstm'] = {
                    'MSE': 'N/A',
                    'MAE': 'N/A',
                    'RMSE': 'N/A',
                    'MAPE': 'N/A',
                    'Model_Type': 'LSTM Forecaster',
                    'Status': f'Error: {str(e)[:50]}'
                }
        else:
            eval_metrics['lstm'] = {
                'MSE': 'N/A',
                'MAE': 'N/A',
                'RMSE': 'N/A',
                'MAPE': 'N/A',
                'Model_Type': 'LSTM Forecaster',
                'Status': 'Not Trained'
            }
        
        # Anomaly Detection Performance
        if 'anomalies' in self.results and len(self.results['anomalies']) > 0:
            anomalies = self.results['anomalies']
            total_days = len(anomalies)
            anomaly_count = sum(anomalies)
            anomaly_rate = (anomaly_count / total_days) * 100 if total_days > 0 else 0
            
            eval_metrics['anomaly_detection'] = {
                'Total_Days_Analyzed': total_days,
                'Anomalies_Detected': anomaly_count,
                'Anomaly_Rate_%': f"{anomaly_rate:.2f}",
                'Normal_Days': total_days - anomaly_count,
                'Status': 'Active'
            }
        else:
            eval_metrics['anomaly_detection'] = {
                'Total_Days_Analyzed': 0,
                'Anomalies_Detected': 0,
                'Anomaly_Rate_%': 'N/A',
                'Normal_Days': 0,
                'Status': 'Not Analyzed'
            }
        
        # Risk Classification Performance
        if 'risk_levels' in self.results and len(self.results['risk_levels']) > 0:
            risk_levels = self.results['risk_levels']
            risk_counts = {}
            for level in risk_levels:
                risk_counts[level] = risk_counts.get(level, 0) + 1
            
            total_risk_days = len(risk_levels)
            eval_metrics['risk_classification'] = {
                'Total_Days': total_risk_days,
                'Normal': risk_counts.get('Normal', 0),
                'Elevated_Risk': risk_counts.get('Elevated Risk', 0),
                'High_Alert': risk_counts.get('High Alert', 0),
                'Critical': risk_counts.get('Critical / Outbreak Likely', 0),
                'Status': 'Active'
            }
        else:
            eval_metrics['risk_classification'] = {
                'Total_Days': 0,
                'Normal': 0,
                'Elevated_Risk': 0,
                'High_Alert': 0,
                'Critical': 0,
                'Status': 'Not Analyzed'
            }
        
        # Alert System Performance
        if 'alert_triggered' in self.results and len(self.results['alert_triggered']) > 0:
            alerts = self.results['alert_triggered']
            total_alerts = sum(alerts)
            alert_rate = (total_alerts / len(alerts)) * 100 if len(alerts) > 0 else 0
            
            eval_metrics['alert_system'] = {
                'Total_Days': len(alerts),
                'Alerts_Triggered': total_alerts,
                'Alert_Rate_%': f"{alert_rate:.2f}",
                'No_Alert_Days': len(alerts) - total_alerts,
                'Status': 'Active'
            }
        else:
            eval_metrics['alert_system'] = {
                'Total_Days': 0,
                'Alerts_Triggered': 0,
                'Alert_Rate_%': 'N/A',
                'No_Alert_Days': 0,
                'Status': 'Not Analyzed'
            }
        
        # Forecast Accuracy (if forecast exists)
        if hasattr(self, 'last_forecast') and self.last_forecast is not None:
            try:
                forecast = self.last_forecast
                eval_metrics['forecast'] = {
                    'Forecast_Days': len(forecast),
                    'Metrics_Forecasted': len([col for col in forecast.columns if col != 'Date']),
                    'Status': 'Available'
                }
            except:
                eval_metrics['forecast'] = {
                    'Forecast_Days': 0,
                    'Metrics_Forecasted': 0,
                    'Status': 'Not Available'
                }
        else:
            eval_metrics['forecast'] = {
                'Forecast_Days': 0,
                'Metrics_Forecasted': 0,
                'Status': 'Not Generated'
            }
        
        return eval_metrics

    def get_evaluation_matrix(self):
        """
        Get evaluation matrix as a pandas DataFrame
        """
        eval_metrics = self.calculate_evaluation_metrics()
        
        matrix_data = []
        
        # Transformer metrics
        if 'transformer' in eval_metrics:
            tf_metrics = eval_metrics['transformer']
            matrix_data.append({
                'Model/System': 'Transformer (Anomaly Detection)',
                'Metric': 'MSE',
                'Value': f"{tf_metrics['MSE']:.6f}" if isinstance(tf_metrics['MSE'], (int, float)) else str(tf_metrics['MSE']),
                'Status': tf_metrics['Status']
            })
            matrix_data.append({
                'Model/System': 'Transformer (Anomaly Detection)',
                'Metric': 'MAE',
                'Value': f"{tf_metrics['MAE']:.6f}" if isinstance(tf_metrics['MAE'], (int, float)) else str(tf_metrics['MAE']),
                'Status': tf_metrics['Status']
            })
            matrix_data.append({
                'Model/System': 'Transformer (Anomaly Detection)',
                'Metric': 'RMSE',
                'Value': f"{tf_metrics['RMSE']:.6f}" if isinstance(tf_metrics['RMSE'], (int, float)) else str(tf_metrics['RMSE']),
                'Status': tf_metrics['Status']
            })
        
        # LSTM metrics
        if 'lstm' in eval_metrics:
            lstm_metrics = eval_metrics['lstm']
            matrix_data.append({
                'Model/System': 'LSTM (Forecasting)',
                'Metric': 'MSE',
                'Value': f"{lstm_metrics['MSE']:.6f}" if isinstance(lstm_metrics['MSE'], (int, float)) else str(lstm_metrics['MSE']),
                'Status': lstm_metrics['Status']
            })
            matrix_data.append({
                'Model/System': 'LSTM (Forecasting)',
                'Metric': 'MAE',
                'Value': f"{lstm_metrics['MAE']:.6f}" if isinstance(lstm_metrics['MAE'], (int, float)) else str(lstm_metrics['MAE']),
                'Status': lstm_metrics['Status']
            })
            matrix_data.append({
                'Model/System': 'LSTM (Forecasting)',
                'Metric': 'RMSE',
                'Value': f"{lstm_metrics['RMSE']:.6f}" if isinstance(lstm_metrics['RMSE'], (int, float)) else str(lstm_metrics['RMSE']),
                'Status': lstm_metrics['Status']
            })
            matrix_data.append({
                'Model/System': 'LSTM (Forecasting)',
                'Metric': 'MAPE (%)',
                'Value': f"{lstm_metrics['MAPE']:.2f}" if isinstance(lstm_metrics['MAPE'], (int, float)) else str(lstm_metrics['MAPE']),
                'Status': lstm_metrics['Status']
            })
        
        # Anomaly Detection
        if 'anomaly_detection' in eval_metrics:
            ad_metrics = eval_metrics['anomaly_detection']
            matrix_data.append({
                'Model/System': 'Anomaly Detection',
                'Metric': 'Total Days Analyzed',
                'Value': str(ad_metrics['Total_Days_Analyzed']),
                'Status': ad_metrics['Status']
            })
            matrix_data.append({
                'Model/System': 'Anomaly Detection',
                'Metric': 'Anomalies Detected',
                'Value': str(ad_metrics['Anomalies_Detected']),
                'Status': ad_metrics['Status']
            })
            matrix_data.append({
                'Model/System': 'Anomaly Detection',
                'Metric': 'Anomaly Rate (%)',
                'Value': str(ad_metrics['Anomaly_Rate_%']),
                'Status': ad_metrics['Status']
            })
        
        # Risk Classification
        if 'risk_classification' in eval_metrics:
            rc_metrics = eval_metrics['risk_classification']
            matrix_data.append({
                'Model/System': 'Risk Classification',
                'Metric': 'Total Days',
                'Value': str(rc_metrics['Total_Days']),
                'Status': rc_metrics['Status']
            })
            matrix_data.append({
                'Model/System': 'Risk Classification',
                'Metric': 'Normal Days',
                'Value': str(rc_metrics['Normal']),
                'Status': rc_metrics['Status']
            })
            matrix_data.append({
                'Model/System': 'Risk Classification',
                'Metric': 'Elevated Risk Days',
                'Value': str(rc_metrics['Elevated_Risk']),
                'Status': rc_metrics['Status']
            })
            matrix_data.append({
                'Model/System': 'Risk Classification',
                'Metric': 'High Alert Days',
                'Value': str(rc_metrics['High_Alert']),
                'Status': rc_metrics['Status']
            })
            matrix_data.append({
                'Model/System': 'Risk Classification',
                'Metric': 'Critical Days',
                'Value': str(rc_metrics['Critical']),
                'Status': rc_metrics['Status']
            })
        
        # Alert System
        if 'alert_system' in eval_metrics:
            as_metrics = eval_metrics['alert_system']
            matrix_data.append({
                'Model/System': 'Alert System',
                'Metric': 'Total Days',
                'Value': str(as_metrics['Total_Days']),
                'Status': as_metrics['Status']
            })
            matrix_data.append({
                'Model/System': 'Alert System',
                'Metric': 'Alerts Triggered',
                'Value': str(as_metrics['Alerts_Triggered']),
                'Status': as_metrics['Status']
            })
            matrix_data.append({
                'Model/System': 'Alert System',
                'Metric': 'Alert Rate (%)',
                'Value': str(as_metrics['Alert_Rate_%']),
                'Status': as_metrics['Status']
            })
        
        # Forecast
        if 'forecast' in eval_metrics:
            fc_metrics = eval_metrics['forecast']
            matrix_data.append({
                'Model/System': 'Forecast System',
                'Metric': 'Forecast Days',
                'Value': str(fc_metrics['Forecast_Days']),
                'Status': fc_metrics['Status']
            })
            matrix_data.append({
                'Model/System': 'Forecast System',
                'Metric': 'Metrics Forecasted',
                'Value': str(fc_metrics['Metrics_Forecasted']),
                'Status': fc_metrics['Status']
            })
        
        if not matrix_data:
            return None
        
        eval_df = pd.DataFrame(matrix_data)
        return eval_df

    def save_models(self, model_dir='saved_models'):
        """Save trained models and scalers to disk"""
        import os
        import pickle
        
        os.makedirs(model_dir, exist_ok=True)
        
        # Save models
        for model_name, model in self.models.items():
            if model is not None:
                model_path = os.path.join(model_dir, f'{model_name}_model.pth')
                torch.save(model.state_dict(), model_path)
                print(f"✅ Saved {model_name} model to {model_path}")
        
        # Save scalers
        for scaler_name, scaler in self.scalers.items():
            if scaler is not None:
                scaler_path = os.path.join(model_dir, f'{scaler_name}_scaler.pkl')
                with open(scaler_path, 'wb') as f:
                    pickle.dump(scaler, f)
                print(f"✅ Saved {scaler_name} scaler to {scaler_path}")
        
        # Save metadata
        metadata = {
            'window_size': self.window_size,
            'sensitivity': self.sensitivity,
            'disease_name': self.disease_name,
            'feature_cols': self.feature_cols,
            'device': str(self.device)
        }
        metadata_path = os.path.join(model_dir, 'metadata.pkl')
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        print(f"✅ Saved metadata to {metadata_path}")
        
        return True

    def load_models(self, model_dir='saved_models'):
        """Load pre-trained models and scalers from disk"""
        import os
        import pickle
        
        if not os.path.exists(model_dir):
            print(f"❌ Model directory {model_dir} does not exist")
            return False
        
        # Load metadata
        metadata_path = os.path.join(model_dir, 'metadata.pkl')
        if not os.path.exists(metadata_path):
            print(f"❌ Metadata file not found at {metadata_path}")
            return False
        
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        # Verify compatibility
        if metadata['window_size'] != self.window_size:
            print(f"⚠️ Warning: Window size mismatch. Model: {metadata['window_size']}, Current: {self.window_size}")
        if metadata['feature_cols'] != self.feature_cols:
            print(f"⚠️ Warning: Feature columns mismatch. Model: {metadata['feature_cols']}, Current: {self.feature_cols}")
        
        # Load Transformer model
        transformer_path = os.path.join(model_dir, 'transformer_model.pth')
        if os.path.exists(transformer_path):
            input_dim = len(self.feature_cols)
            model = TranAD(input_dim=input_dim, d_model=64, nhead=4, num_layers=2)
            model.load_state_dict(torch.load(transformer_path, map_location=self.device))
            model = model.to(self.device)
            model.eval()
            self.models['transformer'] = model
            print(f"✅ Loaded Transformer model from {transformer_path}")
        
        # Load Transformer scaler
        transformer_scaler_path = os.path.join(model_dir, 'transformer_scaler.pkl')
        if os.path.exists(transformer_scaler_path):
            with open(transformer_scaler_path, 'rb') as f:
                self.scalers['transformer'] = pickle.load(f)
            print(f"✅ Loaded Transformer scaler")
        
        # Load LSTM model
        lstm_path = os.path.join(model_dir, 'lstm_model.pth')
        if os.path.exists(lstm_path):
            input_dim = len(self.feature_cols)
            model = LSTMForecaster(input_dim=input_dim, hidden_dim=128, num_layers=2)
            model.load_state_dict(torch.load(lstm_path, map_location=self.device))
            model = model.to(self.device)
            model.eval()
            self.models['lstm'] = model
            print(f"✅ Loaded LSTM model from {lstm_path}")
        
        # Load LSTM scaler
        lstm_scaler_path = os.path.join(model_dir, 'lstm_scaler.pkl')
        if os.path.exists(lstm_scaler_path):
            with open(lstm_scaler_path, 'rb') as f:
                self.scalers['lstm'] = pickle.load(f)
            print(f"✅ Loaded LSTM scaler")
        
        return True

    def detect_anomalies_and_trends(self):
        """Detect anomalies and trends using all methods with EEPD risk classification"""
        # Try to load models if not already loaded
        if 'transformer' not in self.models:
            self.load_models()
        
        print("\n" + "="*60)
        print(f"EEPD: Detecting Anomalies and Trends for {self.disease_name}")
        print("="*60)

        results = {
            'dates': [],
            'trends': [],
            'anomalies': [],
            'alert_triggered': [],
            'alert_messages': [],
            'risk_levels': [],
            'risk_scores': [],
            'icu_utilization': [],
            'ventilator_utilization': []
        }

        # Use transformer model for anomaly detection
        if 'transformer' in self.models:
            model = self.models['transformer']
            scaler = self.scalers['transformer']
            model.eval()

            data_scaled = scaler.transform(self.df[self.feature_cols].values)
            anomaly_scores = []

            with torch.no_grad():
                for i in range(len(data_scaled) - self.window_size):
                    window = data_scaled[i:i+self.window_size]
                    window_tensor = torch.FloatTensor(window).unsqueeze(0).to(self.device)

                    output = model(window_tensor, phase='test')
                    reconstruction_error = torch.mean((output - window_tensor) ** 2).item()
                    anomaly_scores.append(reconstruction_error)

            # Determine anomaly threshold (95th percentile)
            threshold = np.percentile(anomaly_scores, 95)
            anomalies = [score > threshold for score in anomaly_scores]
            
            # Normalize anomaly scores to 0-1 range
            max_score = max(anomaly_scores) if anomaly_scores else 1
            normalized_scores = [s / max_score if max_score > 0 else 0 for s in anomaly_scores]

            # Pad beginning
            anomalies = [False] * self.window_size + anomalies
            anomaly_scores = [0] * self.window_size + anomaly_scores
            normalized_scores = [0] * self.window_size + normalized_scores
        else:
            anomalies = [False] * len(self.df)
            anomaly_scores = [0] * len(self.df)
            normalized_scores = [0] * len(self.df)

        # Detect trends
        trends_list = []
        growth_rates = []
        for i in range(len(self.df)):
            if i < self.window_size:
                trends_list.append('insufficient_data')
                growth_rates.append(0)
            else:
                values = self.df['New_Cases'].values[:i+1]
                trend, confidence, growth_rate = self.eepd_alert_system.detect_trend(values, window=7)
                trends_list.append(trend)
                growth_rates.append(growth_rate)

        # Check alerts with EEPD risk classification
        for i in range(len(self.df)):
            results['dates'].append(self.df['Date'].iloc[i])
            results['trends'].append(trends_list[i])
            results['anomalies'].append(anomalies[i])
            
            # Get ICU and ventilator utilization
            icu_util = self.df['ICU_Admission_Rate'].iloc[i] if 'ICU_Admission_Rate' in self.df.columns else 0
            vent_util = self.df['Ventilator_Utilization'].iloc[i] if 'Ventilator_Utilization' in self.df.columns else 0
            results['icu_utilization'].append(icu_util)
            results['ventilator_utilization'].append(vent_util)

            if i >= self.window_size:
                recent_trends = trends_list[max(0, i-7):i+1]
                metrics = {
                    'growth_rate': growth_rates[i],
                    'icu_utilization': icu_util,
                    'ventilator_utilization': vent_util,
                    'anomaly_score': normalized_scores[i]
                }
                alert, risk_assessment = self.eepd_alert_system.check_alert(
                    recent_trends, metrics, self.disease_name
                )
                results['alert_triggered'].append(alert)
                results['alert_messages'].append(risk_assessment.get('message', ''))
                results['risk_levels'].append(risk_assessment.get('risk_level', 'Normal'))
                results['risk_scores'].append(risk_assessment.get('risk_score', 0))
            else:
                results['alert_triggered'].append(False)
                results['alert_messages'].append('')
                results['risk_levels'].append('Normal')
                results['risk_scores'].append(0)

        self.results = results
        return results

    def predict_waves(self):
        """Predict upcoming waves using LSTM and wave detection"""
        print("\n" + "="*60)
        print(f"Predicting Waves for {self.disease_name}")
        print("="*60)
        
        if 'New_Cases' not in self.df.columns:
            print("New_Cases column not found. Cannot predict waves.")
            return None
        
        cases_data = self.df['New_Cases'].values
        
        # Detect past waves
        past_waves = self.wave_detector.detect_waves(cases_data)
        print(f"Detected {len(past_waves)} past waves")
        
        # Generate forecast for wave prediction
        forecast = self.generate_forecast(days_ahead=60)
        forecast_cases = None
        if forecast is not None and 'New_Cases' in forecast.columns:
            forecast_cases = forecast['New_Cases'].values
        
        # Predict next wave
        wave_prediction = self.wave_detector.predict_next_wave(
            cases_data, forecast_cases
        )
        
        self.wave_predictions = {
            'past_waves': past_waves,
            'next_wave': wave_prediction,
            'forecast': forecast
        }
        
        return self.wave_predictions

    def monitor_icu_ventilator_load(self, forecast_days=30):
        """Monitor and predict ICU/ventilator load"""
        print("\n" + "="*60)
        print(f"Monitoring ICU/Ventilator Load for {self.disease_name}")
        print("="*60)
        
        icu_col = 'ICU_Admission_Rate' if 'ICU_Admission_Rate' in self.df.columns else None
        vent_col = 'Ventilator_Utilization' if 'Ventilator_Utilization' in self.df.columns else None
        
        if not icu_col and not vent_col:
            print("ICU/Ventilator columns not found.")
            return None
        
        monitoring = {}
        
        # Current utilization
        if icu_col:
            current_icu = self.df[icu_col].iloc[-1]
            avg_icu = self.df[icu_col].mean()
            max_icu = self.df[icu_col].max()
            monitoring['icu'] = {
                'current': current_icu,
                'average': avg_icu,
                'max': max_icu,
                'status': 'Critical' if current_icu > 0.85 else 'High' if current_icu > 0.7 else 'Moderate' if current_icu > 0.5 else 'Normal'
            }
        
        if vent_col:
            current_vent = self.df[vent_col].iloc[-1]
            avg_vent = self.df[vent_col].mean()
            max_vent = self.df[vent_col].max()
            monitoring['ventilator'] = {
                'current': current_vent,
                'average': avg_vent,
                'max': max_vent,
                'status': 'Critical' if current_vent > 0.75 else 'High' if current_vent > 0.6 else 'Moderate' if current_vent > 0.4 else 'Normal'
            }
        
        # Predict future load using LSTM if available
        if 'lstm' in self.models:
            forecast = self.generate_forecast(days_ahead=forecast_days)
            if forecast is not None:
                if icu_col and 'ICU_Admission_Rate' in forecast.columns:
                    monitoring['icu']['forecast'] = forecast['ICU_Admission_Rate'].values
                    monitoring['icu']['forecast_peak'] = forecast['ICU_Admission_Rate'].max()
                    monitoring['icu']['forecast_peak_day'] = forecast['ICU_Admission_Rate'].idxmax()
                
                if vent_col and 'Ventilator_Utilization' in forecast.columns:
                    monitoring['ventilator']['forecast'] = forecast['Ventilator_Utilization'].values
                    monitoring['ventilator']['forecast_peak'] = forecast['Ventilator_Utilization'].max()
                    monitoring['ventilator']['forecast_peak_day'] = forecast['Ventilator_Utilization'].idxmax()
        
        self.icu_monitoring = monitoring
        return monitoring

    def generate_forecast(self, days_ahead=30):
        """Generate forecast for next N days"""
        print(f"\nGenerating {days_ahead}-day forecast...")

        if 'lstm' not in self.models:
            print("LSTM model not trained. Please train first.")
            return None

        model = self.models['lstm']
        scaler = self.scalers['lstm']
        model.eval()

        # Prepare last window
        data_scaled = scaler.transform(self.df[self.feature_cols].values)
        current_window = data_scaled[-self.window_size:]

        forecasts = []

        with torch.no_grad():
            for _ in range(days_ahead):
                window_tensor = torch.FloatTensor(current_window).unsqueeze(0).to(self.device)
                prediction = model(window_tensor)
                pred_value = prediction[0, -1, :].cpu().numpy()

                forecasts.append(pred_value)

                # Update window
                current_window = np.vstack([current_window[1:], pred_value])

        # Inverse transform
        forecasts = scaler.inverse_transform(np.array(forecasts))

        # Create forecast dataframe
        last_date = self.df['Date'].max()
        forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days_ahead)

        forecast_df = pd.DataFrame(forecasts, columns=self.feature_cols)
        forecast_df['Date'] = forecast_dates
        
        # Store forecast for evaluation
        self.last_forecast = forecast_df

        return forecast_df

    def plot_results(self, save_path=None):
        """Plot comprehensive results"""
        fig, axes = plt.subplots(4, 2, figsize=(20, 16))
        fig.suptitle('COVID-19 Trend Detection and Early Warning System', fontsize=16, fontweight='bold')

        # Plot 1: New Cases with trends
        ax = axes[0, 0]
        ax.plot(self.df['Date'], self.df['New_Cases'], 'b-', linewidth=2, label='New Cases')

        if 'trends' in self.results:
            increasing_idx = [i for i, t in enumerate(self.results['trends']) if t == 'increasing']
            if increasing_idx:
                ax.scatter(self.df['Date'].iloc[increasing_idx],
                          self.df['New_Cases'].iloc[increasing_idx],
                          color='red', s=100, marker='^', label='Upward Trend', zorder=5)

        ax.set_title('New Cases with Trend Detection', fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('New Cases')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Anomalies
        ax = axes[0, 1]
        ax.plot(self.df['Date'], self.df['New_Cases'], 'b-', linewidth=2, alpha=0.6, label='New Cases')

        if 'anomalies' in self.results:
            anomaly_idx = [i for i, a in enumerate(self.results['anomalies']) if a]
            if anomaly_idx:
                ax.scatter(self.df['Date'].iloc[anomaly_idx],
                          self.df['New_Cases'].iloc[anomaly_idx],
                          color='red', s=150, marker='X', label='Anomaly Detected', zorder=5)

        ax.set_title('Anomaly Detection (Transformer-based)', fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('New Cases')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: Test Positivity Rate
        ax = axes[1, 0]
        ax.plot(self.df['Date'], self.df['Test_Positivity_Rate'], 'g-', linewidth=2)
        ax.set_title('Test Positivity Rate', fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Rate')
        ax.grid(True, alpha=0.3)

        # Plot 4: Hospitalization Rate
        ax = axes[1, 1]
        ax.plot(self.df['Date'], self.df['Hospitalization_Rate'], 'orange', linewidth=2)
        ax.set_title('Hospitalization Rate', fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Rate')
        ax.grid(True, alpha=0.3)

        # Plot 5: ICU Admission Rate
        ax = axes[2, 0]
        ax.plot(self.df['Date'], self.df['ICU_Admission_Rate'], 'purple', linewidth=2)
        ax.set_title('ICU Admission Rate', fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Rate')
        ax.grid(True, alpha=0.3)

        # Plot 6: Mortality Rate
        ax = axes[2, 1]
        ax.plot(self.df['Date'], self.df['Mortality_Rate'], 'darkred', linewidth=2)
        ax.set_title('Mortality Rate', fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Rate')
        ax.grid(True, alpha=0.3)

        # Plot 7: Alert Timeline
        ax = axes[3, 0]
        if 'alert_triggered' in self.results:
            alert_idx = [i for i, a in enumerate(self.results['alert_triggered']) if a]
            if alert_idx:
                alert_dates = [self.results['dates'][i] for i in alert_idx]
                alert_values = [1] * len(alert_dates)
                ax.scatter(alert_dates, alert_values, color='red', s=200, marker='*',
                          label='Alert Triggered', zorder=5)
                ax.set_ylim(0.5, 1.5)

        ax.set_title('Alert Timeline', fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Alert Status')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 8: Multi-metric Overview
        ax = axes[3, 1]
        metrics_to_plot = ['Test_Positivity_Rate', 'Hospitalization_Rate', 'ICU_Admission_Rate']
        for metric in metrics_to_plot:
            normalized = (self.df[metric] - self.df[metric].min()) / (self.df[metric].max() - self.df[metric].min() + 1e-8)
            ax.plot(self.df['Date'], normalized, linewidth=2, label=metric, alpha=0.7)

        ax.set_title('Normalized Multi-Metric Overview', fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Normalized Value')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")

        plt.show()

    def generate_alert_report(self):
        """Generate comprehensive EEPD alert report with risk levels"""
        print("\n" + "="*60)
        print(f"EEPD ALERT REPORT - {self.disease_name}")
        print("="*60)

        if 'alert_triggered' not in self.results:
            print("No results available. Please run detect_anomalies_and_trends() first.")
            return

        alert_count = sum(self.results['alert_triggered'])
        anomaly_count = sum(self.results['anomalies'])

        # Current risk level
        current_risk = 'Normal'
        if 'risk_levels' in self.results and len(self.results['risk_levels']) > 0:
            current_risk = self.results['risk_levels'][-1]
            current_risk_score = self.results['risk_scores'][-1] if 'risk_scores' in self.results else 0
        else:
            current_risk_score = 0

        print(f"\n{'='*60}")
        print(f"CURRENT RISK LEVEL: {current_risk}")
        print(f"Risk Score: {current_risk_score}/100")
        print(f"{'='*60}")

        print(f"\nTotal Alerts Triggered: {alert_count}")
        print(f"Total Anomalies Detected: {anomaly_count}")

        # Risk level distribution
        if 'risk_levels' in self.results:
            risk_dist = {}
            for risk in self.results['risk_levels']:
                risk_dist[risk] = risk_dist.get(risk, 0) + 1
            print(f"\nRisk Level Distribution:")
            for risk, count in risk_dist.items():
                print(f"  {risk}: {count} days")

        # Show alert details with risk levels
        if alert_count > 0:
            print("\nAlert Details (by Risk Level):")
            print("-" * 60)
            
            # Group by risk level
            critical_alerts = []
            high_alerts = []
            elevated_alerts = []
            
            for i, (date, alert, message, risk_level) in enumerate(zip(
                self.results['dates'],
                self.results['alert_triggered'],
                self.results['alert_messages'],
                self.results.get('risk_levels', ['Normal'] * len(self.results['dates']))
            )):
                if alert:
                    if 'Critical' in risk_level:
                        critical_alerts.append((date, message))
                    elif 'High Alert' in risk_level:
                        high_alerts.append((date, message))
                    elif 'Elevated' in risk_level:
                        elevated_alerts.append((date, message))
            
            if critical_alerts:
                print("\n🚨 CRITICAL / OUTBREAK LIKELY:")
                for date, message in critical_alerts[-5:]:  # Show last 5
                    print(f"  {date.strftime('%Y-%m-%d')}: {message}")
            
            if high_alerts:
                print("\n⚠️ HIGH ALERT:")
                for date, message in high_alerts[-5:]:
                    print(f"  {date.strftime('%Y-%m-%d')}: {message}")
            
            if elevated_alerts:
                print("\n⚡ ELEVATED RISK:")
                for date, message in elevated_alerts[-5:]:
                    print(f"  {date.strftime('%Y-%m-%d')}: {message}")

        # Current trend
        if len(self.results['trends']) > 0:
            current_trend = self.results['trends'][-1]
            print(f"\nCurrent Trend: {current_trend.upper()}")

        # ICU/Ventilator status
        if 'icu_utilization' in self.results and len(self.results['icu_utilization']) > 0:
            current_icu = self.results['icu_utilization'][-1]
            print(f"\nCurrent ICU Utilization: {current_icu*100:.1f}%")
            if current_icu > 0.85:
                print("  ⚠️ CRITICAL: ICU capacity at critical levels")
            elif current_icu > 0.7:
                print("  ⚡ WARNING: ICU capacity high")
        
        if 'ventilator_utilization' in self.results and len(self.results['ventilator_utilization']) > 0:
            current_vent = self.results['ventilator_utilization'][-1]
            print(f"Current Ventilator Utilization: {current_vent*100:.1f}%")
            if current_vent > 0.75:
                print("  ⚠️ CRITICAL: Ventilator capacity at critical levels")
            elif current_vent > 0.6:
                print("  ⚡ WARNING: Ventilator capacity high")

        # Recent anomalies
        recent_anomalies = []
        for i in range(max(0, len(self.results['anomalies']) - 7), len(self.results['anomalies'])):
            if self.results['anomalies'][i]:
                recent_anomalies.append(self.results['dates'][i])

        if recent_anomalies:
            print(f"\nRecent Anomalies (last 7 days): {len(recent_anomalies)}")
            for date in recent_anomalies:
                print(f"  - {date.strftime('%Y-%m-%d')}")

        print("\n" + "="*60)

    def get_detailed_metrics_matrix(self):
        """
        Generate a comprehensive detailed metrics matrix with all variables
        Returns a pandas DataFrame with all available metrics and variables
        """
        if 'alert_triggered' not in self.results:
            print("No results available. Please run detect_anomalies_and_trends() first.")
            return None
        
        latest_idx = len(self.results['dates']) - 1
        metrics_data = []
        
        # Basic epidemiological metrics
        if 'New_Cases' in self.df.columns:
            new_cases = self.df['New_Cases'].values
            metrics_data.append({
                'Variable': 'New_Cases',
                'Current_Value': new_cases[latest_idx],
                'Mean': np.mean(new_cases),
                'Max': np.max(new_cases),
                'Min': np.min(new_cases),
                'Std': np.std(new_cases),
                'Trend': self.results['trends'][latest_idx] if latest_idx < len(self.results['trends']) else 'N/A'
            })
        
        if 'Test_Positivity_Rate' in self.df.columns:
            tpr = self.df['Test_Positivity_Rate'].values
            metrics_data.append({
                'Variable': 'Test_Positivity_Rate',
                'Current_Value': tpr[latest_idx],
                'Mean': np.mean(tpr),
                'Max': np.max(tpr),
                'Min': np.min(tpr),
                'Std': np.std(tpr),
                'Trend': 'N/A'
            })
        
        if 'Case_Growth_Rate' in self.df.columns:
            cgr = self.df['Case_Growth_Rate'].values
            metrics_data.append({
                'Variable': 'Case_Growth_Rate',
                'Current_Value': cgr[latest_idx],
                'Mean': np.mean(cgr),
                'Max': np.max(cgr),
                'Min': np.min(cgr),
                'Std': np.std(cgr),
                'Trend': 'N/A'
            })
        
        if 'Hospitalization_Rate' in self.df.columns:
            hosp = self.df['Hospitalization_Rate'].values
            metrics_data.append({
                'Variable': 'Hospitalization_Rate',
                'Current_Value': hosp[latest_idx],
                'Mean': np.mean(hosp),
                'Max': np.max(hosp),
                'Min': np.min(hosp),
                'Std': np.std(hosp),
                'Trend': 'N/A'
            })
        
        if 'ICU_Admission_Rate' in self.df.columns:
            icu_adm = self.df['ICU_Admission_Rate'].values
            metrics_data.append({
                'Variable': 'ICU_Admission_Rate',
                'Current_Value': icu_adm[latest_idx],
                'Mean': np.mean(icu_adm),
                'Max': np.max(icu_adm),
                'Min': np.min(icu_adm),
                'Std': np.std(icu_adm),
                'Trend': 'N/A'
            })
        
        if 'Ventilator_Utilization' in self.df.columns:
            vent = self.df['Ventilator_Utilization'].values
            metrics_data.append({
                'Variable': 'Ventilator_Utilization',
                'Current_Value': vent[latest_idx],
                'Mean': np.mean(vent),
                'Max': np.max(vent),
                'Min': np.min(vent),
                'Std': np.std(vent),
                'Trend': 'N/A'
            })
        
        if 'Mortality_Rate' in self.df.columns:
            mort = self.df['Mortality_Rate'].values
            metrics_data.append({
                'Variable': 'Mortality_Rate',
                'Current_Value': mort[latest_idx],
                'Mean': np.mean(mort),
                'Max': np.max(mort),
                'Min': np.min(mort),
                'Std': np.std(mort),
                'Trend': 'N/A'
            })
        
        # EEPD-specific variables
        if 'risk_levels' in self.results:
            risk_levels = self.results['risk_levels']
            metrics_data.append({
                'Variable': 'Risk_Level',
                'Current_Value': risk_levels[latest_idx],
                'Mean': 'N/A',
                'Max': 'N/A',
                'Min': 'N/A',
                'Std': 'N/A',
                'Trend': 'N/A'
            })
        
        if 'risk_scores' in self.results:
            risk_scores = self.results['risk_scores']
            metrics_data.append({
                'Variable': 'Risk_Score',
                'Current_Value': risk_scores[latest_idx],
                'Mean': np.mean(risk_scores),
                'Max': np.max(risk_scores),
                'Min': np.min(risk_scores),
                'Std': np.std(risk_scores),
                'Trend': 'N/A'
            })
        
        if 'icu_utilization' in self.results:
            icu_util = self.results['icu_utilization']
            metrics_data.append({
                'Variable': 'ICU_Utilization_EEPD',
                'Current_Value': icu_util[latest_idx],
                'Mean': np.mean(icu_util),
                'Max': np.max(icu_util),
                'Min': np.min(icu_util),
                'Std': np.std(icu_util),
                'Trend': 'N/A'
            })
        
        if 'ventilator_utilization' in self.results:
            vent_util = self.results['ventilator_utilization']
            metrics_data.append({
                'Variable': 'Ventilator_Utilization_EEPD',
                'Current_Value': vent_util[latest_idx],
                'Mean': np.mean(vent_util),
                'Max': np.max(vent_util),
                'Min': np.min(vent_util),
                'Std': np.std(vent_util),
                'Trend': 'N/A'
            })
        
        if 'anomalies' in self.results:
            anomalies = self.results['anomalies']
            anomaly_count = sum(anomalies)
            metrics_data.append({
                'Variable': 'Anomaly_Detected',
                'Current_Value': 1 if anomalies[latest_idx] else 0,
                'Mean': anomaly_count / len(anomalies),
                'Max': 1,
                'Min': 0,
                'Std': np.std([1 if a else 0 for a in anomalies]),
                'Trend': 'N/A'
            })
        
        if 'alert_triggered' in self.results:
            alerts = self.results['alert_triggered']
            alert_count = sum(alerts)
            metrics_data.append({
                'Variable': 'Alert_Triggered',
                'Current_Value': 1 if alerts[latest_idx] else 0,
                'Mean': alert_count / len(alerts),
                'Max': 1,
                'Min': 0,
                'Std': np.std([1 if a else 0 for a in alerts]),
                'Trend': 'N/A'
            })
        
        # Add trend information
        if 'trends' in self.results:
            trends = self.results['trends']
            increasing_count = sum(1 for t in trends if t == 'increasing')
            metrics_data.append({
                'Variable': 'Trend_Status',
                'Current_Value': trends[latest_idx],
                'Mean': 'N/A',
                'Max': 'N/A',
                'Min': 'N/A',
                'Std': 'N/A',
                'Trend': f"{increasing_count}/{len(trends)} increasing"
            })
        
        # Create DataFrame
        metrics_df = pd.DataFrame(metrics_data)
        return metrics_df

    def run_full_analysis(self, train_models=True, epochs=50):
        """Run complete EEPD analysis pipeline"""
        print("\n" + "="*80)
        print(f" EARLY EPIDEMIC & PANDEMIC DETECTION SYSTEM (EEPD) ".center(80, "="))
        print(f" Disease: {self.disease_name} ".center(80, "="))
        print("="*80)

        if train_models:
            # Train models
            self.train_transformer_model(epochs=epochs)
            self.train_lstm_model(epochs=epochs)

        # Detect anomalies and trends with EEPD risk classification
        self.detect_anomalies_and_trends()

        # Generate forecast
        forecast = self.generate_forecast(days_ahead=30)

        # Predict waves
        wave_predictions = self.predict_waves()
        if wave_predictions:
            next_wave = wave_predictions.get('next_wave', {})
            if next_wave.get('wave_likely', False):
                print(f"\n⚠️ WAVE PREDICTION: New wave likely with {next_wave.get('confidence', 0)*100:.1f}% confidence")
                if next_wave.get('estimated_peak_days'):
                    print(f"   Estimated peak in {next_wave['estimated_peak_days']} days")

        # Monitor ICU/Ventilator load
        icu_monitoring = self.monitor_icu_ventilator_load(forecast_days=30)
        if icu_monitoring:
            if 'icu' in icu_monitoring:
                icu_status = icu_monitoring['icu']['status']
                print(f"\n🏥 ICU Status: {icu_status} (Utilization: {icu_monitoring['icu']['current']*100:.1f}%)")
            if 'ventilator' in icu_monitoring:
                vent_status = icu_monitoring['ventilator']['status']
                print(f"💨 Ventilator Status: {vent_status} (Utilization: {icu_monitoring['ventilator']['current']*100:.1f}%)")

        # Generate comprehensive report
        self.generate_alert_report()

        # Plot results
        self.plot_results()

        return {
            'forecast': forecast,
            'wave_predictions': wave_predictions,
            'icu_monitoring': icu_monitoring,
            'results': self.results
        }


# =====================================================
# 8. MAIN EXECUTION
# =====================================================

def main():
    """Main execution function"""

    # Example usage with your data
    # Replace 'your_data.csv' with your actual file path

    # Option 1: Load from file
    # system = COVIDTrendAnalysisSystem(
    #     data_path='your_data.csv',
    #     window_size=14,
    #     sensitivity='medium'
    # )

    # Option 2: Create sample data (for demonstration)
    print("Creating sample COVID-19 dataset...")
    dates = pd.date_range(start='2020-02-14', periods=1166, freq='D')

    # Generate realistic-looking data
    np.random.seed(42)
    t = np.arange(1166)

    # Simulate waves with some randomness
    new_cases = (
        1000 * np.sin(t / 50) ** 2 +
        500 * np.sin(t / 30) +
        np.random.normal(0, 100, 1166)
    ).clip(min=0)

    df = pd.DataFrame({
        'Date': dates,
        'Test_Positivity_Rate': (0.05 + 0.1 * np.sin(t / 40) + np.random.normal(0, 0.02, 1166)).clip(0, 1),
        'New_Cases': new_cases,
        'Case_Growth_Rate': np.gradient(new_cases) / (new_cases + 1),
        'Hospitalization_Rate': (0.1 + 0.05 * np.sin(t / 45) + np.random.normal(0, 0.01, 1166)).clip(0, 1),
        'ICU_Admission_Rate': (0.02 + 0.03 * np.sin(t / 50) + np.random.normal(0, 0.005, 1166)).clip(0, 1),
        'Ventilator_Utilization': (0.3 + 0.2 * np.sin(t / 55) + np.random.normal(0, 0.05, 1166)).clip(0, 1),
        'Mortality_Rate': (0.01 + 0.01 * np.sin(t / 60) + np.random.normal(0, 0.002, 1166)).clip(0, 1)
    })

    # Initialize system
    system = COVIDTrendAnalysisSystem(
        df=df,
        window_size=14,
        sensitivity='medium'
    )

    # Run full analysis
    forecast = system.run_full_analysis(train_models=True, epochs=30)

    # Display forecast
    if forecast is not None:
        print("\n30-Day Forecast:")
        print(forecast.head(10))

    print("\n" + "="*80)
    print(" Analysis Complete! ".center(80, "="))
    print("="*80)


if __name__ == "__main__":
    main()