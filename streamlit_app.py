"""
Early Epidemic & Pandemic Detection System (EEPD) - Streamlit App
Multi-disease outbreak detection with ML predictions and automated alerting

Usage:
    streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import io

# Import from our main system
from covid_trend_detection import COVIDTrendAnalysisSystem, EEPDSystem, TrendDetector, EEPDAlertSystem
from scipy import stats

# Set page config
st.set_page_config(
    page_title="EEPD - Early Epidemic & Pandemic Detection",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .alert-box {
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        font-size: 1.2rem;
        font-weight: bold;
        text-align: center;
    }
    .alert-critical {
        background-color: #ffebee;
        border: 4px solid #d32f2f;
        color: #b71c1c;
        font-size: 1.3rem;
    }
    .alert-high {
        background-color: #fff3e0;
        border: 3px solid #f57c00;
        color: #e65100;
        font-size: 1.2rem;
    }
    .alert-elevated {
        background-color: #fff9c4;
        border: 3px solid #fbc02d;
        color: #f57f17;
        font-size: 1.1rem;
    }
    .alert-normal {
        background-color: #e8f5e9;
        border: 3px solid #4caf50;
        color: #2e7d32;
        font-size: 1.1rem;
    }
    .alert-danger {
        background-color: #ffebee;
        border: 3px solid #f44336;
        color: #c62828;
    }
    .alert-warning {
        background-color: #fff3e0;
        border: 3px solid #ff9800;
        color: #e65100;
    }
    .alert-safe {
        background-color: #e8f5e9;
        border: 3px solid #4caf50;
        color: #2e7d32;
    }
    .metric-card {
        background-color: #f5f5f5;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)


def analyze_trend_simple(df, sensitivity='medium'):
    """Simple trend analysis for one month of data"""

    # Initialize trend detector
    detector = TrendDetector(sensitivity=sensitivity)

    results = {
        'overall_trend': 'stable',
        'alert_level': 'safe',
        'confidence': 0.0,
        'metrics': {},
        'recommendations': []
    }

    # Analyze each metric
    metrics_to_check = [
        'New_Cases',
        'Test_Positivity_Rate',
        'Hospitalization_Rate',
        'ICU_Admission_Rate',
        'Mortality_Rate'
    ]

    increasing_count = 0
    alarming_metrics = []

    for metric in metrics_to_check:
        if metric in df.columns:
            values = df[metric].values

            # Check if there's actual data (not all zeros)
            if np.sum(values) > 0:
                trend, confidence = detector.detect_trend(values, window=7)

                # Calculate growth rate
                if len(values) >= 7:
                    recent_avg = np.mean(values[-7:])
                    older_avg = np.mean(values[:7])
                    growth_rate = (recent_avg - older_avg) / (older_avg + 1e-8) * 100
                else:
                    growth_rate = 0

                results['metrics'][metric] = {
                    'trend': trend,
                    'confidence': confidence,
                    'growth_rate': growth_rate,
                    'current_value': values[-1],
                    'max_value': np.max(values),
                    'mean_value': np.mean(values)
                }

                if trend == 'increasing':
                    increasing_count += 1
                    if growth_rate > 10:  # More than 10% increase
                        alarming_metrics.append((metric, growth_rate))

    # Determine overall alert level
    if increasing_count >= 3 or len(alarming_metrics) >= 2:
        results['alert_level'] = 'danger'
        results['overall_trend'] = 'increasing'
        results['confidence'] = 0.8
    elif increasing_count >= 2 or len(alarming_metrics) >= 1:
        results['alert_level'] = 'warning'
        results['overall_trend'] = 'increasing'
        results['confidence'] = 0.6
    else:
        results['alert_level'] = 'safe'
        results['overall_trend'] = 'stable' if increasing_count == 0 else 'mixed'
        results['confidence'] = 0.7

    # Generate recommendations
    if results['alert_level'] == 'danger':
        results['recommendations'] = [
            "⚠️ URGENT: Multiple metrics showing strong upward trends",
            "📊 Immediate review of epidemic control measures recommended",
            "🏥 Prepare healthcare capacity for potential surge",
            "📢 Consider public health communication about preventive measures"
        ]
    elif results['alert_level'] == 'warning':
        results['recommendations'] = [
            "⚡ CAUTION: Some metrics showing upward trends",
            "📊 Enhanced monitoring recommended for next 7-14 days",
            "🏥 Review current healthcare capacity and preparedness",
            "📢 Consider early preventive messaging to public"
        ]
    else:
        results['recommendations'] = [
            "✅ STABLE: Trends appear stable or decreasing",
            "📊 Continue routine monitoring",
            "🏥 Maintain current measures and protocols",
            "📢 Regular updates to stakeholders"
        ]

    results['alarming_metrics'] = alarming_metrics
    results['increasing_count'] = increasing_count

    return results


def create_trend_plot(df, results):
    """Create visualization of trends"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('COVID-19 Metrics Trend Analysis', fontsize=16, fontweight='bold')

    metrics_to_plot = [
        ('New_Cases', 'New Cases', 'blue'),
        ('Test_Positivity_Rate', 'Test Positivity Rate', 'orange'),
        ('Hospitalization_Rate', 'Hospitalization Rate', 'red'),
        ('ICU_Admission_Rate', 'ICU Admission Rate', 'purple')
    ]

    for idx, (metric, title, color) in enumerate(metrics_to_plot):
        ax = axes[idx // 2, idx % 2]

        if metric in df.columns:
            # Plot the data
            ax.plot(df['Date'], df[metric], color=color, linewidth=2, marker='o', markersize=4)

            # Add trend indicator
            if metric in results['metrics']:
                trend = results['metrics'][metric]['trend']
                growth_rate = results['metrics'][metric]['growth_rate']

                # Add trend arrow
                y_pos = ax.get_ylim()[1] * 0.9
                if trend == 'increasing':
                    ax.text(0.5, 0.95, f'↗ Increasing ({growth_rate:.1f}%)',
                           transform=ax.transAxes, ha='center', va='top',
                           bbox=dict(boxstyle='round', facecolor='red', alpha=0.3),
                           fontsize=10, fontweight='bold')
                elif trend == 'decreasing':
                    ax.text(0.5, 0.95, f'↘ Decreasing ({growth_rate:.1f}%)',
                           transform=ax.transAxes, ha='center', va='top',
                           bbox=dict(boxstyle='round', facecolor='green', alpha=0.3),
                           fontsize=10, fontweight='bold')
                else:
                    ax.text(0.5, 0.95, f'→ Stable ({growth_rate:.1f}%)',
                           transform=ax.transAxes, ha='center', va='top',
                           bbox=dict(boxstyle='round', facecolor='gray', alpha=0.3),
                           fontsize=10, fontweight='bold')

            ax.set_title(title, fontweight='bold', fontsize=12)
            ax.set_xlabel('Date', fontsize=10)
            ax.set_ylabel('Value', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
        else:
            ax.text(0.5, 0.5, f'{title}\nData Not Available',
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12, color='gray')
            ax.axis('off')

    plt.tight_layout()
    return fig


def get_risk_level_display(risk_level):
    """Get display styling for risk level"""
    risk_styles = {
        'Critical / Outbreak Likely': {
            'class': 'alert-critical',
            'icon': '🚨',
            'color': '#d32f2f'
        },
        'High Alert': {
            'class': 'alert-high',
            'icon': '⚠️',
            'color': '#f57c00'
        },
        'Elevated Risk': {
            'class': 'alert-elevated',
            'icon': '⚡',
            'color': '#fbc02d'
        },
        'Normal': {
            'class': 'alert-normal',
            'icon': '✅',
            'color': '#4caf50'
        }
    }
    return risk_styles.get(risk_level, risk_styles['Normal'])


def main():
    # Header
    st.markdown('<div class="main-header">🚨 Early Epidemic & Pandemic Detection System (EEPD)</div>', unsafe_allow_html=True)
    st.markdown("### Multi-disease outbreak detection with ML predictions and automated alerting")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")

        disease_name = st.text_input(
            "Disease Name",
            value="COVID-19",
            help="Enter the disease name (e.g., COVID-19, Disease X, Disease Y)"
        )

        sensitivity = st.select_slider(
            "Alert Sensitivity",
            options=['low', 'medium', 'high'],
            value='medium',
            help="Low: Only strong signals | Medium: Balanced | High: Early detection"
        )
        
        enable_wave_prediction = st.checkbox(
            "Enable Wave Prediction",
            value=True,
            help="Use LSTM to predict upcoming epidemic waves"
        )
        
        enable_icu_monitoring = st.checkbox(
            "Enable ICU/Ventilator Monitoring",
            value=True,
            help="Monitor and predict ICU/ventilator capacity"
        )

        st.markdown("---")
        st.markdown("### 📋 Required Data Format")
        st.markdown("""
        Your CSV should have these columns:
        - **Date** (DD/MM/YYYY)
        - **Test_Positivity_Rate**
        - **New_Cases**
        - **Case_Growth_Rate**
        - **Hospitalization_Rate**
        - **ICU_Admission_Rate**
        - **Ventilator_Utilization**
        - **Mortality_Rate**

        **Minimum:** 7 days of data
        **Recommended:** 30+ days for better predictions
        """)
        
        st.markdown("### 🎯 System Features")
        st.markdown("""
        - ✅ Multi-disease outbreak detection
        - 📈 Wave prediction using LSTM
        - 🏥 ICU/Ventilator load monitoring
        - 🚨 4-tier risk classification
        - 📊 Real-time dashboard visualization
        - 📧 Automated alerting and reporting
        """)

        st.markdown("---")
        st.markdown("### 📥 Download Sample")

        # Create sample data
        sample_dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        sample_df = pd.DataFrame({
            'Date': sample_dates.strftime('%d/%m/%Y'),
            'Test_Positivity_Rate': np.random.uniform(0.03, 0.08, 30),
            'New_Cases': np.random.randint(50, 200, 30),
            'Case_Growth_Rate': np.random.uniform(-0.1, 0.1, 30),
            'Hospitalization_Rate': np.random.uniform(0.1, 0.2, 30),
            'ICU_Admission_Rate': np.random.uniform(0.02, 0.05, 30),
            'Ventilator_Utilization': np.random.uniform(0.2, 0.4, 30),
            'Mortality_Rate': np.random.uniform(0.005, 0.015, 30)
        })

        csv_buffer = io.StringIO()
        sample_df.to_csv(csv_buffer, index=False)

        st.download_button(
            label="📄 Download Sample CSV",
            data=csv_buffer.getvalue(),
            file_name="sample_covid_data.csv",
            mime="text/csv"
        )

    # Main content
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📤 Upload Your Data")
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload your disease data file"
        )

    with col2:
        st.subheader("ℹ️ Quick Info")
        st.info(f"""
        **EEPD System** analyzes disease data 
        to detect early signs of outbreaks and predict future trends.
        """)

    if uploaded_file is not None:
        # Initialize session state for storing results
        if 'system' not in st.session_state:
            st.session_state.system = None
        if 'full_analysis_done' not in st.session_state:
            st.session_state.full_analysis_done = False
        if 'forecast_data' not in st.session_state:
            st.session_state.forecast_data = None
        if 'wave_predictions' not in st.session_state:
            st.session_state.wave_predictions = None
        if 'icu_monitoring' not in st.session_state:
            st.session_state.icu_monitoring = None
        try:
            # Load data
            df = pd.read_csv(uploaded_file)

            # Convert date
            df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
            df = df.sort_values('Date').reset_index(drop=True)

            st.success(f"✅ Data loaded successfully! {len(df)} days of data")

            # Show data preview
            with st.expander("📊 Preview Data"):
                st.dataframe(df.head(10))

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Days", len(df))
                with col2:
                    st.metric("Start Date", df['Date'].min().strftime('%Y-%m-%d'))
                with col3:
                    st.metric("End Date", df['Date'].max().strftime('%Y-%m-%d'))

            # Action Buttons Section
            st.markdown("---")
            st.subheader("🎯 Analysis Actions")
            
            col_btn1, col_btn2, col_btn3 = st.columns(3)
            
            with col_btn1:
                run_quick_analysis = st.button(
                    "⚡ Quick Analysis",
                    help="Fast analysis without model training",
                    use_container_width=True
                )
            
            with col_btn2:
                run_full_analysis = st.button(
                    "🔬 Full Analysis",
                    help="Complete EEPD analysis with model training (takes 5-15 minutes)",
                    use_container_width=True,
                    type="primary"
                )
            
            with col_btn3:
                predict_next_month = st.button(
                    "📅 Predict Next Month",
                    help="Generate 30-day forecast predictions",
                    use_container_width=True
                )
            
            # Initialize system (store in session state)
            # Reinitialize if data has changed or system is None
            if st.session_state.system is None or 'last_uploaded_file' not in st.session_state or st.session_state.last_uploaded_file != uploaded_file.name:
                st.session_state.system = COVIDTrendAnalysisSystem(
                    df=df,
                    window_size=14,
                    sensitivity=sensitivity,
                    disease_name=disease_name
                )
                st.session_state.last_uploaded_file = uploaded_file.name
                st.session_state.full_analysis_done = False
                st.session_state.forecast_data = None
                st.session_state.wave_predictions = None
                st.session_state.icu_monitoring = None
            
            system = st.session_state.system
            
            # Quick Analysis
            if run_quick_analysis:
                st.markdown("---")
                st.subheader("⚡ Quick Analysis Results")
                
                with st.spinner("Running quick analysis..."):
                    results = analyze_trend_simple(df, sensitivity=sensitivity)

                try:
                    system.detect_anomalies_and_trends()
                    if 'risk_levels' in system.results and len(system.results['risk_levels']) > 0:
                        current_risk_level = system.results['risk_levels'][-1]
                        current_risk_score = system.results['risk_scores'][-1] if 'risk_scores' in system.results else 0
                    else:
                        current_risk_level = 'Normal'
                        current_risk_score = 0
                except:
                    current_risk_level = results.get('alert_level', 'Normal').title()
                    current_risk_score = 0
                
                # Display risk level
                risk_level_map = {
                    'danger': 'Critical / Outbreak Likely',
                    'warning': 'High Alert',
                    'safe': 'Normal'
                }
                
                if current_risk_level in ['Critical / Outbreak Likely', 'High Alert', 'Elevated Risk', 'Normal']:
                    display_risk = current_risk_level
                else:
                    display_risk = risk_level_map.get(results.get("alert_level", "safe"), "Normal")
                
                risk_display = get_risk_level_display(display_risk)
                
                st.markdown(f"""
                <div class="alert-box {risk_display['class']}">
                    {risk_display['icon']} {display_risk}
                    <br>
                    <span style="font-size: 0.9rem;">Risk Score: {current_risk_score}/100</span>
                </div>
                """, unsafe_allow_html=True)

                st.success("✅ Quick analysis complete!")
            
            # Full Analysis
            if run_full_analysis:
                st.markdown("---")
                st.subheader("🔬 Full EEPD Analysis")
                
                epochs_input = st.number_input(
                    "Training Epochs",
                    min_value=20,
                    max_value=100,
                    value=50,
                    help="More epochs = better accuracy but slower (20-50 recommended)"
                )
                
                if st.button("🚀 Start Full Analysis", type="primary"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    try:
                        # Ensure system is initialized with current data
                        if st.session_state.system is None:
                            st.session_state.system = COVIDTrendAnalysisSystem(
                                df=df,
                                window_size=14,
                                sensitivity=sensitivity,
                                disease_name=disease_name
                            )
                            system = st.session_state.system
                        
                        # Verify data is prepared
                        if not hasattr(system, 'feature_cols') or system.feature_cols is None:
                            system.prepare_data()
                        
                        # Check if pre-trained models exist
                        import os
                        models_exist = os.path.exists('saved_models') and \
                                      os.path.exists('saved_models/transformer_model.pth') and \
                                      os.path.exists('saved_models/lstm_model.pth')
                        
                        if models_exist:
                            status_text.text("Step 1/2: Loading pre-trained models...")
                            progress_bar.progress(30)
                            if system.load_models():
                                st.info("✅ Using pre-trained models from saved_models/")
                            else:
                                st.warning("⚠️ Failed to load pre-trained models. Training new models...")
                                models_exist = False
                        
                        if not models_exist:
                            status_text.text("Step 1/5: Training Transformer model...")
                            progress_bar.progress(20)
                            system.train_transformer_model(epochs=epochs_input, batch_size=32)
                            
                            status_text.text("Step 2/5: Training LSTM model...")
                            progress_bar.progress(40)
                            system.train_lstm_model(epochs=epochs_input, batch_size=32)
                        else:
                            progress_bar.progress(50)
                        
                        if not models_exist:
                            status_text.text("Step 3/5: Detecting anomalies and trends...")
                            progress_bar.progress(60)
                        else:
                            status_text.text("Step 2/5: Detecting anomalies and trends...")
                            progress_bar.progress(60)
                        system.detect_anomalies_and_trends()
                        
                        if not models_exist:
                            status_text.text("Step 4/5: Generating forecast...")
                            progress_bar.progress(80)
                        else:
                            status_text.text("Step 3/5: Generating forecast...")
                            progress_bar.progress(75)
                        forecast = system.generate_forecast(days_ahead=30)
                        st.session_state.forecast_data = forecast
                        
                        if not models_exist:
                            status_text.text("Step 5/5: Predicting waves and monitoring ICU...")
                            progress_bar.progress(90)
                        else:
                            status_text.text("Step 4/5: Predicting waves and monitoring ICU...")
                            progress_bar.progress(85)
                        wave_predictions = system.predict_waves()
                        st.session_state.wave_predictions = wave_predictions
                        
                        icu_monitoring = system.monitor_icu_ventilator_load(forecast_days=30)
                        st.session_state.icu_monitoring = icu_monitoring
                        
                        progress_bar.progress(100)
                        status_text.text("✅ Full analysis complete!")
                        st.session_state.full_analysis_done = True
                        st.session_state.system = system
                        
                        st.success("🎉 Full analysis completed successfully!")
                        st.balloons()
                        
                    except Exception as e:
                        st.error(f"❌ Error during analysis: {str(e)}")
                        import traceback
                        with st.expander("🔍 View Full Error Details"):
                            st.code(traceback.format_exc())
                        st.warning("💡 Tip: Make sure your data has all required columns: Date, New_Cases, Test_Positivity_Rate, Case_Growth_Rate, Hospitalization_Rate, ICU_Admission_Rate, Ventilator_Utilization, Mortality_Rate")
            
            # Predict Next Month
            if predict_next_month:
                st.markdown("---")
                st.subheader("📅 Next Month Predictions")
                
                if 'lstm' not in system.models:
                    st.warning("⚠️ LSTM model not trained. Please run Full Analysis first or train the model.")
                    if st.button("🔧 Train LSTM Model Now"):
                        with st.spinner("Training LSTM model (this may take a few minutes)..."):
                            system.train_lstm_model(epochs=50, batch_size=32)
                            st.session_state.system = system
                            st.success("✅ Model trained! Click 'Predict Next Month' again.")
                else:
                    with st.spinner("Generating next month predictions..."):
                        forecast = system.generate_forecast(days_ahead=30)
                        st.session_state.forecast_data = forecast
                        st.session_state.system = system
                    
                    if st.session_state.forecast_data is not None:
                        st.success("✅ Predictions generated!")
            
            # Display results if available
            if st.session_state.full_analysis_done or st.session_state.forecast_data is not None:
                # Show forecast data
                if st.session_state.forecast_data is not None:
                    st.markdown("---")
                    st.subheader("📊 Predicted Data (Next 30 Days)")
                    
                    forecast = st.session_state.forecast_data
                    
                    # Format Date column for display (similar to input format)
                    forecast_display = forecast.copy()
                    forecast_display['Date'] = forecast_display['Date'].dt.strftime('%d/%m/%Y')
                    
                    # Create a formatted version for display (keep original for download)
                    forecast_formatted = forecast_display.copy()
                    
                    # Summary metrics - show all key metrics
                    st.markdown("#### 📈 Prediction Summary")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Prediction Start", forecast['Date'].min().strftime('%Y-%m-%d'))
                    with col2:
                        st.metric("Prediction End", forecast['Date'].max().strftime('%Y-%m-%d'))
                    with col3:
                        st.metric("Total Days", len(forecast))
                    with col4:
                        st.metric("Avg New Cases", f"{forecast['New_Cases'].mean():.0f}")
                    
                    # Show statistics for all metrics
                    st.markdown("#### 📊 All Metrics Statistics")
                    metrics_stats = []
                    for col in forecast.columns:
                        if col != 'Date':
                            metrics_stats.append({
                                'Metric': col.replace('_', ' '),
                                'Min': f"{forecast[col].min():.4f}",
                                'Max': f"{forecast[col].max():.4f}",
                                'Mean': f"{forecast[col].mean():.4f}",
                                'Std': f"{forecast[col].std():.4f}"
                            })
                    stats_df = pd.DataFrame(metrics_stats)
                    st.dataframe(stats_df, use_container_width=True, hide_index=True)
                    
                    # Display full forecast table with all columns
                    st.markdown("#### 📋 Complete Predicted Data (All Columns)")
                    st.info(f"📅 Showing predictions for {len(forecast)} days from {forecast['Date'].min().strftime('%Y-%m-%d')} to {forecast['Date'].max().strftime('%Y-%m-%d')}")
                    
                    # Reorder columns to show Date first
                    display_cols = ['Date'] + [col for col in forecast_formatted.columns if col != 'Date']
                    forecast_formatted = forecast_formatted[display_cols]
                    
                    # Format numeric columns for better readability (only for display)
                    for col in forecast_formatted.columns:
                        if col != 'Date':
                            if 'Rate' in col or 'Utilization' in col:
                                forecast_formatted[col] = forecast_formatted[col].apply(lambda x: f"{float(x):.4f}")
                            elif 'Cases' in col:
                                forecast_formatted[col] = forecast_formatted[col].apply(lambda x: f"{float(x):.0f}")
                            else:
                                forecast_formatted[col] = forecast_formatted[col].apply(lambda x: f"{float(x):.4f}")
                    
                    st.dataframe(forecast_formatted, use_container_width=True, hide_index=True, height=600)
                    
                    # Show raw numeric data in expander
                    with st.expander("🔢 View Raw Numeric Data (for calculations)"):
                        st.dataframe(forecast, use_container_width=True, hide_index=True)
                    
                    # Forecast visualization
                    st.markdown("#### 📈 Forecast Visualization")
                    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                    
                    # Plot 1: New Cases
                    ax = axes[0, 0]
                    ax.plot(df['Date'], df['New_Cases'], 'b-', label='Historical', linewidth=2)
                    ax.plot(forecast['Date'], forecast['New_Cases'], 'r--', label='Forecast', linewidth=2)
                    ax.axvline(x=df['Date'].max(), color='gray', linestyle=':', label='Today')
                    ax.set_title('New Cases Forecast', fontweight='bold')
                    ax.set_xlabel('Date')
                    ax.set_ylabel('New Cases')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    # Plot 2: Test Positivity Rate
                    ax = axes[0, 1]
                    if 'Test_Positivity_Rate' in df.columns and 'Test_Positivity_Rate' in forecast.columns:
                        ax.plot(df['Date'], df['Test_Positivity_Rate'], 'b-', label='Historical', linewidth=2)
                        ax.plot(forecast['Date'], forecast['Test_Positivity_Rate'], 'r--', label='Forecast', linewidth=2)
                        ax.axvline(x=df['Date'].max(), color='gray', linestyle=':')
                        ax.set_title('Test Positivity Rate Forecast', fontweight='bold')
                        ax.set_xlabel('Date')
                        ax.set_ylabel('Rate')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                    
                    # Plot 3: ICU Admission Rate
                    ax = axes[1, 0]
                    if 'ICU_Admission_Rate' in df.columns and 'ICU_Admission_Rate' in forecast.columns:
                        ax.plot(df['Date'], df['ICU_Admission_Rate'], 'b-', label='Historical', linewidth=2)
                        ax.plot(forecast['Date'], forecast['ICU_Admission_Rate'], 'r--', label='Forecast', linewidth=2)
                        ax.axvline(x=df['Date'].max(), color='gray', linestyle=':')
                        ax.set_title('ICU Admission Rate Forecast', fontweight='bold')
                        ax.set_xlabel('Date')
                        ax.set_ylabel('Rate')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                    
                    # Plot 4: Ventilator Utilization
                    ax = axes[1, 1]
                    if 'Ventilator_Utilization' in df.columns and 'Ventilator_Utilization' in forecast.columns:
                        ax.plot(df['Date'], df['Ventilator_Utilization'], 'b-', label='Historical', linewidth=2)
                        ax.plot(forecast['Date'], forecast['Ventilator_Utilization'], 'r--', label='Forecast', linewidth=2)
                        ax.axvline(x=df['Date'].max(), color='gray', linestyle=':')
                        ax.set_title('Ventilator Utilization Forecast', fontweight='bold')
                        ax.set_xlabel('Date')
                        ax.set_ylabel('Utilization')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Download forecast buttons
                    st.markdown("#### 💾 Download Predicted Data")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Download in display format (with formatted dates)
                        csv_buffer_display = io.StringIO()
                        forecast_display.to_csv(csv_buffer_display, index=False)
                        st.download_button(
                            label="📥 Download Formatted Data (CSV)",
                            data=csv_buffer_display.getvalue(),
                            file_name=f"next_month_forecast_formatted_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            help="Download with formatted dates (DD/MM/YYYY) matching your input format"
                        )
                    
                    with col2:
                        # Download raw numeric data
                        csv_buffer_raw = io.StringIO()
                        forecast.to_csv(csv_buffer_raw, index=False)
                        st.download_button(
                            label="📥 Download Raw Data (CSV)",
                            data=csv_buffer_raw.getvalue(),
                            file_name=f"next_month_forecast_raw_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            help="Download with full precision numeric values"
                        )
            
            # Display wave predictions and ICU monitoring if full analysis was done
            if st.session_state.full_analysis_done:
                # Wave Prediction Section
                if enable_wave_prediction and st.session_state.wave_predictions:
                    st.markdown("---")
                    st.markdown("### 📈 Wave Prediction Results")
                    
                    wave_pred = st.session_state.wave_predictions
                    if wave_pred.get('next_wave'):
                        next_wave = wave_pred['next_wave']
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            wave_likely = next_wave.get('wave_likely', False)
                            st.metric(
                                "Wave Likely",
                                "Yes" if wave_likely else "No",
                                delta=f"{next_wave.get('confidence', 0)*100:.1f}% confidence" if wave_likely else None
                            )
                        
                        with col2:
                            est_peak = next_wave.get('estimated_peak_days')
                            st.metric(
                                "Estimated Peak",
                                f"{est_peak} days" if est_peak else "N/A",
                                delta=f"Growth: {next_wave.get('growth_rate', 0)*100:.1f}%" if next_wave.get('growth_rate') else None
                            )
                        
                        with col3:
                            st.metric(
                                "Past Waves Detected",
                                len(wave_pred.get('past_waves', [])),
                                delta="Historical patterns"
                            )
                        
                        if wave_likely:
                            st.warning(f"⚠️ New wave predicted with {next_wave.get('confidence', 0)*100:.1f}% confidence. Monitor closely.")
                
                # ICU/Ventilator Monitoring Section
                if enable_icu_monitoring and st.session_state.icu_monitoring:
                    st.markdown("---")
                    st.markdown("### 🏥 ICU & Ventilator Load Monitoring")
                    
                    icu_mon = st.session_state.icu_monitoring
                    if icu_mon:
                        col1, col2 = st.columns(2)
                        
                        if 'icu' in icu_mon:
                            with col1:
                                icu_data = icu_mon['icu']
                                st.markdown("#### ICU Capacity")
                                st.metric(
                                    "Current Utilization",
                                    f"{icu_data['current']*100:.1f}%",
                                    delta=f"Status: {icu_data['status']}"
                                )
                                st.progress(icu_data['current'])
                                
                                col1a, col1b = st.columns(2)
                                with col1a:
                                    st.metric("Average", f"{icu_data['average']*100:.1f}%")
                                with col1b:
                                    st.metric("Peak", f"{icu_data['max']*100:.1f}%")
                                
                                if icu_data['current'] > 0.85:
                                    st.error("🚨 CRITICAL: ICU capacity at critical levels!")
                                elif icu_data['current'] > 0.7:
                                    st.warning("⚠️ WARNING: ICU capacity high")
                        
                        if 'ventilator' in icu_mon:
                            with col2:
                                vent_data = icu_mon['ventilator']
                                st.markdown("#### Ventilator Capacity")
                                st.metric(
                                    "Current Utilization",
                                    f"{vent_data['current']*100:.1f}%",
                                    delta=f"Status: {vent_data['status']}"
                                )
                                st.progress(vent_data['current'])
                                
                                col2a, col2b = st.columns(2)
                                with col2a:
                                    st.metric("Average", f"{vent_data['average']*100:.1f}%")
                                with col2b:
                                    st.metric("Peak", f"{vent_data['max']*100:.1f}%")
                                
                                if vent_data['current'] > 0.75:
                                    st.error("🚨 CRITICAL: Ventilator capacity at critical levels!")
                                elif vent_data['current'] > 0.6:
                                    st.warning("⚠️ WARNING: Ventilator capacity high")
            
            # Quick EEPD Overview (if not running full analysis)
            if not st.session_state.full_analysis_done and not run_quick_analysis and not run_full_analysis and not predict_next_month:
                st.info("👆 Use the buttons above to run analysis or predictions")

            # Summary metrics (only if quick analysis was run)
            if run_quick_analysis:
                # Make sure results is defined
                # if 'results' not in locals():
                #     results = analyze_trend_simple(df, sensitivity=sensitivity)
                
                st.markdown("---")
            st.markdown("### 📊 Summary")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Overall Trend",
                    results['overall_trend'].upper(),
                    delta="Alert" if results['alert_level'] == 'danger' else None
                )

            with col2:
                st.metric(
                    "Metrics Increasing",
                    f"{results['increasing_count']}/{len(results['metrics'])}"
                )

            with col3:
                st.metric(
                    "Alert Level",
                    results['alert_level'].upper()
                )

            # Detailed metrics
            if len(results.get('alarming_metrics', [])) > 0:
                st.markdown("### ⚠️ Alarming Metrics")
                for metric, growth_rate in results['alarming_metrics']:
                    st.warning(f"**{metric}**: {growth_rate:.1f}% increase detected")

                # Detailed Metrics Matrix
                st.markdown("### 📈 Detailed Metrics Matrix")
                
                # Create comprehensive metrics dataframe
                try:
                    # Get latest values from system results if available
                    latest_idx = len(system.results.get('dates', [])) - 1
                    
                    detailed_metrics_data = []
                        
                    # Add all basic metrics
                    for metric, data in results["metrics"].items():
                        detailed_metrics_data.append({
                            'Metric': metric.replace('_', ' '),
                            'Current Value': f"{data['current_value']:.4f}",
                            'Trend': data['trend'].title(),
                            'Growth Rate (%)': f"{data['growth_rate']:.2f}",
                            'Mean Value': f"{data['mean_value']:.4f}",
                            'Max Value': f"{data['max_value']:.4f}",
                            'Confidence': f"{data['confidence']:.2f}"
                        })
                    
                    # Add EEPD-specific variables if available
                    if latest_idx >= 0 and 'risk_levels' in system.results:
                        detailed_metrics_data.append({
                            'Metric': 'Risk Level',
                            'Current Value': system.results['risk_levels'][latest_idx],
                            'Trend': 'N/A',
                            'Growth Rate (%)': 'N/A',
                            'Mean Value': 'N/A',
                            'Max Value': 'N/A',
                            'Confidence': f"{system.results.get('risk_scores', [0])[latest_idx]}/100"
                        })
                    
                    if latest_idx >= 0 and 'risk_scores' in system.results:
                        detailed_metrics_data.append({
                            'Metric': 'Risk Score',
                            'Current Value': f"{system.results['risk_scores'][latest_idx]:.1f}",
                            'Trend': 'N/A',
                            'Growth Rate (%)': 'N/A',
                            'Mean Value': f"{np.mean(system.results['risk_scores']):.1f}",
                            'Max Value': f"{np.max(system.results['risk_scores']):.1f}",
                            'Confidence': 'N/A'
                        })
                    
                    # Add ICU and Ventilator utilization
                    if latest_idx >= 0:
                            if 'icu_utilization' in system.results and len(system.results['icu_utilization']) > 0:
                                icu_vals = system.results['icu_utilization']
                                detailed_metrics_data.append({
                                    'Metric': 'ICU Utilization',
                                    'Current Value': f"{icu_vals[latest_idx]*100:.2f}%",
                                    'Trend': 'N/A',
                                    'Growth Rate (%)': 'N/A',
                                    'Mean Value': f"{np.mean(icu_vals)*100:.2f}%",
                                    'Max Value': f"{np.max(icu_vals)*100:.2f}%",
                                    'Confidence': 'N/A'
                                })
                            
                            if 'ventilator_utilization' in system.results and len(system.results['ventilator_utilization']) > 0:
                                vent_vals = system.results['ventilator_utilization']
                                detailed_metrics_data.append({
                                    'Metric': 'Ventilator Utilization',
                                    'Current Value': f"{vent_vals[latest_idx]*100:.2f}%",
                                    'Trend': 'N/A',
                                    'Growth Rate (%)': 'N/A',
                                    'Mean Value': f"{np.mean(vent_vals)*100:.2f}%",
                                    'Max Value': f"{np.max(vent_vals)*100:.2f}%",
                                    'Confidence': 'N/A'
                                })
                            
                            # Add anomaly information
                            if 'anomalies' in system.results:
                                anomalies = system.results['anomalies']
                                anomaly_count = sum(anomalies)
                                detailed_metrics_data.append({
                                    'Metric': 'Anomalies Detected',
                                    'Current Value': 'Yes' if anomalies[latest_idx] else 'No',
                                    'Trend': 'N/A',
                                    'Growth Rate (%)': 'N/A',
                                    'Mean Value': f"{anomaly_count} total",
                                    'Max Value': 'N/A',
                                    'Confidence': f"{anomaly_count/len(anomalies)*100:.1f}%"
                                })
                            
                    # Add Case Growth Rate if available
                    if "Case_Growth_Rate" in df.columns:
                            case_growth = df['Case_Growth_Rate'].values
                            detailed_metrics_data.append({
                                'Metric': 'Case Growth Rate',
                                'Current Value': f"{case_growth[-1]:.4f}",
                                'Trend': 'N/A',
                                'Growth Rate (%)': 'N/A',
                                'Mean Value': f"{np.mean(case_growth):.4f}",
                                'Max Value': f"{np.max(case_growth):.4f}",
                                'Confidence': 'N/A'
                            })
                    
                    # Add Ventilator Utilization from dataframe if not already added
                    if 'Ventilator_Utilization' in df.columns and 'Ventilator_Utilization' not in [m['Metric'] for m in detailed_metrics_data]:
                        vent_util = df['Ventilator_Utilization'].values
                        detailed_metrics_data.append({
                            'Metric': 'Ventilator Utilization (Raw)',
                            'Current Value': f"{vent_util[-1]*100:.2f}%",
                            'Trend': 'N/A',
                            'Growth Rate (%)': 'N/A',
                            'Mean Value': f"{np.mean(vent_util)*100:.2f}%",
                            'Max Value': f"{np.max(vent_util)*100:.2f}%",
                            'Confidence': 'N/A'
                        })
                            
                        # Display as table
                        if detailed_metrics_data:
                            metrics_df = pd.DataFrame(detailed_metrics_data)
                        # Convert all columns to string to avoid PyArrow type conflicts
                        for col in metrics_df.columns:
                            metrics_df[col] = metrics_df[col].astype(str)
                        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
                        
                        # Also show in expandable detailed view
                        with st.expander("📊 View Detailed Metrics with All Variables"):
                            st.markdown("#### Complete Metrics Matrix")
                            st.dataframe(metrics_df, use_container_width=True)
                            
                            # Add additional computed metrics
                            st.markdown("#### Additional Computed Metrics")
                            additional_metrics = {}
                            
                            if latest_idx >= 0:
                                if 'trends' in system.results:
                                    trends = system.results['trends']
                                    increasing_days = sum(1 for t in trends if t == 'increasing')
                                    additional_metrics['Days with Increasing Trend'] = f"{increasing_days}/{len(trends)}"
                                
                                if 'alert_triggered' in system.results:
                                    alerts = system.results['alert_triggered']
                                    additional_metrics['Total Alerts Triggered'] = sum(alerts)
                                    additional_metrics['Alert Rate'] = f"{sum(alerts)/len(alerts)*100:.1f}%"
                            
                            if additional_metrics:
                                additional_df = pd.DataFrame([additional_metrics]).T
                                additional_df.columns = ['Value']
                                # Convert to string to avoid PyArrow type conflicts
                                additional_df['Value'] = additional_df['Value'].astype(str)
                                st.dataframe(additional_df, use_container_width=True)
                            else:
                                # Fallback to card view if no detailed data
                                cols = st.columns(3)
                                for idx, (metric, data) in enumerate(results["metrics"].items()):
                                    with cols[idx % 3]:
                                        trend_icon = "↗️" if data['trend'] == 'increasing' else "↘️" if data['trend'] == 'decreasing' else "➡️"
                                        trend_color = "red" if data['trend'] == 'increasing' else "green" if data['trend'] == 'decreasing' else "gray"

                                        st.markdown(f"""
                                        <div class="metric-card">
                                            <strong>{metric.replace('_', ' ')}</strong><br>
                                            <span style="color: {trend_color}; font-size: 1.2rem;">{trend_icon} {data['trend'].title()}</span><br>
                                            <small>Growth: {data['growth_rate']:.1f}%</small><br>
                                            <small>Current: {data['current_value']:.4f}</small>
                                        </div>
                                        """, unsafe_allow_html=True)
                except Exception as e:
                        st.warning(f"Could not generate detailed matrix: {str(e)}. Showing basic metrics.")
                        # Fallback to card view
                        if 'results' in locals() and 'metrics' in results:
                            cols = st.columns(3)
                            for idx, (metric, data) in enumerate(results['metrics'].items()):
                                with cols[idx % 3]:
                                    trend_icon = "↗️" if data['trend'] == 'increasing' else "↘️" if data['trend'] == 'decreasing' else "➡️"
                                    trend_color = "red" if data['trend'] == 'increasing' else "green" if data['trend'] == 'decreasing' else "gray"

                                    st.markdown(f"""
                                    <div class="metric-card">
                                        <strong>{metric.replace('_', ' ')}</strong><br>
                                        <span style="color: {trend_color}; font-size: 1.2rem;">{trend_icon} {data['trend'].title()}</span><br>
                                        <small>Growth: {data['growth_rate']:.1f}%</small><br>
                                        <small>Current: {data['current_value']:.4f}</small>
                                    </div>
                                    """, unsafe_allow_html=True)

                                # Visualizations (only if quick analysis was run)
            st.markdown("### 📊 Trend Visualizations")

            fig = create_trend_plot(df, results)
            st.pyplot(fig)

            # Evaluation Matrix
            st.markdown("---")
            st.markdown("### 📊 Model Evaluation Matrix")
            
            try:
                if st.session_state.system is not None:
                    eval_matrix = st.session_state.system.get_evaluation_matrix()
                    if eval_matrix is not None and len(eval_matrix) > 0:
                        # Convert all values to string to avoid PyArrow type conflicts
                        eval_matrix['Value'] = eval_matrix['Value'].astype(str)
                        eval_matrix['Status'] = eval_matrix['Status'].astype(str)
                        st.dataframe(eval_matrix, use_container_width=True, hide_index=True)
                        
                        # Download button for evaluation matrix
                        csv_buffer = io.StringIO()
                        eval_matrix.to_csv(csv_buffer, index=False)
                        st.download_button(
                            label="📥 Download Evaluation Matrix (CSV)",
                            data=csv_buffer.getvalue(),
                            file_name=f"evaluation_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    else:
                        st.info("⚠️ Evaluation matrix not available. Please run Full Analysis first to train models.")
                else:
                    st.info("⚠️ System not initialized. Please upload a file and run analysis first.")
            except Exception as e:
                st.warning(f"⚠️ Could not generate evaluation matrix: {str(e)}")

            # Recommendations
            st.markdown("---")
            st.markdown("### 💡 Recommendations")

            for recommendation in results['recommendations']:
                if '⚠️' in recommendation or '⚡' in recommendation:
                    st.warning(recommendation)
                elif '✅' in recommendation:
                    st.success(recommendation)
                else:
                    st.info(recommendation)

            # Export results
            st.markdown("---")
            st.subheader("�� Export Results")

            col1, col2 = st.columns(2)

            with col1:
                # Create summary report
                report = f"""
COVID-19 TREND ANALYSIS REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

ALERT STATUS: {results['alert_level'].upper()}
Overall Trend: {results['overall_trend'].upper()}
Confidence: {results['confidence']*100:.0f}%

METRICS SUMMARY:
"""
                for metric, data in results['metrics'].items():
                    report += f"\n{metric}:"
                    report += f"\n  - Trend: {data['trend'].upper()}"
                    report += f"\n  - Growth Rate: {data['growth_rate']:.2f}%"
                    report += f"\n  - Current Value: {data['current_value']:.4f}"

                report += "\n\nRECOMMENDATIONS:\n"
                for rec in results['recommendations']:
                    report += f"- {rec}\n"

                st.download_button(
                    label="📄 Download Report (TXT)",
                    data=report,
                    file_name=f"covid_alert_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )

            with col2:
                                            # Export detailed data with all variables
                    try:
                        latest_idx = len(system.results.get('dates', [])) - 1
                        
                        export_data = []
                        
                        # Add all basic metrics
                        for metric, data in results['metrics'].items():
                            export_data.append({
                        'Metric': metric,
                                'Current_Value': data['current_value'],
                        'Trend': data['trend'],
                        'Growth_Rate_%': data['growth_rate'],
                                'Mean_Value': data['mean_value'],
                        'Max_Value': data['max_value'],
                                'Confidence': data['confidence']
                            })
                        
                        # Add EEPD-specific variables
                        if latest_idx >= 0:
                            if 'risk_levels' in system.results:
                                export_data.append({
                                    'Metric': 'Risk_Level',
                                    'Current_Value': system.results['risk_levels'][latest_idx],
                                    'Trend': 'N/A',
                                    'Growth_Rate_%': 'N/A',
                                    'Mean_Value': 'N/A',
                                    'Max_Value': 'N/A',
                                    'Confidence': 'N/A'
                                })
                            
                            if 'risk_scores' in system.results:
                                export_data.append({
                                    'Metric': 'Risk_Score',
                                'Current_Value': system.results['risk_scores'][latest_idx],
                                'Trend': 'N/A',
                                'Growth_Rate_%': 'N/A',
                                'Mean_Value': np.mean(system.results['risk_scores']),
                                'Max_Value': np.max(system.results['risk_scores']),
                                'Confidence': 'N/A'
                            })
                        
                        if 'icu_utilization' in system.results and len(system.results['icu_utilization']) > 0:
                            icu_vals = system.results['icu_utilization']
                            export_data.append({
                                'Metric': 'ICU_Utilization',
                                'Current_Value': icu_vals[latest_idx],
                                'Trend': 'N/A',
                                'Growth_Rate_%': 'N/A',
                                'Mean_Value': np.mean(icu_vals),
                                'Max_Value': np.max(icu_vals),
                                'Confidence': 'N/A'
                            })
                        
                        if 'ventilator_utilization' in system.results and len(system.results['ventilator_utilization']) > 0:
                            vent_vals = system.results['ventilator_utilization']
                            export_data.append({
                                'Metric': 'Ventilator_Utilization',
                                'Current_Value': vent_vals[latest_idx],
                                'Trend': 'N/A',
                                'Growth_Rate_%': 'N/A',
                                'Mean_Value': np.mean(vent_vals),
                                'Max_Value': np.max(vent_vals),
                                'Confidence': 'N/A'
                            })
                        
                        if 'anomalies' in system.results:
                            anomalies = system.results['anomalies']
                            export_data.append({
                                'Metric': 'Anomaly_Detected',
                                'Current_Value': 1 if anomalies[latest_idx] else 0,
                                'Trend': 'N/A',
                                'Growth_Rate_%': 'N/A',
                                'Mean_Value': sum(anomalies) / len(anomalies),
                                'Max_Value': 1,
                                'Confidence': 'N/A'
                            })
                        
                        # Add Case Growth Rate if available
                        if 'Case_Growth_Rate' in df.columns:
                            case_growth = df['Case_Growth_Rate'].values
                            export_data.append({
                                'Metric': 'Case_Growth_Rate',
                                'Current_Value': case_growth[-1],
                                'Trend': 'N/A',
                                'Growth_Rate_%': 'N/A',
                                'Mean_Value': np.mean(case_growth),
                                'Max_Value': np.max(case_growth),
                                'Confidence': 'N/A'
                            })
                        
                        results_df = pd.DataFrame(export_data)
                    
                    except Exception as e:
                        # Fallback to basic export
                        if 'results' in locals() and 'metrics' in results:
                            results_df = pd.DataFrame([
                                {
                                    "Metric": metric,
                                    "Trend": data["trend"],
                                    "Growth_Rate_%": data["growth_rate"],
                                    "Current_Value": data["current_value"],
                                    "Max_Value": data["max_value"],
                                    "Mean_Value": data["mean_value"]
                                }
                                for metric, data in results["metrics"].items()
                            ])
                        else:
                            results_df = pd.DataFrame()

                        csv_buffer = io.StringIO()
                        results_df.to_csv(csv_buffer, index=False)

                        st.download_button(
                            label="📊 Download Detailed Metrics (CSV)",
                    data=csv_buffer.getvalue(),
                            file_name=f"eepd_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                        
                        

        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.info("Please make sure your CSV has the correct format and column names.")

    else:
        # Show example when no file uploaded
        st.info("👆 Please upload a CSV file to begin analysis")

        st.markdown("### 📝 Example Use Case")
        st.markdown("""
        1. **Collect your data**: Gather COVID-19 metrics for the past 7-30 days
        2. **Format as CSV**: Use the required column format (see sidebar)
        3. **Upload**: Use the file uploader above
        4. **Get Results**: Instant trend analysis and alerts
        5. **Take Action**: Follow the recommendations provided
        """)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: gray; font-size: 0.9rem;">
        <p><strong>Early Epidemic & Pandemic Detection System (EEPD)</strong></p>
        <p>Multi-disease outbreak detection | LSTM-based predictions | Automated alerting</p>
        <p>Supports: COVID-19, Disease X, Disease Y, and other infectious diseases</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()