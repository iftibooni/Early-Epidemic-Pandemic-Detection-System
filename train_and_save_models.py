"""
Train models on boo.csv and save them for later use
This script trains both Transformer and LSTM models and saves them to saved_models/ directory
"""

import pandas as pd
import os
from covid_trend_detection import COVIDTrendAnalysisSystem

# Configuration
DATA_FILE = 'boo.csv'  # Your training data file
MODEL_DIR = 'saved_models'  # Directory to save models
EPOCHS = 50  # Number of training epochs
BATCH_SIZE = 32
WINDOW_SIZE = 14
SENSITIVITY = 'medium'
DISEASE_NAME = 'COVID-19'  # or 'Disease X', 'Disease Y', etc.

def main():
    print("="*80)
    print(" TRAIN AND SAVE MODELS ".center(80, "="))
    print("="*80)
    
    # Check if data file exists
    if not os.path.exists(DATA_FILE):
        print(f"❌ Error: Data file '{DATA_FILE}' not found!")
        print(f"   Please make sure '{DATA_FILE}' exists in the current directory.")
        return
    
    print(f"\n📊 Loading training data from: {DATA_FILE}")
    
    try:
        # Load data
        df = pd.read_csv(DATA_FILE)
        
        # Convert date if needed
        if 'Date' in df.columns:
            try:
                df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
            except:
                df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date').reset_index(drop=True)
        
        print(f"✅ Data loaded successfully!")
        print(f"   Shape: {df.shape}")
        print(f"   Date range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
        
        # Initialize system
        print(f"\n🤖 Initializing EEPD System for {DISEASE_NAME}...")
        system = COVIDTrendAnalysisSystem(
            df=df,
            window_size=WINDOW_SIZE,
            sensitivity=SENSITIVITY,
            disease_name=DISEASE_NAME
        )
        
        # Train Transformer model
        print(f"\n🔧 Training Transformer model ({EPOCHS} epochs)...")
        print("   This may take a few minutes...")
        system.train_transformer_model(epochs=EPOCHS, batch_size=BATCH_SIZE)
        print("   ✅ Transformer model trained")
        
        # Train LSTM model
        print(f"\n🔧 Training LSTM model ({EPOCHS} epochs)...")
        print("   This may take a few minutes...")
        system.train_lstm_model(epochs=EPOCHS, batch_size=BATCH_SIZE)
        print("   ✅ LSTM model trained")
        
        # Save models
        print(f"\n💾 Saving models to {MODEL_DIR}/...")
        system.save_models(model_dir=MODEL_DIR)
        
        print("\n" + "="*80)
        print(" ✅ TRAINING COMPLETE! ".center(80, "="))
        print("="*80)
        print(f"\n📁 Models saved to: {MODEL_DIR}/")
        print("   - transformer_model.pth")
        print("   - transformer_scaler.pkl")
        print("   - lstm_model.pth")
        print("   - lstm_scaler.pkl")
        print("   - metadata.pkl")
        print("\n💡 You can now use these pre-trained models for predictions!")
        print("   The system will automatically load them when you run analysis.")
        
    except Exception as e:
        print(f"\n❌ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

