import pandas as pd
import numpy as np
import joblib
import json
import logging
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_score
from sklearn.ensemble import RandomForestClassifier

# Try importing XGBoost
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_regime_model(data_path: str = "ml_training/dataset.csv",
                      model_output: str = "ml_training/regime_model.joblib",
                      features_output: str = "ml_training/features.json"):
    
    if not os.path.exists(data_path):
        logger.error(f"Dataset not found at {data_path}. Run prepare_data.py first.")
        return

    logger.info(f"Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)
    if 'date' in df.columns:
        df.set_index('date', inplace=True)
        
    # Drop non-feature columns
    exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'target', 'date', 'Unnamed: 0']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    X = df[feature_cols]
    y = df['target']
    
    logger.info(f"Features: {len(feature_cols)}")
    logger.info(f"Samples: {len(df)}")
    
    # Split Data (Time-series split is better, but random split is okay for simple regime demo if we respect time order)
    # For regime detection, shuffling is controversial. Strict time split is safer to avoid leaks.
    # We will use simple non-shuffled split: first 80% train, last 20% test.
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    logger.info(f"Training set: {len(X_train)}, Test set: {len(X_test)}")
    
    # Initialize Model
    if HAS_XGB:
        logger.info("Using XGBoost Classifier")
        model = XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='logloss',
            use_label_encoder=False
        )
    else:
        logger.info("XGBoost not found. Using Random Forest Classifier")
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=5,
            random_state=42
        )
        
    # Train
    logger.info("Training model...")
    model.fit(X_train, y_train)
    
    # Evaluate
    logger.info("Evaluating model...")
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    precision = precision_score(y_test, y_pred)
    logger.info(f"Precision (Win Rate of Signal): {precision:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Find optimal threshold for target precision (e.g., 55% or 60%)
    thresholds = np.arange(0.4, 0.9, 0.05)
    print("\nThreshold/Precision Analysis:")
    best_thresh = 0.5
    for t in thresholds:
        preds = (y_pred_proba > t).astype(int)
        if sum(preds) > 0:
            prec = precision_score(y_test, preds)
            count = sum(preds)
            print(f"Threshold {t:.2f}: Precision {prec:.4f} (Count: {count})")
            if prec > 0.55: # If we want at least 55% win rate
                best_thresh = t
        else:
            print(f"Threshold {t:.2f}: No signals")
            
    logger.info(f"Recommended Threshold: {best_thresh}")
    
    # Save
    logger.info(f"Saving model to {model_output}...")
    joblib.dump(model, model_output)
    
    logger.info(f"Saving feature list to {features_output}...")
    with open(features_output, 'w') as f:
        json.dump(feature_cols, f)
        
    logger.info("Training complete.")

if __name__ == "__main__":
    train_regime_model()
