"""
Enhanced ML Training for Market Regime Detection
Improvements:
- TimeSeriesSplit for proper cross-validation
- Class weight balancing for imbalanced data
- Feature importance analysis
- Automatic optimal threshold detection
- More comprehensive metrics
"""

import pandas as pd
import numpy as np
import joblib
import json
import logging
import os
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    classification_report, confusion_matrix, precision_score, 
    recall_score, f1_score, roc_auc_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight

# Try importing XGBoost
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Get base directory
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def train_regime_model(
    data_path: str = None,
    model_output: str = None,
    features_output: str = None,
    n_splits: int = 5,
    target_precision: float = 0.55
):
    """
    Train ML model for market regime detection with improved methodology.
    
    Args:
        data_path: Path to dataset CSV
        model_output: Path to save trained model
        features_output: Path to save feature names
        n_splits: Number of folds for time series cross-validation
        target_precision: Target precision for threshold optimization
    """
    
    # Use dynamic paths
    if data_path is None:
        data_path = os.path.join(_BASE_DIR, "dataset.csv")
    if model_output is None:
        model_output = os.path.join(_BASE_DIR, "regime_model.joblib")
    if features_output is None:
        features_output = os.path.join(_BASE_DIR, "features.json")
    
    if not os.path.exists(data_path):
        logger.error(f"Dataset not found at {data_path}. Run prepare_data.py first.")
        return None

    logger.info(f"Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)
    if 'date' in df.columns:
        df.set_index('date', inplace=True)
    
    # Handle unnamed columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    # Drop non-feature columns
    exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'target', 'date']
    feature_cols = [c for c in df.columns if c not in exclude_cols and not c.startswith('Unnamed')]
    
    # Handle NaN/Inf values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    
    X = df[feature_cols]
    y = df['target']
    
    logger.info(f"Features: {len(feature_cols)}")
    logger.info(f"Samples: {len(df)}")
    logger.info(f"Class distribution:\n{y.value_counts(normalize=True)}")
    
    # Time-series split for proper evaluation
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    # Calculate class weights for imbalanced data
    sample_weights = compute_sample_weight('balanced', y)
    
    # Store CV results
    cv_precision = []
    cv_recall = []
    cv_f1 = []
    cv_auc = []
    
    logger.info(f"Performing {n_splits}-fold Time Series Cross-Validation...")
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        weights_train = sample_weights[train_idx]
        
        # Initialize Model
        if HAS_XGB:
            model = XGBClassifier(
                n_estimators=150,
                learning_rate=0.08,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,  # L1 regularization
                reg_lambda=1.0,  # L2 regularization
                eval_metric='logloss',
                random_state=42
            )
            model.fit(X_train, y_train, sample_weight=weights_train)
        else:
            model = RandomForestClassifier(
                n_estimators=150,
                max_depth=10,
                min_samples_leaf=5,
                class_weight='balanced',
                random_state=42
            )
            model.fit(X_train, y_train)
        
        # Evaluate fold
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        cv_precision.append(precision_score(y_test, y_pred, zero_division=0))
        cv_recall.append(recall_score(y_test, y_pred, zero_division=0))
        cv_f1.append(f1_score(y_test, y_pred, zero_division=0))
        try:
            cv_auc.append(roc_auc_score(y_test, y_pred_proba))
        except:
            cv_auc.append(0.5)
        
        logger.info(f"Fold {fold+1}: Precision={cv_precision[-1]:.4f}, Recall={cv_recall[-1]:.4f}, F1={cv_f1[-1]:.4f}, AUC={cv_auc[-1]:.4f}")
    
    # Print CV Summary
    print("\n" + "="*60)
    print("CROSS-VALIDATION SUMMARY")
    print("="*60)
    print(f"Mean Precision: {np.mean(cv_precision):.4f} (+/- {np.std(cv_precision):.4f})")
    print(f"Mean Recall:    {np.mean(cv_recall):.4f} (+/- {np.std(cv_recall):.4f})")
    print(f"Mean F1 Score:  {np.mean(cv_f1):.4f} (+/- {np.std(cv_f1):.4f})")
    print(f"Mean AUC:       {np.mean(cv_auc):.4f} (+/- {np.std(cv_auc):.4f})")
    
    # Final Train on full data (80/20 split for final eval)
    print("\n" + "="*60)
    print("FINAL MODEL TRAINING")
    print("="*60)
    
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    weights_train = sample_weights[:split_idx]
    
    logger.info(f"Training set: {len(X_train)}, Test set: {len(X_test)}")
    
    # Final model training
    if HAS_XGB:
        logger.info("Using XGBoost Classifier")
        final_model = XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            eval_metric='logloss',
            random_state=42
        )
        final_model.fit(X_train, y_train, sample_weight=weights_train)
    else:
        logger.info("Using Random Forest Classifier")
        final_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=42
        )
        final_model.fit(X_train, y_train)
    
    # Evaluate
    y_pred_proba = final_model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Feature Importance
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE (Top 15)")
    print("="*60)
    
    if HAS_XGB:
        importance = final_model.feature_importances_
    else:
        importance = final_model.feature_importances_
    
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    for i, row in importance_df.head(15).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    # Find optimal threshold
    print("\n" + "="*60)
    print("THRESHOLD OPTIMIZATION")
    print("="*60)
    
    thresholds = np.arange(0.35, 0.75, 0.05)
    best_thresh = 0.5
    best_f1 = 0
    
    for t in thresholds:
        preds = (y_pred_proba > t).astype(int)
        if sum(preds) > 0:
            prec = precision_score(y_test, preds, zero_division=0)
            rec = recall_score(y_test, preds, zero_division=0)
            f1 = f1_score(y_test, preds, zero_division=0)
            count = sum(preds)
            print(f"Threshold {t:.2f}: Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}, Signals={count}")
            
            # Optimize for target precision with best F1
            if prec >= target_precision and f1 > best_f1:
                best_f1 = f1
                best_thresh = t
        else:
            print(f"Threshold {t:.2f}: No signals")
    
    print(f"\n✅ Recommended Threshold: {best_thresh:.2f}")
    
    # Save model and features
    logger.info(f"Saving model to {model_output}...")
    joblib.dump(final_model, model_output)
    
    logger.info(f"Saving feature list to {features_output}...")
    with open(features_output, 'w') as f:
        json.dump(feature_cols, f)
    
    # Save training report
    report_path = os.path.join(_BASE_DIR, "training_report.json")
    report = {
        "samples": len(df),
        "features": len(feature_cols),
        "cv_precision_mean": float(np.mean(cv_precision)),
        "cv_recall_mean": float(np.mean(cv_recall)),
        "cv_f1_mean": float(np.mean(cv_f1)),
        "cv_auc_mean": float(np.mean(cv_auc)),
        "test_precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "test_recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "test_f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "recommended_threshold": float(best_thresh),
        "model_type": "XGBoost" if HAS_XGB else "RandomForest",
        "top_features": importance_df.head(10)['feature'].tolist()
    }
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    logger.info(f"Saved training report to {report_path}")
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE")
    print("="*60)
    
    return report


if __name__ == "__main__":
    train_regime_model()
