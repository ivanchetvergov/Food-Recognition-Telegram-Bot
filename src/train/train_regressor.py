import argparse
import logging
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import os

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config import MODEL_CONFIG
from src.utils import setup_logging, ensure_dir

logger = logging.getLogger(__name__)

def train_regressor(features_path, out_model_path):
    """Train CatBoost regressor for calorie estimation."""
    setup_logging()
    
    logger.info(f"Loading features from {features_path}")
    df = pd.read_parquet(features_path)
    
    X = df.drop(columns=['image_path', 'category_label', 'kcal_target'])
    y = df['kcal_target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    logger.info(f"Training set: {len(X_train)}, Test set: {len(X_test)}")
    logger.info(f"Target range: {y.min():.1f} - {y.max():.1f} kcal")
    
    model = CatBoostRegressor(
        iterations=MODEL_CONFIG.reg_iterations,
        learning_rate=MODEL_CONFIG.reg_learning_rate,
        depth=MODEL_CONFIG.reg_depth,
        loss_function='RMSE',
        verbose=100,
        early_stopping_rounds=MODEL_CONFIG.reg_early_stopping,
        random_seed=42
    )
    
    model.fit(X_train, y_train, eval_set=(X_test, y_test))
    
    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    logger.info(f"Test MAE: {mae:.2f} kcal")
    logger.info(f"Test RMSE: {rmse:.2f} kcal")
    
    ensure_dir(os.path.dirname(out_model_path))
    model.save_model(out_model_path)
    
    print(f"Regressor trained and saved to {out_model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()
    
    train_regressor(args.features, args.out)
