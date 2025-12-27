import argparse
import logging
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config import MODEL_CONFIG
from src.utils import setup_logging, ensure_dir

logger = logging.getLogger(__name__)


def train_classifier(features_path, out_model_path):
    """Train CatBoost classifier for food category prediction."""
    setup_logging()
    
    logger.info(f"Loading features from {features_path}")
    df = pd.read_parquet(features_path)
    
    le = LabelEncoder()
    df['category_encoded'] = le.fit_transform(df['category_label'])
    
    X = df.drop(columns=['image_path', 'category_label', 'kcal_target', 'category_encoded'])
    y = df['category_encoded']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    logger.info(f"Training set: {len(X_train)}, Test set: {len(X_test)}")
    logger.info(f"Categories: {le.classes_}")
    
    model = CatBoostClassifier(
        iterations=MODEL_CONFIG.clf_iterations,
        learning_rate=MODEL_CONFIG.clf_learning_rate,
        depth=MODEL_CONFIG.clf_depth,
        loss_function='MultiClass',
        early_stopping_rounds=MODEL_CONFIG.clf_early_stopping,
        random_seed=42
    )
    
    model.fit(X_train, y_train, eval_set=(X_test, y_test))
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Test accuracy: {accuracy:.4f}")
    
    ensure_dir(os.path.dirname(out_model_path))
    model.save_model(out_model_path)
    
    # Save label encoder
    le_path = out_model_path.replace('.cbm', '_le.joblib')
    joblib.dump(le, le_path)
    
    print(f"Classifier trained and saved to {out_model_path}")
    print(f"Label encoder saved to {le_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()
    
    train_classifier(args.features, args.out)
