"""Centralized configuration for the food recognition bot."""
import os
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent

@dataclass(frozen=True)
class Paths:
    """All project paths."""
    data_dir: Path = PROJECT_ROOT / "data"
    images_dir: Path = PROJECT_ROOT / "data" / "images"
    embeddings_dir: Path = PROJECT_ROOT / "data" / "embeddings"
    models_dir: Path = PROJECT_ROOT / "models"
    uploads_dir: Path = PROJECT_ROOT / "data" / "uploads"
    
    # Data files
    dataset_csv: Path = PROJECT_ROOT / "data" / "dataset.csv"
    features_parquet: Path = PROJECT_ROOT / "data" / "features.parquet"
    feedback_db: Path = PROJECT_ROOT / "data" / "feedback.db"
    
    # Model files
    pca_model: Path = PROJECT_ROOT / "models" / "pca.joblib"
    faiss_index: Path = PROJECT_ROOT / "models" / "faiss.index"
    classifier_model: Path = PROJECT_ROOT / "models" / "catboost_classifier.cbm"
    regressor_model: Path = PROJECT_ROOT / "models" / "catboost_regressor.cbm"
    label_encoder: Path = PROJECT_ROOT / "models" / "catboost_classifier_le.joblib"
    
    # Embedding files
    embeddings_npy: Path = PROJECT_ROOT / "data" / "embeddings" / "embeddings.npy"
    reduced_embeddings_npy: Path = PROJECT_ROOT / "data" / "embeddings" / "all_reduced.npy"
    metadata_csv: Path = PROJECT_ROOT / "data" / "embeddings" / "metadata.csv"


@dataclass(frozen=True)
class ModelConfig:
    """Model hyperparameters."""
    clip_model_name: str = "openai/clip-vit-base-patch32"
    pca_components: int = 128
    faiss_top_k: int = 5
    
    # CatBoost classifier
    clf_iterations: int = 1000
    clf_learning_rate: float = 0.05
    clf_depth: int = 6
    clf_early_stopping: int = 50
    
    # CatBoost regressor
    reg_iterations: int = 1000
    reg_learning_rate: float = 0.05
    reg_depth: int = 6
    reg_early_stopping: int = 50
    
    # Thresholds
    confidence_threshold: float = 0.7
    low_confidence_threshold: float = 0.4


@dataclass(frozen=True)
class APIConfig:
    """External API configuration."""
    telegram_token: str = os.getenv("TELEGRAM_TOKEN", "")
    nutritionix_app_id: str = os.getenv("NUTRITIONIX_APP_ID", "")
    nutritionix_api_key: str = os.getenv("NUTRITIONIX_API_KEY", "")


# Global config instances
PATHS = Paths()
MODEL_CONFIG = ModelConfig()
API_CONFIG = APIConfig()


