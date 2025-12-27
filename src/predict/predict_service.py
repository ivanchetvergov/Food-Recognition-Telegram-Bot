import os
import logging
import torch
import numpy as np
import faiss
import joblib
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from catboost import CatBoostClassifier, CatBoostRegressor
import pandas as pd

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils import get_image_quality_from_array
from src.config import MODEL_CONFIG, PATHS

logger = logging.getLogger(__name__)


class PredictService:
    """Service for food image classification and calorie estimation."""
    
    def __init__(self, models_dir=None, emb_dir=None):
        models_dir = models_dir or str(PATHS.models_dir)
        emb_dir = emb_dir or str(PATHS.embeddings_dir)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        self.clip_model = CLIPModel.from_pretrained(MODEL_CONFIG.clip_model_name).to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained(MODEL_CONFIG.clip_model_name)
        
        self.pca = joblib.load(os.path.join(models_dir, 'pca.joblib'))
        self.faiss_index = faiss.read_index(os.path.join(models_dir, 'faiss.index'))
        
        self.classifier = CatBoostClassifier()
        self.classifier.load_model(os.path.join(models_dir, 'catboost_classifier.cbm'))
        
        self.regressor = CatBoostRegressor()
        self.regressor.load_model(os.path.join(models_dir, 'catboost_regressor.cbm'))
        
        self.label_encoder = joblib.load(os.path.join(models_dir, 'catboost_classifier_le.joblib'))
        self.metadata = pd.read_csv(os.path.join(emb_dir, 'metadata.csv'))
        logger.info("PredictService initialized successfully")

    def infer(self, image_path, user_meta=None):
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        
        # 1. Extract CLIP embedding
        inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            emb = self.clip_model.get_image_features(**inputs)
            emb /= emb.norm(dim=-1, keepdim=True)
            emb = emb.cpu().numpy()
            
        # 2. Reduce dimensionality
        emb_reduced = self.pca.transform(emb).astype('float32')
        
        # 3. FAISS search
        D, I = self.faiss_index.search(emb_reduced, 5)
        sim_scores = D[0]
        
        # 4. Image quality
        blur, bright, aspect = get_image_quality_from_array(image_np)
        
        # 5. Build feature vector for CatBoost
        feat_dict = {
            'blur_score': blur,
            'brightness': bright,
            'aspect_ratio': aspect,
            'hour': 12, # Default
            'weekday': 0, # Default
            'nutrition_ref_kcal': 0 # Mock
        }
        
        for j in range(emb_reduced.shape[1]):
            feat_dict[f'emb_{j}'] = emb_reduced[0][j]
            
        for j in range(len(sim_scores)):
            feat_dict[f'sim_score_{j}'] = sim_scores[j]
            
        X = pd.DataFrame([feat_dict])
        
        # 6. Predict
        probs = self.classifier.predict_proba(X)[0]
        top_idx = np.argsort(probs)[::-1][:3]
        top_categories = [
            {'label': self.label_encoder.inverse_transform([i])[0], 'prob': float(probs[i])}
            for i in top_idx
        ]
        
        predicted_kcal = self.regressor.predict(X)[0]
        
        return {
            "predicted_category": top_categories[0]['label'],
            "category_confidence": top_categories[0]['prob'],
            "top_k": top_categories,
            "predicted_kcal": float(predicted_kcal),
            "kcal_confidence": 0.7, # Mock
            "kcal_source": "catboost+faiss_lookup",
            "image_quality": {"blur": blur, "brightness": bright}
        }
