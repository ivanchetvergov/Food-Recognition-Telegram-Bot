import os
import argparse
import logging
import pandas as pd
import numpy as np
import faiss
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils import get_image_quality, setup_logging

logger = logging.getLogger(__name__)

def build_features(dataset_csv, emb_dir, index_path, out_path):
    """Build feature dataset from embeddings and metadata."""
    setup_logging()
    
    logger.info(f"Loading dataset from {dataset_csv}")
    df = pd.read_csv(dataset_csv)
    embeddings_reduced = np.load(os.path.join(emb_dir, 'all_reduced.npy')).astype('float32')
    index = faiss.read_index(index_path)
    
    meta_df = pd.read_csv(os.path.join(emb_dir, 'metadata.csv'))
    full_meta = meta_df.merge(df, on='image_path', how='left')
    
    logger.info(f"Building features for {len(full_meta)} samples")
    features = []
    
    for i, row in tqdm(full_meta.iterrows(), total=len(full_meta), desc="Building features"):
        emb = embeddings_reduced[i].reshape(1, -1)

        D, I = index.search(emb, 6)
        sim_scores = D[0][1:6]

        blur, bright, aspect = get_image_quality(row['image_path'])

        # mock
        hour = 12
        weekday = 0

        nutrition_ref_kcal = row['kcal_ref'] if 'kcal_ref' in row else 0
        
        feat_row = {
            'image_path': row['image_path'],
            'category_label': row['category_label'],
            'kcal_target': row['kcal_ref'],
            'blur_score': blur,
            'brightness': bright,
            'aspect_ratio': aspect,
            'hour': hour,
            'weekday': weekday,
            'nutrition_ref_kcal': nutrition_ref_kcal
        }

        for j in range(emb.shape[1]):
            feat_row[f'emb_{j}'] = emb[0][j]

        for j in range(len(sim_scores)):
            feat_row[f'sim_score_{j}'] = sim_scores[j]
            
        features.append(feat_row)
        
    feat_df = pd.DataFrame(features)
    feat_df.to_parquet(out_path, index=False)
    print(f"Features built and saved to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--emb-dir", type=str, required=True)
    parser.add_argument("--index", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()
    
    build_features(args.dataset, args.emb_dir, args.index, args.out)
