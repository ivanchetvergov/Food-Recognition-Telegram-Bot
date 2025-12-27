import os 
import sys
from pathlib import Path
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import PATHS

import logging
logging.basicConfig(level=logging.INFO)

def download_food101(samples_per_class: int = 20):
    """Download Food101 dataset and save a subset locally."""
    logging.info("Loading Food101 dataset from Hugging Face Hub")
    dataset = load_dataset("ethz/food101", split="train")
    
    categories = dataset.features["label"].names
    PATHS.food101_dir.mkdir(parents=True, exist_ok=True)
    
    data = []
    counts = {cat: 0 for cat in categories}

    for item in tqdm(dataset, desc="Downloading Food101 images"):
        label_idx = item["label"]
        label_name = categories[label_idx]
        
        if counts[label_name] < samples_per_class:
            img = item['image']
            img_name = f"{label_name}_{counts[label_name]:04d}.jpg"
            img_path = PATHS.food101_dir / img_name
            
            img.convert("RGB").save(img_path, quality=90)
            
            data.append({
                'image_path': str(img_path),
                'dish_name': label_name.replace('_', ' ').title(),
                'category_label': label_name,
                'kcal_ref': None,  # Placeholder, real kcal values can be added later
                'source': 'food101'
            })
            counts[label_name] += 1
            
    df = pd.DataFrame(data)
    df.to_csv(PATHS.food101_csv, index=False)
    logging.info(f"Saved Food101 subset categories {len(categories)} to {PATHS.food101_dir}")
    
if __name__ == "__main__":
    download_food101(samples_per_class=50)