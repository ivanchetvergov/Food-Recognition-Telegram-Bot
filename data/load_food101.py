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
    logging.info("Loading Food101 dataset from Hugging Face datasets")
    dataset = load_dataset("ethz/food101", split="train")
    
    categories = dataset.features["label"].names
    PATHS.images_dir.mkdir(parents=True, exist_ok=True)
    
    data = []
    counts = {cat: 0 for cat in categories}
    total_to_download = len(categories) * samples_per_class

    with tqdm(total=total_to_download, desc="Downloading Food101 images") as pbar:
        for item in dataset:
            label_idx = item["label"]
            label_name = categories[label_idx]
            
            if counts[label_name] < samples_per_class:
                img = item['image']
                img_name = f"{label_name}_{counts[label_name]:04d}.jpg"
                img_path = PATHS.images_dir / img_name
                
                img.convert("RGB").save(img_path, quality=90)
                
                data.append({
                    'image_path': str(img_path),
                    'dish_name': label_name.replace('_', ' ').title(),
                    'category_label': label_name,
                    'kcal_ref': 0, 
                    'source': 'food101'
                })
                counts[label_name] += 1
                pbar.update(1)
                
            if all(c >= samples_per_class for c in counts.values()):
                break
            
    df = pd.DataFrame(data)
    df.to_csv(PATHS.food101_csv, index=False)
    logging.info(f"Saved Food101 subset categories {len(categories)} to {PATHS.images_dir}")
    
if __name__ == "__main__":
    download_food101(samples_per_class=50)