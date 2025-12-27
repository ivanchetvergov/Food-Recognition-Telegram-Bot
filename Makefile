# Makefile for Food Recognition Telegram Bot
# Usage: make <target>

.PHONY: help install clean data embeddings pca index features train-classifier train-regressor train all bot docker test lint

# Python interpreter
PYTHON := python

# Directories
DATA_DIR := data
IMAGES_DIR := $(DATA_DIR)/images
EMB_DIR := $(DATA_DIR)/embeddings
MODELS_DIR := models

# Files
DATASET_CSV := $(DATA_DIR)/dataset.csv
FEATURES_PARQUET := $(DATA_DIR)/features.parquet
EMBEDDINGS_NPY := $(EMB_DIR)/embeddings.npy
REDUCED_NPY := $(EMB_DIR)/all_reduced.npy
PCA_MODEL := $(MODELS_DIR)/pca.joblib
FAISS_INDEX := $(MODELS_DIR)/faiss.index
CLASSIFIER := $(MODELS_DIR)/catboost_classifier.cbm
REGRESSOR := $(MODELS_DIR)/catboost_regressor.cbm

# Model parameters
CLIP_MODEL := openai/clip-vit-base-patch32
PCA_DIM := 128

# Default target
help:
	@echo "Food Recognition Bot - Makefile"
	@echo ""
	@echo "Setup:"
	@echo "  make install      - Install Python dependencies"
	@echo "  make clean        - Remove generated files"
	@echo ""
	@echo "Data & Training Pipeline:"
	@echo "  make data         - Generate synthetic training data"
	@echo "  make embeddings   - Extract CLIP embeddings from images"
	@echo "  make pca          - Fit PCA and reduce embeddings"
	@echo "  make index        - Build FAISS index"
	@echo "  make features     - Build feature dataset"
	@echo "  make train-classifier - Train category classifier"
	@echo "  make train-regressor  - Train calorie regressor"
	@echo "  make train        - Train both models"
	@echo "  make all          - Run full pipeline (data -> train)"
	@echo ""
	@echo "Run:"
	@echo "  make bot          - Start Telegram bot"
	@echo "  make docker       - Build and run with Docker"
	@echo ""
	@echo "Development:"
	@echo "  make test         - Run tests"
	@echo "  make lint         - Run linter"

# Install dependencies
install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

# Extract embeddings
embeddings: $(EMBEDDINGS_NPY)

$(EMBEDDINGS_NPY): $(DATASET_CSV)
	@echo "Extracting CLIP embeddings..."
	@mkdir -p $(EMB_DIR)
	$(PYTHON) src/embeddings/extract_embeddings.py \
		--images-dir $(IMAGES_DIR) \
		--out-dir $(EMB_DIR) \
		--model $(CLIP_MODEL)
	@echo " Embeddings saved to $(EMB_DIR)"

# Fit PCA
pca: $(PCA_MODEL)

$(PCA_MODEL): $(EMBEDDINGS_NPY)
	@echo "Fitting PCA..."
	@mkdir -p $(MODELS_DIR)
	$(PYTHON) src/embeddings/fit_pca.py \
		--emb-dir $(EMB_DIR) \
		--out $(PCA_MODEL) \
		--dim $(PCA_DIM)
	@echo " PCA model saved to $(PCA_MODEL)"

# Build FAISS index
index: $(FAISS_INDEX)

$(FAISS_INDEX): $(REDUCED_NPY)
	@echo "Building FAISS index..."
	$(PYTHON) src/embeddings/build_index.py \
		--emb-npy $(REDUCED_NPY) \
		--out $(FAISS_INDEX)
	@echo " FAISS index saved to $(FAISS_INDEX)"

# Build features
features: $(FEATURES_PARQUET)

$(FEATURES_PARQUET): $(FAISS_INDEX) $(DATASET_CSV)
	@echo "Building feature dataset..."
	$(PYTHON) src/train/features_builder.py \
		--dataset $(DATASET_CSV) \
		--emb-dir $(EMB_DIR) \
		--index $(FAISS_INDEX) \
		--out $(FEATURES_PARQUET)
	@echo " Features saved to $(FEATURES_PARQUET)"

# Train classifier
train-classifier: $(CLASSIFIER)

$(CLASSIFIER): $(FEATURES_PARQUET)
	@echo "Training classifier..."
	$(PYTHON) src/train/train_classifier.py \
		--features $(FEATURES_PARQUET) \
		--out $(CLASSIFIER)
	@echo " Classifier saved to $(CLASSIFIER)"

# Train regressor
train-regressor: $(REGRESSOR)

$(REGRESSOR): $(FEATURES_PARQUET)
	@echo "Training regressor..."
	$(PYTHON) src/train/train_regressor.py \
		--features $(FEATURES_PARQUET) \
		--out $(REGRESSOR)
	@echo " Regressor saved to $(REGRESSOR)"

# Train both models
train: train-classifier train-regressor
	@echo " All models trained successfully"

# Full pipeline
all: data embeddings pca index features train
	@echo ""
	@echo "═══════════════════════════════════════════"
	@echo " Full pipeline completed successfully!"
	@echo "═══════════════════════════════════════════"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Set TELEGRAM_TOKEN in .env file"
	@echo "  2. Run: make bot"

# Retrain models (for active learning)
retrain: clean-models embeddings pca index features train
	@echo " Retrain completed"

# Start bot
bot:
	@echo "Starting Telegram bot..."
	$(PYTHON) -m src.bot.bot

# Docker
docker:
	docker-compose up --build

docker-detach:
	docker-compose up --build -d

docker-stop:
	docker-compose down

# Clean targets
clean: clean-data clean-models
	@echo " Cleaned all generated files"

clean-data:
	rm -rf $(IMAGES_DIR)/*.jpg $(IMAGES_DIR)/*.png
	rm -f $(DATASET_CSV)
	rm -rf $(EMB_DIR)
	rm -f $(FEATURES_PARQUET)
	rm -f $(DATA_DIR)/feedback.db

clean-models:
	rm -f $(MODELS_DIR)/*.cbm
	rm -f $(MODELS_DIR)/*.joblib
	rm -f $(MODELS_DIR)/*.index

# Development
test:
	$(PYTHON) -m pytest tests/ -v

lint:
	$(PYTHON) -m ruff check src/
	$(PYTHON) -m mypy src/

format:
	$(PYTHON) -m ruff format src/

# Show current state
status:
	@echo "Pipeline Status:"
	@echo "────────────────"
	@test -f $(DATASET_CSV) && echo " Dataset" || echo "✗ Dataset"
	@test -f $(EMBEDDINGS_NPY) && echo " Embeddings" || echo "✗ Embeddings"
	@test -f $(PCA_MODEL) && echo " PCA Model" || echo "✗ PCA Model"
	@test -f $(FAISS_INDEX) && echo " FAISS Index" || echo "✗ FAISS Index"
	@test -f $(FEATURES_PARQUET) && echo " Features" || echo "✗ Features"
	@test -f $(CLASSIFIER) && echo " Classifier" || echo "✗ Classifier"
	@test -f $(REGRESSOR) && echo " Regressor" || echo "✗ Regressor"
