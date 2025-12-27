# Food Recognition Telegram Bot

Telegram-бот для распознавания еды по фотографии и оценки калорийности. Использует CLIP для извлечения эмбеддингов, FAISS для поиска похожих блюд и CatBoost для классификации и регрессии.

## Возможности

- Распознавание категории блюда по фото
- Оценка калорийности
- Поиск похожих блюд (FAISS)
- Active Learning: обучение на обратной связи пользователей

## Быстрый старт

### 1. Установка

```bash
# Клонируй репозиторий и перейди в директорию
make install
```

### 2. Настройка окружения

Создай `.env` файл:

```bash
cp .env.example .env
# Отредактируй .env и добавь свой TELEGRAM_TOKEN
```

### 3. Обучение моделей

```bash
# Полный пайплайн: данные → эмбеддинги → обучение
make all
```

### 4. Запуск бота

```bash
make bot
```

## Makefile команды

```bash
make help           # Показать все команды
make status         # Проверить статус пайплайна

# Пайплайн по шагам
make data           # Генерация синтетических данных
make embeddings     # Извлечение CLIP эмбеддингов
make pca            # Снижение размерности (PCA)
make index          # Построение FAISS индекса
make features       # Сборка признаков
make train          # Обучение моделей

# Запуск
make bot            # Запустить бота
make docker         # Запустить в Docker

# Разработка
make test           # Запустить тесты
make lint           # Проверка кода
make clean          # Очистить сгенерированные файлы
```

## Docker

```bash
# Сборка и запуск
make docker

# Или напрямую
docker-compose up --build
```

## Структура проекта

```
├── data/
│   ├── images/              # Изображения для обучения
│   ├── embeddings/          # CLIP эмбеддинги
│   ├── dataset.csv          # Датасет
│   ├── features.parquet     # Признаки для обучения
│   └── gen_synthetic_food.py
├── models/
│   ├── pca.joblib           # PCA модель
│   ├── faiss.index          # FAISS индекс
│   ├── catboost_classifier.cbm
│   └── catboost_regressor.cbm
├── src/
│   ├── config.py            # Централизованная конфигурация
│   ├── utils.py             # Общие утилиты
│   ├── embeddings/          # Извлечение эмбеддингов
│   ├── train/               # Обучение моделей
│   ├── predict/             # Сервис предсказаний
│   └── bot/                 # Telegram бот
├── Makefile                 # Автоматизация
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Конфигурация

Все параметры в [src/config.py](src/config.py):

- `MODEL_CONFIG` — гиперпараметры моделей
- `PATHS` — пути к файлам и директориям
- `API_CONFIG` — токены и ключи API
- `FOOD_CATEGORIES` — категории еды с метаданными

## Active Learning

Бот сохраняет обратную связь пользователей в `data/feedback.db`. Для дообучения:

```bash
make retrain
```

## Метрики (MVP)

| Метрика | Целевое значение |
|---------|------------------|
| Classifier Top-1 Accuracy | ≥ 60% |
| Classifier Top-3 Accuracy | ≥ 75% |
| Regressor MAE | ≤ 200 kcal |

## Требования

- Python 3.10+
- ~4GB RAM для CLIP модели
- (Опционально) GPU для ускорения
