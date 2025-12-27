import os
import logging
import sqlite3
from pathlib import Path
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.predict.predict_service import PredictService
from src.config import API_CONFIG, PATHS, MODEL_CONFIG
from src.utils import setup_logging, ensure_dir

logger = logging.getLogger(__name__)

# Database setup for active learning
DB_PATH = str(PATHS.feedback_db)

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS feedback
                 (image_path TEXT, predicted_category TEXT, user_category TEXT, predicted_kcal REAL, user_kcal REAL, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

class FoodBot:
    """Telegram bot for food recognition and calorie estimation."""
    
    def __init__(self, token: str):
        self.token = token
        self.predict_service = None
        self._init_predict_service()
    
    def _init_predict_service(self):
        """Initialize prediction service with proper error handling."""
        try:
            self.predict_service = PredictService()
            logger.info("PredictService initialized successfully")
        except FileNotFoundError as e:
            logger.error(f"Model files not found. Run 'make all' first: {e}")
        except Exception as e:
            logger.error(f"Failed to initialize PredictService: {e}")

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command."""
        welcome_msg = (
            "Привет! Я бот для распознавания еды.\n\n"
            "Отправь мне фото блюда, и я попробую:\n"
            "• Определить категорию блюда\n"
            "• Оценить калорийность\n\n"
            "Используй /help для списка команд."
        )
        await update.message.reply_text(welcome_msg)

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command."""
        help_msg = (
            "📚 **Команды:**\n\n"
            "/start - Начать работу\n"
            "/help - Показать эту справку\n\n"
            "📷 **Как использовать:**\n"
            "1. Отправь фото блюда\n"
            "2. Получи предсказание\n"
            "3. Подтверди или исправь\n\n"
            "Твоя обратная связь помогает улучшать модель!"
        )
        await update.message.reply_text(help_msg, parse_mode='Markdown')

    async def handle_photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle photo messages."""
        if not self.predict_service:
            await update.message.reply_text(
                "Сервис предсказаний недоступен.\n"
                "Убедитесь, что модели обучены (make all)."
            )
            return

        await update.message.reply_text("Анализирую изображение...")
        
        photo_file = await update.message.photo[-1].get_file()
        uploads_dir = ensure_dir(PATHS.uploads_dir)
        file_path = uploads_dir / f"{photo_file.file_id}.jpg"
        await photo_file.download_to_drive(str(file_path))

        try:
            result = self.predict_service.infer(str(file_path))
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            await update.message.reply_text("Ошибка при анализе изображения.")
            return
        
        confidence_emoji = "✅" if result['category_confidence'] >= MODEL_CONFIG.confidence_threshold else "🤔"
        
        response = (
            f"{confidence_emoji} **Категория:** {result['predicted_category']}\n"
            f"📊 Уверенность: {result['category_confidence']:.0%}\n\n"
            f"🔥 **Калории:** ~{result['predicted_kcal']:.0f} ккал\n\n"
            "Это верно?"
        )

        keyboard = [
            [
                InlineKeyboardButton("Yes", callback_data=f"correct|{photo_file.file_id}|{result['predicted_category']}|{result['predicted_kcal']}"),
                InlineKeyboardButton("No", callback_data=f"incorrect|{photo_file.file_id}")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await update.message.reply_text(response, reply_markup=reply_markup)

    async def button_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()
        
        data = query.data.split('|')
        action = data[0]
        
        if action == "correct":
            file_id, category, kcal = data[1], data[2], data[3]
            # Save to DB
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute("INSERT INTO feedback (image_path, predicted_category, user_category, predicted_kcal, user_kcal) VALUES (?, ?, ?, ?, ?)",
                      (f"data/uploads/{file_id}.jpg", category, category, kcal, kcal))
            conn.commit()
            conn.close()
            await query.edit_message_text(text="Thanks for your feedback!")
        elif action == "incorrect":
            await query.edit_message_text(text="Sorry about that! I'll try to learn from this.")

    def run(self):
        init_db()
        application = ApplicationBuilder().token(self.token).build()
        
        application.add_handler(CommandHandler("start", self.start))
        application.add_handler(CommandHandler("help", self.help_command))
        application.add_handler(MessageHandler(filters.PHOTO, self.handle_photo))
        application.add_handler(CallbackQueryHandler(self.button_callback))
        
        application.run_polling()

if __name__ == '__main__':
    setup_logging()
    
    TOKEN = API_CONFIG.telegram_token
    if not TOKEN or TOKEN == "your_telegram_bot_token_here":
        logger.error("Please set TELEGRAM_TOKEN in .env file")
        print("❌ Токен Telegram не настроен. Создайте .env файл с TELEGRAM_TOKEN.")
    else:
        bot = FoodBot(TOKEN)
        bot.run()
