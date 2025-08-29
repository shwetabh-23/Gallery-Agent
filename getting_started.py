import os
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
import pytz
import os

# Replace with your actual token
TELEGRAM_BOT_TOKEN = "7821449837:AAEahuirUHd8GGig3JmgNukD4qSt9Znaqe4"
IMAGE_PATH = "D:\ML-Projects\gallery_agent\data\IMG_20180714_190015.jpg"

# /start command
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("👋 Hello! Send me any message and I'll reply with an image!")

# Handles any message and sends the image
async def send_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if os.path.exists(IMAGE_PATH):
        with open(IMAGE_PATH, 'rb') as img:
            await update.message.reply_photo(photo=img, caption="🖼️ Here's the image from the server!")
    else:
        await update.message.reply_text("❌ Image not found on the server.")

# Main setup
if __name__ == "__main__":
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, send_image))

    print("✅ Bot is running. Send a message to get an image.")
    app.run_polling()