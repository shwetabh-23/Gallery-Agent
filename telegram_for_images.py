import os
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
from prompt import build_structured_query_prompt
from faiss_retrieval_2 import retrieve_top_k_per_embedding, fuse_embedding_results
from phi3 import load_phi3, get_phi3_inference
from sentence_embedding import load_embedding_model
from get_recommendations import get_image_recommendations

phi3_model, phi3_tokenizer = load_phi3()
embedding_model, embedding_tokenizer = load_embedding_model()

# Replace with your actual token
TELEGRAM_BOT_TOKEN = "7821449837:AAEahuirUHd8GGig3JmgNukD4qSt9Znaqe4"
# Replace this with your image folder path or retrieval logic
IMAGE_FOLDER = "D:/ML-Projects/gallery_agent/data/"

# /start command handler
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        "👋 Hi, I am a gallery agent named *agent_biswas*.\n"
        "I can help you find images from your gallery (to some degree).\n\n"
        "📨 Just send me a description, and I’ll try to find matching images."
    )
    await update.message.reply_markdown(msg)

# Message handler for queries
async def handle_query(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.message.text
    await update.message.reply_text("🔍 Searching for matching images... for the 10 images I return, there is a 50% chance one if it will be the one you are searching for :)))")
    image_paths = get_image_recommendations(query, phi3_model, phi3_tokenizer, embedding_model, embedding_tokenizer)

    if not image_paths:
        await update.message.reply_text("❌ Sorry, no matching images found.")
        return

    for path in image_paths:
        try:
            with open(path, "rb") as img_file:
                await update.message.reply_photo(photo=img_file)
        except Exception as e:
            await update.message.reply_text(f"⚠️ Error sending image: {e}")

# Main app setup
if __name__ == "__main__":
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_query))

    print("✅ Bot is running. Send a message to get an image.")
    app.run_polling()
