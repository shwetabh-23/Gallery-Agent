# 📸 Gallery Agent

A smart **gallery management and retrieval system** powered by AI.  
This project integrates **face recognition, OCR, caption generation, semantic embeddings, and an agentic extension** to help users **find and explore images effortlessly** via a Telegram bot.

---

## 🚀 Features

### ✅ Part 1 – Gallery Search & Retrieval
- **Google Photos Integration**: Works with your photos and metadata.
- **Face Recognition**:  
  - **MTCNN** for face detection.  
  - **FaceNet** for face embeddings & identification.
- **OCR with Florence Model**: Extract text from images for better searchability.
- **Caption Generation**: Generate semantic captions for photos using Florence.
- **Semantic Search with Embeddings**:  
  - Uses `all-MiniLM-L6-v2` (Sentence Transformers) to create text embeddings.  
  - Enables **natural language queries** to find relevant images.
- **Telegram Bot Interface**: Query your gallery in plain text and receive the most relevant photos instantly.

### ✅ Part 2 – Agentic Internet Integration *(Upcoming)*
- Extends beyond your gallery:
  - Access the **internet** to fetch context and enrich answers.  
  - Combine gallery data + online information for smarter responses.  
  - Example: Ask _“Show me my Paris trip photos and tell me about the Eiffel Tower’s history.”_

---

## 🛠️ Tech Stack

- **Face Recognition**: [MTCNN](https://arxiv.org/abs/1604.02878), [FaceNet](https://arxiv.org/abs/1503.03832)  
- **OCR & Captions**: [Florence](https://huggingface.co/microsoft/Florence)  
- **Embeddings**: [all-MiniLM-L6-v2](https://www.sbert.net/docs/pretrained_models.html)  
- **Bot Interface**: [Telegram Bot API](https://core.telegram.org/bots/api)  
- **Storage**: Google Photos metadata + FAISS for similarity search  
- **Backend**: Python, FastAPI/Flask (depending on config)  

---