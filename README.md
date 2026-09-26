# Krishi AI - Crop Plan Backend

The backend engine for the Krishi AI Precision Agriculture platform. 

This repository contains the Django backend responsible for managing agricultural data, weather context, soil analysis, and the RAG (Retrieval-Augmented Generation) Chatbot system.

## 🚀 Key Features

*   **RAG Engine V2:** An advanced, highly optimized RAG system designed specifically for Indian farmers.
*   **Model Waterfall Fallback:** Automatically handles rate limits by seamlessly cascading from `qwen/qwen3.8-27b` to `openai/gpt-oss-20b` and `openai/gpt-oss-120b`.
*   **Memory Optimization:** Retains localized chronological context using Django SQLite sessions to bypass token payload constraints while perfectly preserving conversational state.
*   **Vector Search & Document Retrieval:** Ingests and queries thousands of pages of precision agriculture PDFs to provide highly accurate, cited agronomic advice.
*   **Django Admin Integration:** Complete graphical interface to monitor session histories and debug RAG generation flows.

## 🛠️ Tech Stack

*   **Framework:** Django / Django REST Framework
*   **Database:** SQLite (Relational), ChromaDB (Vector)
*   **LLM Providers:** Groq API, OpenRouter
*   **AI Tooling:** Langchain, HuggingFace Embeddings

## ⚙️ Setup Instructions

### 1. Environment Variables
Create a `.env` file in the `backend/` directory:
```env
# Add your specific keys here
GROQ_API_KEY=your_key_here
OPENROUTER_API_KEY=your_key_here
```

### 2. Installation
```bash
# Create a virtual environment (optional but recommended)
python3 -m venv .venv_linux
source .venv_linux/bin/activate

# Install dependencies
pip install -r requirements.txt

# Move into the Django root
cd backend/

# Run migrations
python manage.py migrate

# Start the development server
python manage.py runserver
```

### 3. Admin Access
The SQLite database stores complete conversation logs and session IDs.
To view or manage chats, log into the local Django admin:
```bash
http://localhost:8000/admin
```
