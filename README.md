# Krishi AI - Precision Agriculture Backend

The central intelligence and data management engine for the Krishi AI Precision Agriculture platform. 

This repository contains the Django backend responsible for modeling geographic land parcels, caching geospatial environmental data, and powering a highly context-aware Retrieval-Augmented Generation (RAG) advisory chatbot designed specifically for Indian agriculture.

---

## 🏗️ Architecture Overview

The backend acts as the orchestrator between user interfaces, external environmental APIs, and Large Language Models. 

When a farmer asks a question about their crop, the system doesn't just pass the question to an LLM. It first executes a spatial data pipeline:
1. **Locates the Farm:** Retrieves the farmer's specific field boundaries from the PostGIS spatial database.
2. **Gathers Environment Context:** Checks the geospatial cache for weather trends (Open-Meteo) and edaphic soil profiles (SoilGrids) for that exact coordinate.
3. **Retrieves Agronomic Science:** Queries a local ChromaDB vector store containing thousands of pages of Indian agricultural research and best practices.
4. **Synthesizes:** Compiles the weather, soil, literature, and conversation history into a massive context window.
5. **Generates Advice:** Routes the highly-enriched prompt through an advanced model waterfall to generate hyper-personalized agronomic advice.

---

## 🧩 Core Modules (Django Apps)

### `lands` - Spatial Farm Modeling
Manages the geographic representation of farmer fields.
* Uses GeoDjango and PostGIS to store farm locations as `PointField` and boundaries as `PolygonField`.
* Tracks area (hectares) and spatial relationships for mapping.

### `data` - Geospatial Environmental Cache
A critical optimization layer to prevent redundant external API calls and handle rate limits.
* **WeatherCache:** Stores precipitation, radiation, and temperature metrics from Open-Meteo tied to a spatial coordinate.
* **SoilCache:** Stores deep soil profiles (pH, Organic Carbon, Bulk Density, Texture, NPK) fetched from ISRIC SoilGrids.
* Reuses cached data for multiple farms within a 2km radius to save bandwidth and compute.

### `rag` - The RAG Engine V2
The brain of the Krishi AI advisory system.
* **Ingestion Pipeline:** Chunks and embeds massive PDF datasets (using HuggingFace embeddings) into a local ChromaDB vector store.
* **Model Waterfall:** Intelligently routes requests to bypass rate limits (like Groq's 429 errors). Cascades from `qwen3.8-27b` to `gpt-oss-20b` down to `gpt-oss-120b` automatically upon failure.
* **Memory Management:** Persists entire conversation trajectories (`ChatSession` & `ChatTurn`) into relational SQLite. It filters and injects rolling chronological context into the LLM without exploding token limits.

### `accounts` - Identity
* Custom User models and authentication routing for farmers and administrators.

---

## 🛠️ Technology Stack

* **Core Framework:** Django 4.x & Django REST Framework (DRF)
* **Spatial & Relational Database:** GeoDjango / SQLite (dev) / PostGIS (prod capability)
* **Vector Database:** ChromaDB (Local persistence)
* **LLM Orchestration:** Langchain, HuggingFace Local Embeddings
* **LLM Providers:** Groq API, OpenRouter
* **External APIs:** Open-Meteo (Weather), SoilGrids (Soil)

---

## ⚙️ Setup & Installation

### 1. Environment Configuration
Create a `.env` file in the `backend/` directory. Do **not** commit this file.
```env
# API Keys for LLM Generation
GROQ_API_KEY=your_groq_key_here
OPENROUTER_API_KEY=your_openrouter_key_here
```

### 2. Virtual Environment & Dependencies
```bash
# Create and activate a virtual environment
python3 -m venv .venv_linux
source .venv_linux/bin/activate

# Install all backend packages
pip install -r requirements.txt
```

### 3. Database Initialization
```bash
# Move into the Django root
cd backend/

# Run spatial and relational migrations
python manage.py migrate
```

### 4. Running the Server
```bash
# Start the Django development server
python manage.py runserver
```
The API will be available at `http://localhost:8000/`.

---

## 📊 Administration & Debugging

Krishi AI heavily utilizes the Django Admin panel for observability. 
By navigating to `http://localhost:8000/admin`, administrators can:
* Inspect raw `WeatherCache` and `SoilCache` geographic entries.
* View exact coordinates and boundaries of registered `Land` fields.
* Read through full `ChatSession` transcripts to debug the LLM's RAG performance and history retention.
