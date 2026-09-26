import os
import re



# ==========================================================
# 1️⃣ PREPROCESSING STAGE 1
# ==========================================================

def preprocessing_1(elements, source_name, actual_page_number):
    from unstructured.documents.elements import Header, Footer, PageBreak, Image

    cleaned_blocks = []
    unwanted_types = (Header, Footer, PageBreak, Image)

    for el in elements:

        if isinstance(el, unwanted_types):
            continue

        text = el.text
        if not text:
            continue

        text = text.strip()

        # Remove page numbers
        if re.match(r'^\s*(page\s*)?\d+(\s*of\s*\d+)?\s*$', text.lower()):
            continue

        # Fix hyphenation
        text = re.sub(r'-\s*\n\s*', '', text)

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        if len(text) < 20 and type(el).__name__ != "Title":
            continue

        cleaned_blocks.append({
            "text": text,
            "page": actual_page_number,
            "source": source_name,
            "block_type": type(el).__name__
        })

    return cleaned_blocks


# ==========================================================
# 2️⃣ STREAM PAGE-BY-PAGE
# ==========================================================

def stream_pdf_pages_true(pdf_path, strategy="fast"):
    from pypdf import PdfReader, PdfWriter
    from unstructured.partition.pdf import partition_pdf

    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)
    source_name = os.path.basename(pdf_path)

    for page_number in range(total_pages):

        writer = PdfWriter()
        writer.add_page(reader.pages[page_number])

        temp_path = f"_temp_page_{page_number+1}.pdf"

        with open(temp_path, "wb") as f:
            writer.write(f)

        try:
            elements = partition_pdf(
                filename=temp_path,
                strategy=strategy,
                languages=["eng"],
            )

            yield page_number + 1, elements, source_name

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)



import requests
import json
import os
import random
import time
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load .env from backend/ directory
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

logger = logging.getLogger(__name__)

# ==============================
# GEMINI API with Key Rotation
# ==============================

def _load_gemini_keys():
    """Load comma-separated Gemini API keys from environment."""
    raw = os.environ.get("GEMINI_API_KEYS", "")
    keys = [k.strip() for k in raw.split(",")
            if k.strip() and not k.strip().startswith("YOUR_KEY")]
    return keys

GEMINI_API_KEYS = _load_gemini_keys()
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
FALLBACK_TO_OLLAMA = os.environ.get("FALLBACK_TO_OLLAMA", "true").lower() == "true"

# Track temporarily exhausted keys: { key: expiry_timestamp }
_exhausted_keys = {}

def _get_available_key():
    """Pick a random non-exhausted Gemini key."""
    now = time.time()
    # Clear expired blacklist entries
    expired = [k for k, exp in _exhausted_keys.items() if now > exp]
    for k in expired:
        del _exhausted_keys[k]

    available = [k for k in GEMINI_API_KEYS if k not in _exhausted_keys]
    if not available:
        return None
    return random.choice(available)

def _mark_exhausted(key, cooldown=60):
    """Temporarily blacklist a key that hit rate limits."""
    _exhausted_keys[key] = time.time() + cooldown
    remaining = len(GEMINI_API_KEYS) - len(_exhausted_keys)
    logger.warning(f"Gemini key ...{key[-6:]} rate-limited, cooling {cooldown}s. "
                   f"{remaining} keys remaining.")


def call_gemini(prompt, max_retries=3, require_json=False):
    """
    Call Google Gemini API with random key rotation.

    Uses the NEW google.genai package (not the deprecated google.generativeai).
    On 429/503 (rate limit), blacklists the key for 60s and tries another.
    Returns the response text on success, None on failure.
    """
    if not GEMINI_API_KEYS:
        logger.warning("No Gemini API keys configured — skipping Gemini")
        return None

    from google import genai
    from google.genai import types

    for attempt in range(max_retries):
        key = _get_available_key()
        if not key:
            logger.warning("All Gemini keys exhausted, waiting 5s...")
            time.sleep(5)
            key = _get_available_key()
            if not key:
                return None

        try:
            client = genai.Client(api_key=key)

            config_kwargs = {
                "temperature": 0.2, # Slightly elevate temperature to prevent repetitive looping truncation scenarios
                "top_p": 0.9,
                "max_output_tokens": 8192,
            }
            # We do NOT set response_mime_type="application/json" for Gemini
            # because without an explicit Schema definition, Gemini's STRICT
            # json engine often prematurely assumes it has completed the object
            # and truncates the stream mid-generation!
            
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(**config_kwargs),
            )

            if response and response.text:
                logger.info(f"Gemini response via key ...{key[-6:]} "
                            f"({len(response.text)} chars)")
                print(f"\n🟢 [LLM ENGINE] Providing response via GEMINI API (Key: ...{key[-6:]})")
                print(f"🟢 [RAW OUTPUT]:\n{response.text}\n" + "="*50)
                return response.text

        except Exception as e:
            error_str = str(e).lower()

            # Rate limit or quota exhausted
            if "429" in str(e) or "resource exhausted" in error_str or "quota" in error_str:
                _mark_exhausted(key, cooldown=60)
                continue

            # Server overloaded
            if "503" in str(e) or "overloaded" in error_str:
                _mark_exhausted(key, cooldown=30)
                continue

            # Safety filter or other error
            logger.warning(f"Gemini call failed (attempt {attempt+1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(1)

    return None


# ==============================
# OPENROUTER API (Fallback Mechanism)
# ==============================
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "openrouter/free")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

def call_openrouter(prompt, require_json=False):
    """
    Fallback to OpenRouter.
    """
    if not OPENROUTER_API_KEY:
        logger.warning("No OPENROUTER_API_KEY configured — skipping OpenRouter")
        return None

    try:
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": OPENROUTER_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
        }
        if require_json:
            payload["response_format"] = {"type": "json_object"}

        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=120
        )
        
        data = response.json()
        if "choices" in data and len(data["choices"]) > 0:
            result = data["choices"][0]["message"]["content"]
            logger.info(f"OpenRouter response via {OPENROUTER_MODEL} ({len(result)} chars)")
            print(f"\n🟣 [LLM ENGINE] Providing response via OPENROUTER API (Model: {OPENROUTER_MODEL})")
            print(f"🟣 [RAW OUTPUT]:\n{result}\n" + "="*50)
            return result
        else:
            logger.warning(f"OpenRouter returned unexpected format or error: {data}")
    except Exception as e:
        logger.warning(f"OpenRouter call failed: {e}")

    return None


# ==============================
# GROQ API (Fallback Mechanism)
# ==============================
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

def call_groq(prompt, require_json=False):
    """
    Fallback to Groq using the official Groq SDK.
    """
    if not GROQ_API_KEY:
        logger.warning("No GROQ_API_KEY configured — skipping Groq")
        return None

    from groq import Groq
    try:
        client = Groq(api_key=GROQ_API_KEY)
        kwargs = {
            "model": GROQ_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2, 
            "max_tokens": 8192,
        }
        if require_json:
            kwargs["response_format"] = {"type": "json_object"}

        response = client.chat.completions.create(**kwargs)
        if response and response.choices:
            result = response.choices[0].message.content
            logger.info(f"Groq response via {GROQ_MODEL} ({len(result)} chars)")
            print(f"\n🟠 [LLM ENGINE] Providing response via GROQ API (Model: {GROQ_MODEL})")
            print(f"🟠 [RAW OUTPUT]:\n{result}\n" + "="*50)
            return result
    except Exception as e:
        logger.warning(f"Groq call failed: {e}")

    return None


# ==============================
# OLLAMA (Local Fallback)
# ==============================
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:3b"

def _call_ollama_direct(prompt, max_retries=2, require_json=False):
    """Direct Ollama call — used as fallback when Gemini is unavailable."""
    for _ in range(max_retries):
        try:
            payload = {
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.0,
                    "top_p": 0.9,
                    "repeat_penalty": 1.1,
                    "num_predict": -1,
                    "num_ctx": 8192
                }
            }
            if require_json:
                payload["format"] = "json"

            response = requests.post(
                OLLAMA_URL,
                json=payload,
                timeout=120
            )

            result = response.json()["response"]
            print(f"\n🔵 [LLM ENGINE] Providing response via LOCAL OLLAMA (Model: {OLLAMA_MODEL})")
            print(f"🔵 [RAW OUTPUT]:\n{result}\n" + "="*50)
            return result

        except Exception as e:
            logger.warning(f"Ollama call failed, retrying... {e}")

    return None


# ==============================
# UNIFIED LLM CALL
# ==============================

def call_llm(prompt, max_retries=3, require_json=False):
    """
    Unified LLM call: 
    Priority:
        1. Gemini API (random key rotation, fast, cloud)
        2. OpenRouter API
        3. Groq API (using Groq SDK)
        4. Ollama local (if FALLBACK_TO_OLLAMA=true)
    """
    print("\n" + "="*50)
    print(f"🧠 [LLM REQUEST] Generated Prompt (first 1000 chars):")
    print(f"{prompt[:1000]}...\n" + "-"*50)
    # 1. Try Gemini first
    result = call_gemini(prompt, max_retries=max_retries, require_json=require_json)
    if result:
        return result

    # 2. Try OpenRouter
    logger.info("Gemini unavailable, falling back to OpenRouter...")
    result = call_openrouter(prompt, require_json=require_json)
    if result:
        return result

    # 3. Try Groq
    logger.info("OpenRouter unavailable, falling back to Groq...")
    result = call_groq(prompt, require_json=require_json)
    if result:
        return result

    # 4. Fallback to Ollama
    if FALLBACK_TO_OLLAMA:
        logger.info("Groq unavailable, falling back to Ollama local...")
        result = _call_ollama_direct(prompt, max_retries=max_retries, require_json=require_json)
        if result:
            return result

    logger.error("All LLM providers (Gemini, OpenRouter, Groq, Ollama) failed")
    return None


# Backward-compatible alias — existing code imports this name
call_ollama = call_llm