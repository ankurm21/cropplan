import json
import re
from transformers import AutoTokenizer
try:
    from .utility import stream_pdf_pages_true, preprocessing_1, call_ollama
except ImportError:
    from utility import stream_pdf_pages_true, preprocessing_1, call_ollama

# ==============================
# CONFIG
# ==============================

HF_MODEL = "Qwen/Qwen2.5-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)

TARGET_TOKENS = 500

ALLOWED_TOPICS = {
    "research": "Research",
    "advisory": "Advisory",
    "economic": "Economic",
    "policy": "Policy",
    "weather": "Weather",
    "technology": "Technology",
    "pest": "Pest",
    "disease": "Disease",
    "soil": "Soil",
    "market": "Market",
    "sustainability": "Sustainability",
    "infrastructure": "Infrastructure"
}


# ==============================
# TOKEN BUCKETING
# ==============================

def stream_token_buckets(block_stream, target_tokens=500):

    bucket_text = ""
    bucket_tokens = 0
    bucket_id = 1
    bucket_pages = set()
    bucket_source = None

    for block in block_stream:

        text = block["text"].strip()
        if not text:
            continue

        tokens = tokenizer.encode(text, add_special_tokens=False)
        token_count = len(tokens)

        if bucket_tokens + token_count > target_tokens:
            if bucket_text.strip():
                yield {
                    "bucket_id": bucket_id,
                    "text": bucket_text.strip(),
                    "pages": sorted(bucket_pages),
                    "source": bucket_source
                }
                bucket_id += 1

            bucket_text = text
            bucket_tokens = token_count
            bucket_pages = {block["page"]}
            bucket_source = block["source"]

        else:
            bucket_text += " " + text
            bucket_tokens += token_count
            bucket_pages.add(block["page"])
            bucket_source = block["source"]

    if bucket_text.strip():
        yield {
            "bucket_id": bucket_id,
            "text": bucket_text.strip(),
            "pages": sorted(bucket_pages),
            "source": bucket_source
        }


# ==============================
# LLM METADATA EXTRACTION
# ==============================

import requests
import json

def call_llm_for_metadata(prompt):
    """Call Ollama locally, strictly using qwen2.5:7b for metadata extraction."""
    payload = {
        "model": "qwen2.5:7b",
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.0,
            "top_p": 0.9,
            "num_ctx": 4096
        }
    }
    try:
        response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=120)
        response.raise_for_status()
        result = response.json().get("response", "")
        print(f"[LOCAL OLLAMA (qwen2.5:7b) ENRICHMENT] Response length: {len(result)}")
        return result
    except Exception as e:
        print(f"[OLLAMA ERROR]: {e}")
        return None


def llm_extract_metadata(text):
    """Extract structured metadata from a text chunk using LLM."""

    prompt = f"""You are an agricultural document classifier. Analyze the text below and extract metadata.

Return ONLY a JSON object. No explanation.

FIELD RULES:

1. primary_topic — Pick EXACTLY ONE from this list:
   - "Soil"           = soil types, pH, fertility, organic carbon, amendments
   - "Pest"           = insects, diseases, pest control, IPM, pesticides
   - "Weather"        = rainfall, temperature, climate, monsoon, drought
   - "Advisory"       = farming recommendations, best practices, crop planning, yield improvement
   - "Economic"       = costs, prices, MSP, market trends, income, profit
   - "Policy"         = government schemes, subsidies, NREGA, PM-KISAN
   - "Technology"     = sensors, IoT, drones, AI, precision agriculture, automation
   - "Market"         = supply chain, export, trade, commodity markets, mandi
   - "Sustainability" = organic farming, conservation, water saving, climate-smart
   - "Infrastructure" = irrigation systems, cold storage, roads, warehouses
   - "Research"       = ONLY if the text is purely about a research study methodology or literature review
   - "Disease"        = plant diseases, fungal infections, bacterial wilt
   NOTE: If the text gives farming advice, tips, or crop recommendations, choose "Advisory" NOT "Technology".
   NOTE: If the text is NOT about agriculture at all, set relevance_score to 0.0.

2. crop_entities — List ONLY actual crop/plant names mentioned (e.g., rice, wheat, tomato, sorghum).
   Do NOT include: technologies, methods, tools, chemicals, or general words.

3. practice_entities — List farming practices/methods (e.g., drip irrigation, crop rotation, mulching, seed treatment).
   Do NOT include: crop names, place names, or general descriptions.

4. region_entities — List geographic locations (states, districts, countries).

5. numeric_signals — List ONLY useful agricultural measurements mentioned.
   INCLUDE: pH values, yield (q/ha), rainfall (mm), temperature (°C), percentages of nutrient content, cost (₹).
   EXCLUDE: citation numbers like [34], publication years like 2018, page numbers, reference IDs.

6. relevance_score — Rate 0.0 to 1.0 how useful this text is for an Indian farmer:
   - 0.0-0.2 = bibliography, references list, table of contents, non-agriculture content
   - 0.3-0.5 = general background, methodology description, abstract
   - 0.6-0.8 = useful agricultural information, data, regional practices
   - 0.9-1.0 = directly actionable farming advice with specific crops, yields, or recommendations

EXAMPLE:
Text: "In Maharashtra, rice cultivation in kharif season yields 35-40 q/ha. Farmers should apply DAP at 100kg/ha during sowing. Drip irrigation saves 30% water compared to flood irrigation."
Output:
{{"primary_topic": "Advisory", "crop_entities": ["rice"], "practice_entities": ["drip irrigation", "flood irrigation", "DAP application"], "region_entities": ["Maharashtra"], "numeric_signals": ["35-40 q/ha yield", "100kg/ha DAP", "30% water saving"], "relevance_score": 0.9}}

Now classify this text:
\"\"\"{text}\"\"\"

JSON:"""

    raw = call_llm_for_metadata(prompt)

    if not raw:
        return None

    # Try to parse JSON — handle common LLM quirks
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Try extracting JSON from markdown fences
    import re
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            # Fix trailing commas
            cleaned = re.sub(r',\s*([}\]])', r'\1', match.group(0))
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                pass

    return None


# ==============================
# NORMALIZATION
# ==============================

def normalize_primary_topic(topic):

    if not topic:
        return "Research"

    topic = topic.strip().lower()

    for key in ALLOWED_TOPICS:
        if key in topic:
            return ALLOWED_TOPICS[key]

    return "Research"


def clean_list(value):
    if isinstance(value, list):
        return list(dict.fromkeys([str(v).strip() for v in value if v]))
    return []


# ==============================
# SCHEMA ENFORCEMENT
# ==============================

def enforce_schema(meta, bucket, chunk_counter, text):

    if not meta:
        meta = {}

    relevance = float(meta.get("relevance_score", 0.5))
    relevance = max(0.0, min(1.0, relevance))

    if len(text) < 100:
        relevance *= 0.5

    primary_topic = normalize_primary_topic(meta.get("primary_topic"))

    return {
        "doc_id": bucket["source"].replace(".pdf", "").replace(" ", "_"),
        "chunk_id": str(chunk_counter),
        "document_name": bucket["source"],
        "page_numbers": bucket["pages"],
        "primary_topic": primary_topic,
        "crop_entities": clean_list(meta.get("crop_entities")),
        "practice_entities": clean_list(meta.get("practice_entities")),
        "region_entities": clean_list(meta.get("region_entities")),
        "numeric_signals": clean_list(meta.get("numeric_signals")),
        "relevance_score": relevance,
        "text": text.strip()
    }


# ==============================
# MAIN PROCESSOR
# ==============================

def process_pdf(pdf_path, output_json="processed_chunks.json"):

    all_chunks = []
    chunk_counter = 1

    def cleaned_block_stream():
        for page_number, elements, source_name in stream_pdf_pages_true(pdf_path):
            cleaned_blocks = preprocessing_1(elements, source_name, page_number)
            for block in cleaned_blocks:
                yield block

    for bucket in stream_token_buckets(cleaned_block_stream(), TARGET_TOKENS):

        text = bucket["text"].strip()
        if not text:
            continue

        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) > TARGET_TOKENS:
            tokens = tokens[:TARGET_TOKENS]
            text = tokenizer.decode(tokens)

        meta = llm_extract_metadata(text)

        structured = enforce_schema(meta, bucket, chunk_counter, text)

        all_chunks.append(structured)
        chunk_counter += 1

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(all_chunks)} structured chunks to {output_json}")


# ==============================
# ENTRY POINT
# ==============================

if __name__ == "__main__":

    PDF_PATH = r"C:\Users\hp\Downloads\2303.06049v1.pdf"
    process_pdf(PDF_PATH)