"""
rag_engine.py
────────────────────────────────────────────────────────────
Production RAG Engine for KrishiPlan.

Pipeline:
    1.  Receive user query  (e.g. "What crops grow best in clay soil?")
    2.  Semantic search in ChromaDB  →  top-k relevant chunks
    3.  Construct an agriculture-expert prompt with context
    4.  Call Ollama (Qwen2.5:3B)  →  grounded response
    5.  Return response + source references

Also provides:
    - generate_crop_plan()   → called by DataExtractView — returns STRUCTURED JSON
    - contextual_query()     → plan-aware chatbot for authenticated users
────────────────────────────────────────────────────────────
"""

import logging
import json
import re
from typing import List, Dict, Any, Optional

from .vector_store import search as vector_search, collection_stats
from .utility import call_ollama

logger = logging.getLogger(__name__)


# ── Prompt Templates ──────────────────────────────────────

SYSTEM_PROMPT = """You are **Krishi AI**, an expert precision-agriculture advisor exclusively for Indian farmers (Kisans).

RULES:
• Answer ONLY from the provided CONTEXT documents. If the context does not cover the question, state honestly that you do not have that specific KVK or Mandi data.
• Be highly practical, actionable, and strictly localized to Indian agronomy.
• Use Indian farming terminology where appropriate (Kharif, Rabi, Zaid, Mandi, KVK, Panchayat).
• When recommending fertilizers or practices, prioritize sustainable AI practices, precision dosing, and ISRO weather integration conceptually.
• Use bullet points for clear recommendations.
• If numeric data (yield in quintals/ha, cost in INR, pH) is available in context, cite it strictly.
• Keep answers concise (150-300 words) unless the user asks for detail.
• End with a one-line practical 'Krishi Tip' the farmer can act on today."""

RAG_PROMPT_TEMPLATE = """{system}

─── CONTEXT DOCUMENTS ───
{context_block}
─── END CONTEXT ───

USER QUESTION: {query}

Provide a well-structured, evidence-based answer using ONLY the context above."""


# ─── STRUCTURED CROP PLAN PROMPT ──────────────────────────
# This prompt forces the LLM to return valid JSON matching our schema.

STRUCTURED_PLAN_PROMPT = """You are KrishiPlan AI, an expert agriculture advisor.

You MUST respond with ONLY a valid JSON object. No markdown, no explanation before or after. Just pure JSON.

─── KNOWLEDGE BASE ───
{context_block}
─── END KNOWLEDGE BASE ───

─── FARM PROFILE ───
Location: {location_name} (Coordinates: {lat}°N, {lon}°E)
Area: {area_ha} hectares
Farming Purpose: {farming_purpose}
Experience: {experience} years
Land Ownership: {land_ownership}
Budget: {budget} per acre/season
Labour: {labour}
Water Sources: {water_sources}
Irrigation Type: {irrigation_type}
Water Availability: {water_availability}
Self-Reported Soil: {self_soil}
Previous Crops: {previous_crops}
Known Pest Issues: {pest_issues}
Fertilizers Used: {fertilizers_used}
Market Access: {market_access}
Tractor Available: {tractor}
Storage Facility: {storage}

API Soil Data: pH={soil_ph}, Texture={soil_texture}, Organic Carbon={soil_oc}, Bulk Density={soil_bd}
API Weather Data: Rainfall Trend={rainfall_trend}, Past Rain={past_rain}, Forecast Rain={forecast_rain}, Avg Temp={avg_temp}, Humidity={humidity}, Solar Radiation={solar_rad}
─── END PROFILE ───

Generate a crop plan as a JSON object with this EXACT structure:

{{
  "recommended_crops": [
    {{
      "name": "crop name with variety if known",
      "type": "primary or rotation or intercrop",
      "season": "Kharif/Rabi/Zaid with months",
      "reason": "why this crop suits this farmer (reference soil, weather, budget)",
      "expected_yield": "yield range per hectare",
      "estimated_cost": "input cost per acre in INR",
      "water_requirement": "low/moderate/high with mm range"
    }}
  ],
  "seasonal_calendar": [
    {{
      "month": "month range",
      "activity": "what to do"
    }}
  ],
  "soil_management": {{
    "current_status": "summary of soil condition based on API + self-reported data",
    "recommendations": ["recommendation 1", "recommendation 2", "recommendation 3"]
  }},
  "water_management": {{
    "strategy": "overall water strategy based on sources and availability",
    "recommendations": ["recommendation 1", "recommendation 2", "recommendation 3"]
  }},
  "pest_management": {{
    "risk_pests": ["pest1", "pest2"],
    "preventive_measures": ["measure 1", "measure 2", "measure 3"]
  }},
  "risk_factors": [
    {{
      "risk": "risk description",
      "severity": "low/medium/high",
      "mitigation": "what to do about it"
    }}
  ],
  "budget_analysis": {{
    "budget_tier": "farmer's budget category",
    "estimated_input_cost": "total input cost estimate per acre",
    "estimated_revenue": "expected revenue per acre",
    "roi_estimate": "return on investment percentage"
  }},
  "summary": "2-3 sentence executive summary of the entire plan"
}}

IMPORTANT RULES:
- recommended_crops MUST have 2-4 crops
- seasonal_calendar MUST have 4-6 entries covering the full year
- Each section MUST reference the specific farm data (soil pH, rainfall, budget, etc.)
- For budget_analysis, match recommendations to the farmer's stated budget tier
- For pest_management, prioritize the farmer's reported pest issues
- If the farmer has limited labour, recommend low-labour crops
- If no storage facility, avoid crops needing long post-harvest storage
- Output ONLY the JSON. No other text."""


CONTEXTUAL_PROMPT_TEMPLATE = """{system}

─── AGRICULTURAL KNOWLEDGE BASE ───
{context_block}
─── END KNOWLEDGE BASE ───

─── FARMER'S ACTIVE PLAN CONTEXT ───
Weather: {weather}
Soil: {soil}
Farm Details: {farm}
Area: {area} hectares
─── END PLAN CONTEXT ───

The farmer has an active crop plan (shown above).

USER QUESTION: {query}

Answer the question using BOTH the knowledge base AND the farmer's specific data.
Be personal — reference their soil type, weather, and farm details directly.
Provide practical, actionable advice tailored to their situation."""


class RAGEngine:
    """
    RAG (Retrieval-Augmented Generation) Engine.

    Usage:
        engine = RAGEngine()
        result = engine.query("What fertilizer works for rice in clay soil?")
    """

    def __init__(self, top_k: int = 6, min_relevance: float = 0.25):
        self.top_k = top_k
        self.min_relevance = min_relevance
        logger.info("RAGEngine initialised (min_relevance=%.2f)", min_relevance)

    # ────────────────────────────────────────────────────────
    #  Core pipeline
    # ────────────────────────────────────────────────────────

    def retrieve_context(
        self,
        query: str,
        top_k: Optional[int] = None,
        topic_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Semantic search + metadata-based reranking.
        Over-fetches 2x, then reranks using enriched metadata
        to boost chunks whose entities match the query.
        """
        k = top_k or self.top_k
        fetch_k = k * 2  # over-fetch for reranking headroom

        hits = vector_search(
            query=query,
            top_k=fetch_k,
            topic_filter=topic_filter,
            min_relevance=self.min_relevance,
        )

        # Rerank using enriched metadata if available
        if hits:
            hits = self._rerank_with_metadata(query, hits)

        hits = hits[:k]  # trim to requested count
        logger.info(f"Retrieved {len(hits)} chunks (reranked) for: '{query[:60]}'")
        return hits

    def _rerank_with_metadata(self, query: str, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Boost chunks whose enriched metadata matches query keywords.
        This makes crop_entities, practice_entities, region_entities
        actually influence retrieval — not just cosmetic labels.
        """
        query_lower = query.lower()
        query_words = set(query_lower.split())

        for hit in hits:
            boost = 0.0
            base_score = hit.get("score", 0)

            # Boost if crop entities match query words
            crops = hit.get("crop_entities", "").lower()
            if crops:
                matching_crops = sum(1 for w in query_words if w in crops)
                boost += matching_crops * 0.08

            # Boost if practice entities match
            practices = hit.get("practice_entities", "").lower()
            if practices:
                matching_practices = sum(1 for w in query_words if w in practices)
                boost += matching_practices * 0.06

            # Boost if region entities match
            regions = hit.get("region_entities", "").lower()
            if regions:
                matching_regions = sum(1 for w in query_words if w in regions)
                boost += matching_regions * 0.05

            # Boost by relevance score (quality signal from enrichment)
            relevance = float(hit.get("relevance_score", 0))
            boost += relevance * 0.05  # max +0.05 for score=1.0

            # Apply boost (capped at 0.2 to not overpower semantic similarity)
            hit["score"] = round(base_score + min(boost, 0.2), 4)

        # Re-sort by boosted score
        hits.sort(key=lambda h: h.get("score", 0), reverse=True)
        return hits

    def _build_context_block(self, hits: List[Dict[str, Any]]) -> str:
        """Format retrieved chunks into a numbered context block for the prompt."""
        if not hits:
            return "(No relevant documents found in the knowledge base.)"

        blocks = []
        for i, hit in enumerate(hits, 1):
            source = hit.get("document_name", "unknown")
            topic  = hit.get("primary_topic", "")
            pages  = hit.get("page_numbers", "")
            score  = hit.get("score", 0)
            crops  = hit.get("crop_entities", "")
            # Use raw_text (clean) for LLM prompt; the 'text' field may have
            # enrichment tags prepended for embedding purposes
            text   = hit.get("raw_text", "") or hit.get("text", "")

            header = f"[{i}] Source: {source} | Topic: {topic} | Pages: {pages} | Relevance: {score}"
            if crops:
                header += f" | Crops: {crops}"

            blocks.append(f"{header}\n{text}")

        return "\n\n".join(blocks)

    def generate_response(
        self,
        query: str,
        context_hits: List[Dict[str, Any]],
    ) -> str:
        """Build prompt and call Ollama for generation."""
        context_block = self._build_context_block(context_hits)

        prompt = RAG_PROMPT_TEMPLATE.format(
            system=SYSTEM_PROMPT,
            context_block=context_block,
            query=query,
        )

        logger.info(f"Sending prompt to Ollama ({len(prompt)} chars)")
        response = call_ollama(prompt)

        if not response:
            return (
                "I apologise — I could not generate a response at this time. "
                "Please ensure the Ollama service is running locally."
            )
        return response.strip()

    def query(self, user_query: str, topic_filter: Optional[str] = None) -> Dict[str, Any]:
        """
        Full RAG pipeline:  retrieve -> generate -> return.
        """
        hits = self.retrieve_context(user_query, topic_filter=topic_filter)
        answer = self.generate_response(user_query, hits)

        sources = self._extract_sources(hits)

        return {
            "answer": answer,
            "sources": sources,
            "chunks_used": len(hits),
        }

    # ────────────────────────────────────────────────────────
    #  Contextual query (plan-aware chatbot)
    # ────────────────────────────────────────────────────────

    def contextual_query(
        self,
        user_query: str,
        plan_context: dict,
        topic_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Chat query that also knows about the farmer's active plan.
        """
        soil_text = ""
        weather_text = ""
        if plan_context.get("soil_context") and isinstance(plan_context["soil_context"], dict):
            soil_text = plan_context["soil_context"].get("texture", "")
        if plan_context.get("weather_context") and isinstance(plan_context["weather_context"], dict):
            weather_text = plan_context["weather_context"].get("rainfall_trend", "")

        enriched_query = f"{user_query} (soil: {soil_text}, rainfall: {weather_text})"
        hits = self.retrieve_context(enriched_query, topic_filter=topic_filter, top_k=8)

        context_block = self._build_context_block(hits)

        prompt = CONTEXTUAL_PROMPT_TEMPLATE.format(
            system=SYSTEM_PROMPT,
            context_block=context_block,
            weather=json.dumps(plan_context.get("weather_context", {}), default=str),
            soil=json.dumps(plan_context.get("soil_context", {}), default=str),
            farm=json.dumps(plan_context.get("farm_details", {}), default=str),
            area=plan_context.get("land_area_ha", "N/A"),
            query=user_query,
        )

        logger.info(f"Contextual prompt to Ollama ({len(prompt)} chars)")
        response = call_ollama(prompt)

        if not response:
            response = (
                "I could not generate a personalised response at this time. "
                "Please ensure the AI service is running."
            )

        sources = self._extract_sources(hits)

        return {
            "answer": response.strip(),
            "sources": sources,
            "chunks_used": len(hits),
        }

    # ────────────────────────────────────────────────────────
    #  STRUCTURED Crop Plan Generation
    # ────────────────────────────────────────────────────────

    def generate_crop_plan(
        self,
        lat: float,
        lon: float,
        area_ha: float,
        weather_summary: dict,
        soil_summary: dict,
        farm_details: dict,
    ) -> Dict[str, Any]:
        """
        Generates a STRUCTURED crop plan as a parsed JSON dict.
        Falls back to a reasonable default structure if LLM fails or returns bad JSON.
        """
        # ── Extract all farmer inputs ──
        basic = farm_details.get("basicDetails", {}) if isinstance(farm_details, dict) else {}
        land_water = farm_details.get("landWater", {}) if isinstance(farm_details, dict) else {}
        inputs_pests = farm_details.get("inputsPests", {}) if isinstance(farm_details, dict) else {}
        market = farm_details.get("marketResources", {}) if isinstance(farm_details, dict) else {}

        farming_purpose = basic.get("farmingPurpose", "Mixed")
        experience = basic.get("experience", 0)
        land_ownership = basic.get("landOwnership", "Own")
        budget = basic.get("budget", "medium")
        labour = basic.get("labourAvailability", "medium")
        water_sources = ", ".join(basic.get("waterSources", [])) or "Not specified"
        previous_crops = ", ".join(inputs_pests.get("previousCrops", [])) or "None reported"
        pest_issues = ", ".join(inputs_pests.get("pestIssues", [])) or "None reported"
        fertilizers = ", ".join(inputs_pests.get("fertilizersUsed", [])) or "None reported"
        self_soil = land_water.get("soilType", "Not specified")
        irrigation_type = land_water.get("irrigationType", "Not specified")
        water_avail = land_water.get("waterAvailability", "Not specified")
        market_access = market.get("marketAccess", "Not specified")
        tractor = "Yes" if market.get("transportAvailable") else "No"
        storage = "Yes" if market.get("storageAvailable") else "No"

        # ── Extract API data ──
        w = weather_summary if isinstance(weather_summary, dict) else {}
        s = soil_summary if isinstance(soil_summary, dict) else {}

        def safe_val(val, source_name, suffix=""):
            if val is None or val == "N/A" or val == "unknown" or val == "":
                return f"Missing (Fallback to self-reported data)"
            return f"{val}{suffix} (Sourced strictly from {source_name})"

        soil_ph = safe_val(s.get("ph", "N/A"), "ISRIC API")
        soil_texture = safe_val(s.get("texture", "unknown"), "ISRIC API")
        soil_oc = safe_val(s.get("organic_carbon_pct", "N/A"), "ISRIC API", "%")
        soil_bd = safe_val(s.get("bulk_density", "N/A"), "ISRIC API", " kg/dm³")
        
        rainfall_trend = safe_val(w.get("rainfall_trend", "N/A"), "Open-Meteo API")
        past_rain = safe_val(w.get("past_rainfall_mm", "N/A"), "Open-Meteo API", "mm")
        forecast_rain = safe_val(w.get("forecast_rainfall_mm", "N/A"), "Open-Meteo API", "mm")
        avg_temp = safe_val(w.get("avg_temperature_c", "N/A"), "Open-Meteo API", "°C")
        humidity = safe_val(w.get("avg_humidity_pct", "N/A"), "Open-Meteo API", "%")
        solar_rad = safe_val(w.get("avg_solar_radiation_mj", "N/A"), "Open-Meteo API", " MJ/m²")

        # ── Fetch location name via Reverse Geocoding ──
        import requests
        location_name = "Unknown Location"
        if lat and lon:
            try:
                headers = {'User-Agent': 'KrishiPlanAI/1.0 (contact@example.com)'}
                url = f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}&format=json"
                resp = requests.get(url, headers=headers, timeout=3)
                if resp.status_code == 200:
                    loc_data = resp.json()
                    addr = loc_data.get('address', {})
                    city = addr.get('city', addr.get('town', addr.get('village', addr.get('county', ''))))
                    state = addr.get('state', '')
                    if city and state:
                        location_name = f"{city}, {state}"
                    elif state:
                        location_name = state
            except Exception as e:
                logger.warning(f"Reverse geocode failed: {e}")

        # ── Multi-query retrieval for best knowledge coverage ──
        base_texture = s.get("texture", "unknown") if s.get("texture") != "unknown" else self_soil
        base_ph = s.get("ph", "") if s.get("ph") else ""
        base_trend = w.get("rainfall_trend", "") if w.get("rainfall_trend") else ""
        
        queries = [
            f"Crop recommendations for {base_texture} soil pH {base_ph} with {base_trend} rainfall India {location_name}",
            f"Pest management {pest_issues} for crops in {base_texture} soil",
            f"Water management irrigation {irrigation_type} {water_avail} availability crop calendar",
            f"Budget farming {budget} cost crop rotation {previous_crops}",
        ]

        all_hits = []
        seen_ids = set()
        for q in queries:
            hits = self.retrieve_context(q, top_k=4)
            for h in hits:
                hid = str(h.get("doc_id", "")) + str(h.get("chunk_id", ""))
                if hid not in seen_ids:
                    seen_ids.add(hid)
                    all_hits.append(h)

        context_block = self._build_context_block(all_hits[:12])
        sources = self._extract_sources(all_hits[:12])

        # ── Build the structured prompt ──
        prompt = STRUCTURED_PLAN_PROMPT.format(
            context_block=context_block,
            location_name=location_name,
            lat=lat, lon=lon, area_ha=area_ha,
            farming_purpose=farming_purpose,
            experience=experience,
            land_ownership=land_ownership,
            budget=budget,
            labour=labour,
            water_sources=water_sources,
            irrigation_type=irrigation_type,
            water_availability=water_avail,
            self_soil=self_soil,
            previous_crops=previous_crops,
            pest_issues=pest_issues,
            fertilizers_used=fertilizers,
            market_access=market_access,
            tractor=tractor,
            storage=storage,
            soil_ph=soil_ph,
            soil_texture=soil_texture,
            soil_oc=soil_oc,
            soil_bd=soil_bd,
            rainfall_trend=rainfall_trend,
            past_rain=past_rain,
            forecast_rain=forecast_rain,
            avg_temp=avg_temp,
            humidity=humidity,
            solar_rad=solar_rad,
        )

        logger.info(f"Structured plan prompt: {len(prompt)} chars, {len(all_hits)} chunks")

        # ── Call LLM ──
        raw_response = call_ollama(prompt, require_json=True)

        # ── Parse JSON from response ──
        plan = self._parse_plan_json(raw_response)

        # ── Attach sources ──
        plan["rag_sources"] = sources

        return plan

    def _parse_plan_json(self, raw: str) -> Dict[str, Any]:
        """
        Attempt to parse JSON from LLM response.
        Handles common LLM quirks: markdown fences, trailing commas, preamble text.
        Falls back to a sensible default structure.
        """
        if not raw:
            logger.warning("Empty LLM response, using fallback plan")
            return self._fallback_plan("AI service did not respond")

        # Try 1: Direct parse
        import json_repair
        try:
            return json.loads(raw.strip())
        except json.JSONDecodeError:
            pass

        # Try 2: Extract JSON from markdown code fences
        json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', raw, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1).strip())
            except json.JSONDecodeError:
                pass

        # Try 3: Find first { ... } block
        brace_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if brace_match:
            try:
                return json.loads(brace_match.group(0))
            except json.JSONDecodeError:
                pass

        # Try 4: Use json-repair as a final resilient fallback
        try:
            repaired = json_repair.loads(raw)
            if repaired and isinstance(repaired, dict):
                logger.info("Successfully recovered JSON using json-repair module")
                return repaired
        except Exception as e:
            logger.warning(f"json-repair also failed: {e}")

        logger.warning("Failed to parse LLM JSON, using fallback plan")
        return self._fallback_plan("AI returned an unparseable response. Raw preview: " + raw[:200])

    def _fallback_plan(self, reason: str) -> Dict[str, Any]:
        """Sensible default plan structure when LLM fails."""
        return {
            "recommended_crops": [
                {
                    "name": "Wheat (general variety)",
                    "type": "primary",
                    "season": "Rabi (Nov-Apr)",
                    "reason": "Versatile cereal crop suitable for most Indian conditions",
                    "expected_yield": "35-45 quintals/ha",
                    "estimated_cost": "₹15,000-20,000/acre",
                    "water_requirement": "moderate (400-500mm)"
                },
                {
                    "name": "Moong Dal (Green Gram)",
                    "type": "rotation",
                    "season": "Zaid (Mar-Jun)",
                    "reason": "Short-duration legume that fixes nitrogen for next crop",
                    "expected_yield": "8-10 quintals/ha",
                    "estimated_cost": "₹8,000-10,000/acre",
                    "water_requirement": "low (250-300mm)"
                }
            ],
            "seasonal_calendar": [
                {"month": "Jun-Jul", "activity": "Land preparation, soil testing"},
                {"month": "Oct-Nov", "activity": "Rabi crop sowing"},
                {"month": "Jan-Feb", "activity": "Mid-season crop management"},
                {"month": "Mar-Apr", "activity": "Harvesting + Zaid preparation"},
                {"month": "Apr-Jun", "activity": "Short-duration crop / fallow"}
            ],
            "soil_management": {
                "current_status": "Detailed analysis requires AI service",
                "recommendations": [
                    "Get soil tested at nearest KVK or agriculture lab",
                    "Apply farmyard manure (FYM) at 5-6 tonnes/ha before sowing",
                    "Practice crop rotation to maintain soil health"
                ]
            },
            "water_management": {
                "strategy": "Schedule irrigation based on crop growth stages",
                "recommendations": [
                    "Irrigate at critical growth stages (CRI, flowering, grain filling)",
                    "Consider micro-irrigation to save water",
                    "Monitor soil moisture before irrigating"
                ]
            },
            "pest_management": {
                "risk_pests": ["General field pests"],
                "preventive_measures": [
                    "Seed treatment before sowing",
                    "Regular field scouting every 7-10 days",
                    "Use neem-based bio-pesticides as first line of defence"
                ]
            },
            "risk_factors": [
                {
                    "risk": "Weather variability",
                    "severity": "medium",
                    "mitigation": "Diversify crop selection and maintain contingency plans"
                }
            ],
            "budget_analysis": {
                "budget_tier": "To be determined",
                "estimated_input_cost": "₹15,000-25,000/acre",
                "estimated_revenue": "₹35,000-50,000/acre",
                "roi_estimate": "80-130%"
            },
            "summary": reason,
            "_fallback": True,
        }

    # ────────────────────────────────────────────────────────
    #  Helpers
    # ────────────────────────────────────────────────────────

    def _extract_sources(self, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deduplicate sources from hit list."""
        sources = []
        seen = set()
        for h in hits:
            key = h.get("document_name", "")
            if key and key not in seen:
                seen.add(key)
                sources.append({
                    "document": key,
                    "topic": h.get("primary_topic", ""),
                    "score": h.get("score", 0),
                    "pages": h.get("page_numbers", ""),
                })
        return sources

    # ────────────────────────────────────────────────────────
    #  Diagnostics
    # ────────────────────────────────────────────────────────

    @staticmethod
    def health_check() -> Dict[str, Any]:
        """Return vector store stats for the /rag/status/ endpoint."""
        return collection_stats()
