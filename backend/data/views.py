from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated, AllowAny
from .serializers import DataExtractSerializer
import logging

logger = logging.getLogger(__name__)

class DataExtractView(APIView):
    """
    API Endpoint to receive farm details, land data, and optional information
    to generate an AI-powered STRUCTURED crop plan.

    Pipeline:
        1. Validate incoming data
        2. Fetch weather context (Open-Meteo)
        3. Fetch soil context (ISRIC SoilGrids)
        4. RAG-augmented STRUCTURED crop plan generation via Ollama
        5. Return structured response with 8 plan sections
    """
    permission_classes = [IsAuthenticated]

    def post(self, request):
        # 1. Validate Data
        serializer = DataExtractSerializer(data=request.data)
        if not serializer.is_valid():
            return Response({
                "status": "error",
                "message": "Invalid data format",
                "errors": serializer.errors
            }, status=status.HTTP_400_BAD_REQUEST)

        # 2. Extract Validated Data
        data = serializer.validated_data

        # User context
        user = request.user
        user_name = data.get('user_name') or user.username or user.first_name

        # 3. Process Data — Weather & Soil
        from .services.weather_engine import WeatherEngine
        from .services.soil_engine import SoilEngine

        weather_service = WeatherEngine()
        soil_service = SoilEngine()

        weather_summary = None
        soil_summary = None

        # Get coordinates from validated data
        centroid = data.get('land_centroid', {})
        lat = centroid.get('lat')
        lon = centroid.get('lon')

        if lat and lon:
            try:
                weather_summary = weather_service.get_weather_summary(user, lat, lon)
                logger.info(f"Weather summary generated for {user_name}")
            except Exception as e:
                logger.error(f"Weather engine failed: {e}")
                weather_summary = {"error": "Weather data unavailable"}

            try:
                soil_summary = soil_service.get_soil_summary(user, lat, lon)
                logger.info(f"Soil summary generated for {user_name}")
            except Exception as e:
                logger.error(f"Soil engine failed: {e}")
                soil_summary = {"error": "Soil data unavailable"}

        # 4. RAG-Augmented STRUCTURED Crop Plan Generation
        plan_data = self._generate_plan(
            lat=lat,
            lon=lon,
            area_ha=data.get('land_area'),
            weather_summary=weather_summary,
            soil_summary=soil_summary,
            farm_details=data.get('details', {}),
        )

        response_payload = {
            "status": "success",
            "message": "Crop plan generated successfully.",
            "data": {
                "user": user_name,
                "plan_id": "CP-" + str(user.id) + "-2024",
                # Structured plan sections
                "plan": plan_data,
                # Raw context data for the chatbot
                "weather_context": weather_summary,
                "soil_context": soil_summary,
                "land_area_ha": data.get('land_area'),
                "farm_details": data.get('details', {}),
            }
        }

        return Response(response_payload, status=status.HTTP_200_OK)

    def _generate_plan(self, lat, lon, area_ha, weather_summary, soil_summary, farm_details):
        """
        Try RAG-augmented structured generation first.
        Falls back to a default plan structure if RAG or Ollama is unavailable.
        """
        try:
            from rag.services.rag_engine import RAGEngine

            engine = RAGEngine()
            plan = engine.generate_crop_plan(
                lat=lat or 0,
                lon=lon or 0,
                area_ha=area_ha or 0,
                weather_summary=weather_summary or {},
                soil_summary=soil_summary or {},
                farm_details=farm_details or {},
            )
            if plan:
                return plan
        except Exception as e:
            logger.warning(f"RAG crop plan failed, using fallback: {e}")

        # Fallback — basic structured plan
        soil_text = ""
        weather_text = ""
        if isinstance(soil_summary, dict) and "texture" in soil_summary:
            soil_text = soil_summary["texture"]
        if isinstance(weather_summary, dict) and "rainfall_trend" in weather_summary:
            weather_text = weather_summary["rainfall_trend"]

        return {
            "recommended_crops": [
                {
                    "name": "Wheat",
                    "type": "primary",
                    "season": "Rabi (Nov-Apr)",
                    "reason": f"Suited for {soil_text or 'local'} soil with {weather_text or 'current'} rainfall",
                    "expected_yield": "35-45 quintals/ha",
                    "estimated_cost": "₹15,000-20,000/acre",
                    "water_requirement": "moderate"
                },
                {
                    "name": "Pulses (Moong)",
                    "type": "rotation",
                    "season": "Zaid (Mar-Jun)",
                    "reason": "Nitrogen-fixing rotation crop to improve soil fertility",
                    "expected_yield": "8-10 quintals/ha",
                    "estimated_cost": "₹8,000-10,000/acre",
                    "water_requirement": "low"
                }
            ],
            "seasonal_calendar": [
                {"month": "Jun-Jul", "activity": "Land preparation and soil testing"},
                {"month": "Oct-Nov", "activity": "Rabi crop sowing"},
                {"month": "Jan-Feb", "activity": "Mid-season management"},
                {"month": "Mar-Apr", "activity": "Harvesting"},
                {"month": "Apr-Jun", "activity": "Short-duration crop or fallow"},
            ],
            "soil_management": {
                "current_status": f"Soil type: {soil_text or 'Not determined'}",
                "recommendations": [
                    "Get soil tested at nearest KVK",
                    "Apply FYM at 5-6 tonnes/ha",
                    "Practice crop rotation",
                ]
            },
            "water_management": {
                "strategy": "Irrigate at critical growth stages",
                "recommendations": [
                    "Crown root initiation irrigation is critical",
                    "Monitor soil moisture regularly",
                    "Consider drip/sprinkler to save water",
                ]
            },
            "pest_management": {
                "risk_pests": ["General field pests"],
                "preventive_measures": [
                    "Seed treatment before sowing",
                    "Regular field scouting",
                    "Use bio-pesticides as first defence",
                ]
            },
            "risk_factors": [
                {
                    "risk": "Weather variability",
                    "severity": "medium",
                    "mitigation": "Diversify crops and maintain contingency plans"
                }
            ],
            "budget_analysis": {
                "budget_tier": "Medium",
                "estimated_input_cost": "₹15,000-25,000/acre",
                "estimated_revenue": "₹35,000-50,000/acre",
                "roi_estimate": "80-130%"
            },
            "summary": "Basic plan generated using available data. Enable the AI service for detailed, personalised recommendations.",
            "_fallback": True,
            "rag_sources": [],
        }


class TelemetryExtractView(APIView):
    """
    Independent environmental extraction API:
    GET /api/data/extract-telemetry/?lat=...&lon=...
    Extracts real-time weather and soil telemetry using the 2km PostGIS spatial cache.
    Does NOT trigger LLM / RAG generation. Fast (~50-100ms response).
    """
    permission_classes = [AllowAny]

    def get(self, request):
        lat_raw = request.GET.get('lat')
        lon_raw = request.GET.get('lon')

        if not lat_raw or not lon_raw:
            return Response(
                {"status": "error", "message": "Both 'lat' and 'lon' query parameters are required."},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            lat = float(lat_raw)
            lon = float(lon_raw)
        except ValueError:
            return Response(
                {"status": "error", "message": "Invalid coordinates format. 'lat' and 'lon' must be floats."},
                status=status.HTTP_400_BAD_REQUEST
            )

        user = request.user if request.user.is_authenticated else None

        from .services.weather_engine import WeatherEngine
        from .services.soil_engine import SoilEngine
        from .models import WeatherCache
        from datetime import date
        from django.contrib.gis.geos import Point
        from django.contrib.gis.measure import D
        from django.contrib.gis.db.models.functions import Distance

        force_refresh = request.query_params.get('refresh', 'false').lower() in ('true', '1')
        weather_service = WeatherEngine()
        soil_service = SoilEngine()

        # 1. Fetch / Cache Weather Data FIRST (2 km radius, 31-day history/forecast)
        try:
            weather_summary = weather_service.get_weather_summary(user, lat, lon) or {}

            # Extract today's live conditions from WeatherCache
            field_point = Point(float(lon), float(lat), srid=4326)
            today = date.today()
            today_record = (
                WeatherCache.objects.filter(
                    location__distance_lte=(field_point, D(km=2.0)),
                    date=today
                )
                .annotate(dist=Distance('location', field_point))
                .order_by('dist')
                .first()
            )

            if today_record:
                temp = round((today_record.temp_max + today_record.temp_min) / 2) if (today_record.temp_max is not None and today_record.temp_min is not None) else 28
                t_max = round(today_record.temp_max) if today_record.temp_max is not None else temp + 4
                t_min = round(today_record.temp_min) if today_record.temp_min is not None else temp - 4
                humidity = round(today_record.humidity) if today_record.humidity is not None else 60
                rain_mm = round(today_record.precipitation, 1) if today_record.precipitation is not None else 0.0
            else:
                temp = round(weather_summary.get("avg_temperature_c", 28))
                t_max = temp + 4
                t_min = temp - 4
                humidity = round(weather_summary.get("avg_humidity_pct", 60))
                rain_mm = round(weather_summary.get("forecast_rainfall_mm", 0.0) / 15, 1)

            rain_chance = min(95, max(5, int(rain_mm * 15)))
            condition = "Rain Showers" if rain_mm > 5.0 else ("Passing Showers" if rain_mm > 1.0 else ("Partly Cloudy" if humidity > 65 else "Clear Sky"))
            icon = "rainy" if rain_mm > 2.0 else ("partly-cloudy" if humidity > 50 else "sunny")
            feels_like = temp + (2 if humidity > 60 else 0)

            # Contextual agronomic advisory
            if rain_chance >= 60:
                advisory = "Rain expected soon. Delay fertilizer & chemical spray applications."
            elif temp >= 35:
                advisory = "High daytime temperatures. Schedule irrigation during early morning or evening."
            elif rain_mm > 0:
                advisory = f"Light precipitation ({rain_mm} mm) expected. Good soil moisture for field preparation."
            else:
                advisory = "Good conditions for field work today. No significant rain expected."

            weather_data = {
                "temperature": temp,
                "temp_max": t_max,
                "temp_min": t_min,
                "feelsLike": feels_like,
                "condition": condition,
                "icon": icon,
                "humidity": humidity,
                "rainChance": rain_chance,
                "windSpeed": 12,
                "advisory": advisory,
                "rainfall_trend": weather_summary.get("rainfall_trend", "stable"),
                "forecast_rainfall_mm": weather_summary.get("forecast_rainfall_mm", 0.0),
                "past_rainfall_mm": weather_summary.get("past_rainfall_mm", 0.0),
                "isLive": True
            }
        except Exception as e:
            logger.error(f"Weather extraction failed for ({lat}, {lon}): {e}")
            weather_data = {
                "temperature": 28,
                "feelsLike": 30,
                "condition": "Partly Cloudy",
                "icon": "partly-cloudy",
                "humidity": 60,
                "rainChance": 15,
                "windSpeed": 10,
                "advisory": "Good conditions for field work. Weather engine initializing.",
                "rainfall_trend": "stable",
                "isLive": False
            }

        # 2. Fetch / Cache Soil Data (2 km radius, 90 days validity, circuit breaker protection)
        try:
            soil_data = soil_service.get_soil_summary(user, lat, lon, force_refresh=force_refresh)
        except Exception as e:
            logger.error(f"Soil extraction failed for ({lat}, {lon}): {e}")
            soil_data = {
                "available": False,
                "status": "unavailable",
                "message": "Soil telemetry data currently unavailable from ISRIC SoilGrids.",
                "ph": None,
                "organic_carbon_pct": None,
                "bulk_density": None,
                "texture": None,
                "confidence": "none",
                "source": None
            }

        return Response({
            "status": "success",
            "message": "Environmental telemetry extracted successfully via PostGIS spatial cache.",
            "data": {
                "coordinates": {"lat": lat, "lon": lon},
                "weather": weather_data,
                "soil": soil_data
            }
        }, status=status.HTTP_200_OK)


# Backward-compatible alias
FieldDataExtractView = TelemetryExtractView

