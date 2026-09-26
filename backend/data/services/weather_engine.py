import requests
import logging
from datetime import date, timedelta
from statistics import mean
from django.contrib.gis.geos import Point
from django.contrib.gis.measure import D
from django.contrib.gis.db.models.functions import Distance
from ..models import WeatherCache

logger = logging.getLogger(__name__)

# Spatial threshold for reusing environmental cache (2 km)
CACHE_DISTANCE_KM = 2.0
PAST_DAYS = 15
FORECAST_DAYS = 16  # 1 current day + 15 predicted days (Open-Meteo max forecast limit)


class WeatherEngine:
    """
    Service for fetching, caching, and processing weather data.
    Uses PostGIS spatial queries with a 2 km distance radius threshold.
    Timeline: 15 days history + 1 current day + 15-16 days forecast.
    """

    BASE_URL = "https://api.open-meteo.com/v1/forecast"

    def __init__(self, cache_radius_km=CACHE_DISTANCE_KM, past_days=PAST_DAYS, forecast_days=FORECAST_DAYS):
        self.cache_radius_km = cache_radius_km
        self.past_days = past_days
        self.forecast_days = forecast_days

    def get_weather_summary(self, user, lat, lon):
        """
        Main entry point:
        1. Checks for a fresh cache record within 2 km radius.
        2. If miss or stale, fetches from Open-Meteo and caches with PostGIS Point geometry.
        3. Preprocesses and returns engineered features & aggregation summary.
        """
        field_point = Point(float(lon), float(lat), srid=4326)

        # 1. Sync / Retrieve weather rows (Cache hit or fresh API fetch)
        rows = self._sync_and_load_weather(field_point, lat, lon)

        # 2. Process
        cleaned = self._preprocess_weather(rows)
        if not cleaned:
            logger.warning(f"No valid weather data found for ({lat}, {lon})")
            return None

        # 3. Compute Features
        features = self._engineer_features(cleaned)
        aggregation = self._aggregate_weather(cleaned)

        summary = {
            **aggregation,
            **features,
            "confidence": "high" if len(cleaned) >= 28 else "medium",
            "data_points": len(cleaned)
        }

        return summary

    def _sync_and_load_weather(self, field_point, lat, lon):
        """
        Spatial caching check:
        Finds the nearest weather cache record within 2 km of field_point.
        Checks freshness: must contain a record for date.today().
        If valid: reuses cached weather rows.
        If missing or stale: calls Open-Meteo API and saves new records.
        """
        today = date.today()

        # Spatial search: nearest cache record within 2 km having data for today
        nearest_fresh_record = (
            WeatherCache.objects.filter(
                location__distance_lte=(field_point, D(km=self.cache_radius_km)),
                date=today
            )
            .annotate(dist=Distance('location', field_point))
            .order_by('dist')
            .first()
        )

        if nearest_fresh_record:
            dist_km = nearest_fresh_record.dist.km if hasattr(nearest_fresh_record, 'dist') else 0.0
            logger.info(
                f"Weather cache HIT: Reusing cached weather data from {dist_km:.2f} km away "
                f"for field at ({lat}, {lon})"
            )
            return WeatherCache.objects.filter(location=nearest_fresh_record.location).order_by('date')

        logger.info(
            f"Weather cache MISS/STALE for field at ({lat}, {lon}) (no fresh cache within {self.cache_radius_km} km). "
            f"Fetching from Open-Meteo API..."
        )
        self._fetch_and_store(field_point, lat, lon)
        return WeatherCache.objects.filter(location=field_point).order_by('date')

    def _fetch_and_store(self, field_point, lat, lon):
        weather_objects = []
        today = date.today()

        # 1. Fetch Historical Data (Past 15 days)
        hist_start = today - timedelta(days=self.past_days)
        hist_end = today - timedelta(days=1)

        hist_params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": hist_start.isoformat(),
            "end_date": hist_end.isoformat(),
            "daily": "precipitation_sum,temperature_2m_max,temperature_2m_min,relative_humidity_2m_mean,shortwave_radiation_sum",
            "timezone": "auto"
        }

        try:
            r_hist = requests.get("https://archive-api.open-meteo.com/v1/archive", params=hist_params, timeout=10)
            r_hist.raise_for_status()
            hist_data = r_hist.json().get("daily", {})
            self._parse_daily_data(hist_data, field_point, weather_objects)
        except Exception as e:
            logger.error(f"Historical weather fetch failed for ({lat}, {lon}): {e}")

        # 2. Fetch Forecast Data (1 current day + 15 predicted days = 16 forecast days)
        forecast_params = {
            "latitude": lat,
            "longitude": lon,
            "daily": "precipitation_sum,temperature_2m_max,temperature_2m_min,relative_humidity_2m_mean,shortwave_radiation_sum",
            "forecast_days": self.forecast_days,
            "timezone": "auto"
        }

        try:
            r_cast = requests.get("https://api.open-meteo.com/v1/forecast", params=forecast_params, timeout=10)
            r_cast.raise_for_status()
            cast_data = r_cast.json().get("daily", {})
            self._parse_daily_data(cast_data, field_point, weather_objects)
        except Exception as e:
            logger.error(f"Forecast weather fetch failed for ({lat}, {lon}): {e}")

        # 3. Store in WeatherCache
        if weather_objects:
            full_start = hist_start
            full_end = today + timedelta(days=self.forecast_days - 1)

            WeatherCache.objects.filter(
                location=field_point,
                date__range=[full_start, full_end]
            ).delete()

            WeatherCache.objects.bulk_create(weather_objects)
            logger.info(f"Cached {len(weather_objects)} weather days at PostGIS point for ({lat}, {lon})")
        else:
            logger.warning(f"No weather data collected to cache for ({lat}, {lon})")

    def _parse_daily_data(self, data, field_point, accumulator):
        """Helper to parse raw API response and append WeatherCache objects to list."""
        if not data:
            return

        time_list = data.get("time", [])
        for i, date_str in enumerate(time_list):
            try:
                precip = data["precipitation_sum"][i]
                t_max = data["temperature_2m_max"][i]
                t_min = data["temperature_2m_min"][i]
                humid = data["relative_humidity_2m_mean"][i]
                rad = data["shortwave_radiation_sum"][i]

                if date_str:
                    obj = WeatherCache(
                        location=field_point,
                        date=date_str,
                        precipitation=precip,
                        temp_max=t_max,
                        temp_min=t_min,
                        humidity=humid,
                        radiation=rad
                    )
                    accumulator.append(obj)
            except (IndexError, KeyError, ValueError):
                continue

    def _preprocess_weather(self, queryset):
        cleaned = []
        for row in queryset:
            if row.precipitation is not None and row.precipitation < 0:
                continue
            if row.temp_max is None or row.temp_min is None or row.temp_max < row.temp_min:
                continue
            if row.humidity is None or row.humidity < 0 or row.humidity > 100:
                continue
            if row.radiation is not None and row.radiation < 0:
                continue

            cleaned.append({
                "date": row.date,
                "rain": row.precipitation or 0.0,
                "tmax": row.temp_max,
                "tmin": row.temp_min,
                "humidity": row.humidity,
                "radiation": row.radiation or 0.0
            })
        return cleaned

    def _engineer_features(self, cleaned):
        if not cleaned:
            return {}

        daily_means = [(d["tmax"] + d["tmin"]) / 2 for d in cleaned]
        daily_ranges = [d["tmax"] - d["tmin"] for d in cleaned]

        features = {
            "avg_temperature_c": round(mean(daily_means), 2),
            "temperature_range_c": round(mean(daily_ranges), 2),
            "avg_humidity_pct": round(mean([d["humidity"] for d in cleaned]), 1),
            "avg_solar_radiation_mj": round(mean([d["radiation"] for d in cleaned]), 2)
        }
        return features

    def _aggregate_weather(self, cleaned):
        if not cleaned:
            return {}

        today = date.today()

        past = [d for d in cleaned if d["date"] < today]
        future = [d for d in cleaned if d["date"] >= today]

        past_rain = sum(d["rain"] for d in past)
        future_rain = sum(d["rain"] for d in future)

        delta = (future_rain - past_rain) / past_rain if past_rain > 0 else 0

        if delta > 0.1:
            trend = "increasing"
        elif delta < -0.1:
            trend = "decreasing"
        else:
            trend = "stable"

        return {
            "past_rainfall_mm": round(past_rain, 1),
            "forecast_rainfall_mm": round(future_rain, 1),
            "total_rainfall_mm": round(past_rain + future_rain, 1),
            "rainfall_trend": trend
        }
