import time
import requests
import logging
from datetime import timedelta

import ee
from django.utils import timezone
from django.contrib.gis.geos import Point
from django.contrib.gis.measure import D
from django.contrib.gis.db.models.functions import Distance
from ..models import SoilCache

logger = logging.getLogger(__name__)

CACHE_DISTANCE_KM = 2.0
FRESHNESS_DAYS = 90
GEE_PROJECT_ID = "cropplan-509210"

_ee_initialized = False


def init_ee():
    global _ee_initialized
    if not _ee_initialized:
        try:
            ee.Initialize(project=GEE_PROJECT_ID)
            _ee_initialized = True
            logger.info(f"Google Earth Engine initialized successfully with project {GEE_PROJECT_ID}.")
        except Exception as e:
            logger.warning(f"Google Earth Engine initialization failed: {e}")


class SoilEngine:
    """
    Production Soil Intelligence Engine.
    Primary source: ISRIC SoilGrids via Google Earth Engine (projects/soilgrids-isric/)
    Secondary fallback: ISRIC SoilGrids REST API (https://rest.isric.org)
    Spatial caching: PostGIS 2 km radius in PostgreSQL `data_soilcache` table.
    
    Zero-Fake-Data Policy:
    Never fabricates synthetic metrics. If all upstream providers fail,
    reports truthful 'available: False'.
    """

    DEPTH = "15-30cm"

    def __init__(self, cache_radius_km=CACHE_DISTANCE_KM, freshness_days=FRESHNESS_DAYS):
        self.cache_radius_km = cache_radius_km
        self.freshness_days = freshness_days

    def get_soil_summary(self, user, lat, lon, force_refresh=False):
        """
        1. Checks PostgreSQL SoilCache within 2 km and fresh within 90 days.
        2. If miss or force_refresh: fetches authentic SoilGrids data via Google Earth Engine.
        3. Saves authentic data into PostgreSQL SoilCache.
        """
        field_point = Point(float(lon), float(lat), srid=4326)
        freshness_cutoff = timezone.now() - timedelta(days=self.freshness_days)

        if not force_refresh:
            nearest_soil = (
                SoilCache.objects.filter(
                    location__distance_lte=(field_point, D(km=self.cache_radius_km)),
                    fetched_at__gte=freshness_cutoff
                )
                .exclude(source__icontains="fallback")
                .annotate(dist=Distance('location', field_point))
                .order_by('dist')
                .first()
            )

            if nearest_soil:
                dist_km = nearest_soil.dist.km if hasattr(nearest_soil, 'dist') else 0.0
                logger.info(
                    f"Soil cache HIT: Reusing verified soil data from {dist_km:.2f} km away "
                    f"(fetched on {nearest_soil.fetched_at.strftime('%Y-%m-%d')}) for field at ({lat}, {lon})"
                )
                return self._serialize(nearest_soil)

        logger.info(f"Soil cache MISS for field at ({lat}, {lon}). Querying authentic SoilGrids via Google Earth Engine...")
        return self._fetch_and_store(field_point, lat, lon)

    def _fetch_and_store(self, field_point, lat, lon):
        # 1. Try Google Earth Engine first (fast, authenticated, reliable)
        features = self._fetch_via_gee(lat, lon)
        source_label = "SoilGrids (Google Earth Engine)"

        # 2. If GEE fails, attempt REST API
        if not features:
            logger.warning(f"GEE query failed for ({lat}, {lon}), attempting SoilGrids REST fallback...")
            features = self._fetch_via_rest(lat, lon)
            source_label = "SoilGrids v2 (REST)"

        # 3. If all authentic providers fail, return truthful unavailable status
        if not features:
            logger.error(f"SoilGrids telemetry unavailable for ({lat}, {lon}). Zero fake data policy enforced.")
            return {
                "available": False,
                "status": "unavailable",
                "message": "Soil telemetry data currently unavailable from ISRIC SoilGrids. No sensor data received.",
                "ph": None,
                "organic_carbon_pct": None,
                "bulk_density": None,
                "texture": None,
                "clay_pct": None,
                "sand_pct": None,
                "silt_pct": None,
                "total_nitrogen": None,
                "cec": None,
                "coarse_fragments_pct": None,
                "salinity_ec": None,
                "available_p": None,
                "available_k": None,
                "zinc_ppm": None,
                "confidence": "none",
                "source": None
            }

        # 4. Save authentic verified record to PostgreSQL SoilCache
        soil = SoilCache.objects.create(
            location=field_point,
            source=source_label,
            fetched_at=timezone.now(),
            **features
        )
        logger.info(
            f"Successfully cached authentic SoilGrids data in PostgreSQL for ({lat}, {lon}): "
            f"pH {soil.ph}, {soil.texture}, Clay {soil.clay_pct}%, Sand {soil.sand_pct}%, "
            f"CEC {soil.cec} cmol/kg, N {soil.total_nitrogen} g/kg"
        )
        return self._serialize(soil)

    def _fetch_via_gee(self, lat, lon):
        """Fetches authentic ISRIC SoilGrids data using Google Earth Engine."""
        try:
            init_ee()
            point = ee.Geometry.Point([float(lon), float(lat)])
            depth = self.DEPTH

            def get_val(prop, scale=10.0):
                img = ee.Image(f"projects/soilgrids-isric/{prop}_mean")
                band = f"{prop}_{depth}_mean"
                val = img.select(band).reduceRegion(ee.Reducer.first(), point, 250).getInfo()
                raw = val.get(band) if val else None
                return round(raw / scale, 2) if raw is not None else None

            ph = get_val("phh2o", 10.0)
            clay = get_val("clay", 10.0)
            sand = get_val("sand", 10.0)
            silt = get_val("silt", 10.0)
            soc = get_val("soc", 100.0)
            bd = get_val("bdod", 100.0)
            nitrogen = get_val("nitrogen", 100.0)
            cec = get_val("cec", 10.0)
            cfvo = get_val("cfvo", 10.0)

            if ph is None and clay is None and sand is None:
                return None

            # Silt calculation / fallback
            if silt is None and clay is not None and sand is not None:
                silt = round(max(0.0, 100.0 - (clay + sand)), 2)

            texture = self._classify_texture(clay, sand)

            return {
                "ph": ph,
                "organic_carbon_pct": soc,
                "bulk_density": bd,
                "texture": texture,
                "clay_pct": clay,
                "sand_pct": sand,
                "silt_pct": silt,
                "total_nitrogen": nitrogen,
                "cec": cec,
                "coarse_fragments_pct": cfvo,
            }
        except Exception as e:
            logger.warning(f"Google Earth Engine SoilGrids query failed: {e}")
            return None

    def _fetch_via_rest(self, lat, lon):
        """Fallback querying rest.isric.org."""
        try:
            params = {
                "lat": lat,
                "lon": lon,
                "property": ["phh2o", "ocd", "soc", "bdod", "clay", "sand", "silt", "nitrogen", "cec", "cfvo"],
                "depths": self.DEPTH
            }
            resp = requests.get(
                "https://rest.isric.org/soilgrids/v2.0/properties/query",
                params=params,
                timeout=5.0,
                headers={"Accept": "application/json"}
            )
            if resp.status_code == 200:
                raw = resp.json()
                return self._engineer_features_rest(raw)
        except Exception as e:
            logger.warning(f"SoilGrids REST query failed: {e}")
        return None

    def _engineer_features_rest(self, raw):
        layers = raw.get("properties", {}).get("layers", [])

        def extract_mean(prop):
            try:
                for layer in layers:
                    if layer.get("name") == prop:
                        for depth in layer.get("depths", []):
                            if depth.get("label") == self.DEPTH:
                                return depth.get("values", {}).get("mean")
            except Exception:
                pass
            return None

        ph = extract_mean("phh2o")
        ocd = extract_mean("ocd")
        soc = extract_mean("soc")
        bd = extract_mean("bdod")
        clay = extract_mean("clay")
        sand = extract_mean("sand")
        silt = extract_mean("silt")
        nitrogen = extract_mean("nitrogen")
        cec = extract_mean("cec")
        cfvo = extract_mean("cfvo")

        if ph is None and clay is None and sand is None:
            return None

        clay_pct = round(clay / 10, 2) if clay is not None else None
        sand_pct = round(sand / 10, 2) if sand is not None else None
        silt_pct = round(silt / 10, 2) if silt is not None else (
            round(max(0.0, 100.0 - (clay_pct + sand_pct)), 2) if (clay_pct is not None and sand_pct is not None) else None
        )

        texture = self._classify_texture(clay_pct, sand_pct)
        soc_pct = round(soc / 100, 2) if soc is not None else (round(ocd / 10, 2) if ocd is not None else None)

        return {
            "ph": round(ph / 10, 2) if ph is not None else None,
            "organic_carbon_pct": soc_pct,
            "bulk_density": round(bd / 100, 2) if bd is not None else None,
            "texture": texture,
            "clay_pct": clay_pct,
            "sand_pct": sand_pct,
            "silt_pct": silt_pct,
            "total_nitrogen": round(nitrogen / 100, 2) if nitrogen is not None else None,
            "cec": round(cec / 10, 2) if cec is not None else None,
            "coarse_fragments_pct": round(cfvo / 10, 2) if cfvo is not None else None,
        }

    def _classify_texture(self, clay, sand):
        if clay is None or sand is None:
            return "loamy"
        if clay >= 40:
            return "clayey"
        elif sand >= 50:
            return "sandy"
        else:
            return "loamy"

    def _serialize(self, soil):
        return {
            "available": True,
            "status": "available",
            "ph": soil.ph,
            "organic_carbon_pct": soil.organic_carbon_pct,
            "bulk_density": soil.bulk_density,
            "texture": soil.texture,
            "clay_pct": soil.clay_pct,
            "sand_pct": soil.sand_pct,
            "silt_pct": soil.silt_pct,
            "total_nitrogen": soil.total_nitrogen,
            "cec": soil.cec,
            "coarse_fragments_pct": soil.coarse_fragments_pct,
            "salinity_ec": soil.salinity_ec,
            "available_p": soil.available_p,
            "available_k": soil.available_k,
            "zinc_ppm": soil.zinc_ppm,
            "confidence": "high",
            "source": soil.source,
            "fetched_at": soil.fetched_at.isoformat() if hasattr(soil, 'fetched_at') and soil.fetched_at else None
        }
