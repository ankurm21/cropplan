from django.contrib.gis.db import models


class WeatherCache(models.Model):
    """
    Shared geographic cache for raw weather data fetched from external APIs (Open-Meteo).
    Stored with PostGIS PointField to allow 2km spatial radius reuse across multiple fields.
    No user_id and no land_id (shared environmental cache).
    """
    location = models.PointField(srid=4326, help_text="Geographic point (lon, lat) of weather cache")
    date = models.DateField()

    # Raw Weather Metrics
    precipitation = models.FloatField(null=True, blank=True)
    temp_max = models.FloatField(null=True, blank=True)
    temp_min = models.FloatField(null=True, blank=True)
    humidity = models.FloatField(null=True, blank=True) # relative_humidity_2m_mean
    radiation = models.FloatField(null=True, blank=True) # shortwave_radiation_sum

    # Audit
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = "Weather Cache"
        verbose_name_plural = "Weather Caches"
        unique_together = ('location', 'date')
        indexes = [
            models.Index(fields=['location', 'date']),
        ]

    def __str__(self):
        loc_str = f"({self.location.y:.4f}, {self.location.x:.4f})" if self.location else "No location"
        return f"Weather at {loc_str} on {self.date}"


class SoilCache(models.Model):
    """
    Shared geographic cache for soil attributes fetched from SoilGrids / ISRIC.
    Stored with PostGIS PointField to allow 2km spatial radius reuse across multiple fields.
    No user_id and no land_id (shared environmental cache).
    """
    location = models.PointField(srid=4326, help_text="Geographic point (lon, lat) of soil cache")

    ph = models.FloatField(null=True, blank=True, help_text="Soil pH (1:5 H2O suspension)")
    organic_carbon_pct = models.FloatField(null=True, blank=True, help_text="Soil Organic Carbon (mass %)")
    bulk_density = models.FloatField(null=True, blank=True, help_text="Bulk density of fine earth (g/cm³)")
    texture = models.CharField(max_length=20, help_text="USDA texture class (e.g., loamy, clayey, sandy)")

    # Granulometry / Physical Fractions
    clay_pct = models.FloatField(null=True, blank=True, help_text="Clay fraction percentage (<2 um)")
    sand_pct = models.FloatField(null=True, blank=True, help_text="Sand fraction percentage (50-2000 um)")
    silt_pct = models.FloatField(null=True, blank=True, help_text="Silt fraction percentage (2-50 um)")

    # SoilGrids Advanced Edaphic Features
    total_nitrogen = models.FloatField(null=True, blank=True, help_text="Total nitrogen (g/kg)")
    cec = models.FloatField(null=True, blank=True, help_text="Cation Exchange Capacity at pH 7 (cmol(+)/kg)")
    coarse_fragments_pct = models.FloatField(null=True, blank=True, help_text="Coarse fragments volume percentage (>2mm)")

    # Lab / Soil Health Card (SHC) Overrides
    salinity_ec = models.FloatField(null=True, blank=True, help_text="Electrical conductivity (dS/m)")
    available_p = models.FloatField(null=True, blank=True, help_text="Available phosphorus (kg/ha)")
    available_k = models.FloatField(null=True, blank=True, help_text="Available potassium (kg/ha)")
    zinc_ppm = models.FloatField(null=True, blank=True, help_text="Available zinc (ppm)")

    source = models.CharField(max_length=50, default="SoilGrids")
    fetched_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = "Soil Cache"
        verbose_name_plural = "Soil Caches"

    def __str__(self):
        loc_str = f"({self.location.y:.4f}, {self.location.x:.4f})" if self.location else "No location"
        return f"Soil at {loc_str} ({self.texture}, pH: {self.ph}) via {self.source}"
