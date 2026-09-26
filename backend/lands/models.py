"""
models.py - Agricultural Land Database Schema

This module defines the data structure for farmer land parcels (fields).
Uses GeoDjango spatial fields for PostGIS GIS operations.
"""

from django.conf import settings
from django.contrib.gis.db import models


class Land(models.Model):
    """
    Land Model (Farmer Fields)

    Represents a farmer's individual agricultural field.
    A farmer (accounts_user) can have multiple fields (1:N relationship).
    """
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="lands",
        help_text="Farmer who owns this land parcel/field"
    )
    # Location: Center point coordinates (SRID 4326: WGS84 - lon, lat)
    location = models.PointField(srid=4326, help_text="Spatial point (longitude, latitude)")

    # Measurements: Calculated scale of the property
    area_ha = models.FloatField(null=True, blank=True, help_text="Total land area in hectares")

    # Spatial Data: Boundary definition stored as a GIS Polygon
    boundary = models.PolygonField(srid=4326, null=True, blank=True, help_text="GIS Polygon defining the land boundary")

    # Metadata: Auditing and temporal tracking
    created_at = models.DateTimeField(auto_now_add=True, help_text="Timestamp when the land record was initialized")

    class Meta:
        verbose_name = "Land Field"
        verbose_name_plural = "Land Fields"
        ordering = ["-created_at"]

    def __str__(self):
        loc_str = f"({self.location.y:.4f}, {self.location.x:.4f})" if self.location else "No location"
        return f"Field #{self.id} for {self.user.email} at {loc_str} - {self.area_ha or 'N/A'} ha"
