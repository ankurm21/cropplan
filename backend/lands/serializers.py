from rest_framework import serializers
from django.contrib.gis.geos import Point, Polygon
from .models import Land


class LandSerializer(serializers.ModelSerializer):
    """
    Serializer for Land field records.
    Supports inputting lat/lon coordinates or GeoJSON geometry,
    and returns lat/lon plus boundary for frontend mapping.
    """
    lat = serializers.SerializerMethodField()
    lon = serializers.SerializerMethodField()
    latitude = serializers.SerializerMethodField()
    longitude = serializers.SerializerMethodField()
    boundary_coordinates = serializers.SerializerMethodField()
    polygon = serializers.SerializerMethodField()
    name = serializers.SerializerMethodField()
    location = serializers.SerializerMethodField()
    area = serializers.SerializerMethodField()
    areaHa = serializers.SerializerMethodField()

    # Write-only helper fields for easy creation
    latitude_in = serializers.FloatField(write_only=True, required=False)
    longitude_in = serializers.FloatField(write_only=True, required=False)
    polygon_in = serializers.ListField(
        child=serializers.ListField(child=serializers.FloatField(), min_length=2, max_length=2),
        write_only=True,
        required=False,
    )
    polygon_points = serializers.ListField(
        child=serializers.ListField(child=serializers.FloatField(), min_length=2, max_length=2),
        write_only=True,
        required=False,
    )

    class Meta:
        model = Land
        fields = [
            'id',
            'user',
            'lat',
            'lon',
            'latitude',
            'longitude',
            'latitude_in',
            'longitude_in',
            'area_ha',
            'areaHa',
            'area',
            'name',
            'location',
            'polygon',
            'boundary_coordinates',
            'polygon_in',
            'polygon_points',
            'created_at',
        ]
        read_only_fields = ['id', 'user', 'created_at']

    def get_lat(self, obj):
        return obj.location.y if obj.location else None

    def get_lon(self, obj):
        return obj.location.x if obj.location else None

    def get_latitude(self, obj):
        return obj.location.y if obj.location else None

    def get_longitude(self, obj):
        return obj.location.x if obj.location else None

    def get_areaHa(self, obj):
        return obj.area_ha

    def get_area(self, obj):
        if obj.area_ha:
            acres = round(obj.area_ha * 2.47105, 1)
            return f"{acres} Acres"
        return "1.0 Acre"

    def get_name(self, obj):
        return f"Field #{obj.id}"

    def get_location(self, obj):
        if obj.location:
            return f"{obj.location.y:.4f}° N, {obj.location.x:.4f}° E"
        return "Coordinates Set"

    def get_boundary_coordinates(self, obj):
        if obj.boundary:
            return list(obj.boundary.coords[0])
        return None

    def get_polygon(self, obj):
        if obj.boundary:
            # Leaflet expects [[lat, lon], ...]
            return [[coord[1], coord[0]] for coord in obj.boundary.coords[0]]
        return None

    def create(self, validated_data):
        # Extract lat & lon from initial data or validated data
        lat = (
            self.initial_data.get('latitude')
            or self.initial_data.get('lat')
            or validated_data.pop('latitude_in', None)
        )
        lon = (
            self.initial_data.get('longitude')
            or self.initial_data.get('lon')
            or validated_data.pop('longitude_in', None)
        )

        if lat is not None and lon is not None:
            validated_data['location'] = Point(float(lon), float(lat), srid=4326)

        # Handle polygon points
        raw_poly = (
            self.initial_data.get('polygon')
            or self.initial_data.get('polygon_points')
            or validated_data.pop('polygon_in', None)
            or validated_data.pop('polygon_points', None)
        )

        if raw_poly and len(raw_poly) >= 3:
            converted_poly = []
            for pt in raw_poly:
                p0, p1 = float(pt[0]), float(pt[1])
                # If first is lat (e.g. 22.7) and second is lon (75.8), convert to [lon, lat]
                if abs(p0) <= 38 and abs(p1) >= 60:
                    converted_poly.append([p1, p0])
                else:
                    converted_poly.append([p0, p1])

            # Ensure closed ring
            if converted_poly[0] != converted_poly[-1]:
                converted_poly.append(converted_poly[0])

            try:
                validated_data['boundary'] = Polygon(converted_poly, srid=4326)
            except Exception:
                pass

        # Handle area_ha
        area_ha = self.initial_data.get('area_ha') or self.initial_data.get('areaHa')
        if area_ha is not None:
            try:
                validated_data['area_ha'] = float(area_ha)
            except (ValueError, TypeError):
                pass

        return super().create(validated_data)
