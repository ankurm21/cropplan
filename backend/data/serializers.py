from rest_framework import serializers

class LandCentroidSerializer(serializers.Serializer):
    lat = serializers.FloatField()
    lon = serializers.FloatField()

class FarmDetailsBasicSerializer(serializers.Serializer):
    farmingPurpose = serializers.CharField(max_length=100)
    experience = serializers.IntegerField()
    landOwnership = serializers.CharField(max_length=50)
    rentPrice = serializers.CharField(max_length=50, required=False, allow_blank=True)
    labourAvailability = serializers.CharField(max_length=50)
    waterSources = serializers.ListField(child=serializers.CharField(max_length=50))
    budget = serializers.CharField(max_length=50)

class FarmDetailsSerializer(serializers.Serializer):
    basicDetails = FarmDetailsBasicSerializer()
    # Optional sections can be flexible DictFields
    landWater = serializers.DictField(required=False)
    inputsPests = serializers.DictField(required=False)
    marketResources = serializers.DictField(required=False)




class SoilSummarySerializer(serializers.Serializer):
    ph = serializers.FloatField(allow_null=True)
    organic_carbon_pct = serializers.FloatField(allow_null=True)
    bulk_density = serializers.FloatField(allow_null=True)
    texture = serializers.CharField()
    confidence = serializers.CharField()
    source = serializers.CharField()


class DataExtractSerializer(serializers.Serializer):
    land_centroid = LandCentroidSerializer()
    user_name = serializers.CharField(required=False)  # Can be derived from auth

    polygon_coords = serializers.ListField(
        child=serializers.DictField(child=serializers.FloatField()),
        required=False,
        allow_empty=True
    )

    land_area = serializers.FloatField(required=False, allow_null=True)
    details = FarmDetailsSerializer()

    # ✅ SYSTEM GENERATED (SOIL INTELLIGENCE)
    soil = SoilSummarySerializer(required=False, read_only=True)
