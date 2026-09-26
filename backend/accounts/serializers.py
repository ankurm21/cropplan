from rest_framework import serializers
from django.contrib.auth import get_user_model
from django.contrib.auth.password_validation import validate_password
from django.contrib.gis.geos import Point

User = get_user_model()

class UserSignupSerializer(serializers.ModelSerializer):
    """
    Serializer for handling user registration.
    Initializes a new User instance with hashed password and initial location.
    """
    password = serializers.CharField(write_only=True, required=True, validators=[validate_password])
    confirmPassword = serializers.CharField(write_only=True, required=True)
    lat = serializers.FloatField(write_only=True, required=False, allow_null=True)
    long = serializers.FloatField(write_only=True, required=False, allow_null=True)

    class Meta:
        model = User
        fields = ('first_name', 'last_name', 'email', 'password', 'confirmPassword', 'lat', 'long')

    def validate(self, attrs):
        if attrs.get('password') and attrs['password'] != attrs.get('confirmPassword'):
            raise serializers.ValidationError({"password": "Password fields didn't match."})
        return attrs

    def create(self, validated_data):
        validated_data.pop('confirmPassword')
        lat = validated_data.pop('lat', None)
        long = validated_data.pop('long', None)
        
        # Initialize home_location if coordinates provided
        home_location = None
        if lat is not None and long is not None:
            home_location = Point(long, lat, srid=4326) # Explicit SRID for safety

        email = validated_data.pop('email')
        password = validated_data.pop('password')

        user = User.objects.create_user(
            username=email, # Use email as username default
            email=email,
            password=password,
            home_location=home_location,
            **validated_data
        )
        return user

class UserLoginSerializer(serializers.Serializer):
    """
    Serializer for validating login credentials.
    Supports email identifiers.
    """
    identifier = serializers.CharField(required=True) # User email
    password = serializers.CharField(required=True, write_only=True)
    lat = serializers.FloatField(required=False, allow_null=True)
    long = serializers.FloatField(required=False, allow_null=True)

class UserProfileSerializer(serializers.ModelSerializer):
    """
    Minimal serializer for public profile data returned after auth.
    """
    lat = serializers.SerializerMethodField()
    long = serializers.SerializerMethodField()

    class Meta:
        model = User
        fields = ('id', 'first_name', 'last_name', 'email', 'lat', 'long', 'created_at')

    def get_lat(self, obj):
        return obj.home_location.y if obj.home_location else None

    def get_long(self, obj):
        return obj.home_location.x if obj.home_location else None
