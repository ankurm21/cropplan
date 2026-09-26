from django.contrib.gis import admin
from .models import WeatherCache, SoilCache


@admin.register(WeatherCache)
class WeatherCacheAdmin(admin.GISModelAdmin):
    list_display = ('id', 'location', 'date', 'temp_max', 'temp_min', 'precipitation')
    list_filter = ('date',)


@admin.register(SoilCache)
class SoilCacheAdmin(admin.GISModelAdmin):
    list_display = ('id', 'location', 'ph', 'texture', 'organic_carbon_pct', 'clay_pct', 'sand_pct', 'total_nitrogen', 'cec', 'source', 'fetched_at')
    list_filter = ('texture', 'source', 'fetched_at')
