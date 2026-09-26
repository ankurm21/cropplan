from django.contrib.gis import admin
from .models import Land


@admin.register(Land)
class LandAdmin(admin.GISModelAdmin):
    list_display = ('id', 'user', 'area_ha', 'created_at')
    search_fields = ('user__email', 'user__username')
    list_filter = ('created_at',)
