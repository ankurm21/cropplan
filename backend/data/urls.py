from django.urls import path
from .views import DataExtractView, TelemetryExtractView

urlpatterns = [
    path('extract/', DataExtractView.as_view(), name='data-extract'),
    path('extract-telemetry/', TelemetryExtractView.as_view(), name='extract-telemetry'),
    path('extract-field-data/', TelemetryExtractView.as_view(), name='extract-field-data'),
]

