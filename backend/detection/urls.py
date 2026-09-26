"""
urls.py - URL Routing for Detection App

Provides endpoints for automated spatial analysis and parcel discovery.
"""

from django.urls import path
from . import views

urlpatterns = [
    # Primary endpoint for triggering boundary detection algorithms
    path('parcels/', views.ParcelDetectionView.as_view(), name='detect-parcels'),
]
