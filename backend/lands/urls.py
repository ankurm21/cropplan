"""
urls.py - URL Routing for Lands App

Defines the API endpoints specifically for land and user-land interactions.
"""

from django.urls import path
from . import views

urlpatterns = [
    path('location/', views.HandleUser.as_view(), name='location'),
    path('fields/', views.LandListCreateView.as_view(), name='land-list-create'),
    path('fields/<int:pk>/', views.LandDetailView.as_view(), name='land-detail'),
]
