from django.contrib.auth.models import AbstractUser
from django.contrib.gis.db import models
from django.utils.translation import gettext_lazy as _

class User(AbstractUser):
    """
    Custom User model for KrishiPlan AI.
    Includes support for phone authentication and location tracking.
    Now using GeoDjango PointField for location accuracy on PostGIS.
    """
    email = models.EmailField(_('email address'), unique=True)
    
    # Precise spatial location context
    home_location = models.PointField(srid=4326, null=True, blank=True, help_text="User's home or primary operation center")
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    # Use email instead of username for login if desired, 
    # but for simplicity and to match AbstractUser defaults we keep both.
    # To use email as primary:
    # USERNAME_FIELD = 'email'
    # REQUIRED_FIELDS = ['username', 'first_name', 'last_name']

    class Meta:
        verbose_name = _('user')
        verbose_name_plural = _('users')
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.email} ({self.first_name} {self.last_name})"
