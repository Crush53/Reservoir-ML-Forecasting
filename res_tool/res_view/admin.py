from django.contrib import admin
from .models import ResMeta, ResData, WeatherData  # Import your models

# Register your models here.
admin.site.register(ResMeta)
admin.site.register(ResData)
admin.site.register(WeatherData)
