# pages/urls.py

from django.urls import path
from res_view import views

urlpatterns = [
    path("", views.home, name='home'),
]