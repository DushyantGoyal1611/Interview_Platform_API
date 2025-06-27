from django.urls import path
from . import views

urlpatterns = [
    path("test/", views.test_candidate, name="test_candidate"),
]