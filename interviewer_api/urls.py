from django.urls import path
from . import views

urlpatterns = [
    path("schedule/", views.schedule_interview_view),
    path("track/", views.track_candidate_view),
    path("scheduled-roles/", views.list_scheduled_roles_view),
]