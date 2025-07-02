from django.urls import path
from . import views

urlpatterns = [
     path('scheduled-roles/', views.ScheduledRolesView.as_view(), name='scheduled_roles'),
     path('start-interview/', views.StartInterviewView.as_view(), name='start_interview'),
     path('submit-answer/', views.SubmitAnswerView.as_view(), name='submit_answer'),
]