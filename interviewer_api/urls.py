from django.urls import path
from . import views

urlpatterns = [
    path('chat', views.AskAIView.as_view(), name='interviewer_chatbot')
]