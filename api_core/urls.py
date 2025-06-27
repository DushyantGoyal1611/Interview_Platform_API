from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/interviewer/', include('interviewer_api.urls')),
    path('api/candidate/', include('candidate_api.urls')),
]
