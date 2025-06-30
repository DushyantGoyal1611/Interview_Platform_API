from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/interviewer/', include('interviewer_api.urls')),
    path('api/candidate/', include('candidate_api.urls')),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)