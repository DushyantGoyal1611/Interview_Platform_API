from django.contrib import admin
from .models import Candidate, InterviewInvitation, InterviewDetail

# Register your models here.
admin.site.register(Candidate)
admin.site.register(InterviewInvitation)
admin.site.register(InterviewDetail)