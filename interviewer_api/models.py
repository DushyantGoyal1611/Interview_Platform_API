from django.db import models

# Candidate Table
class Candidate(models.Model):
    name = models.CharField(max_length=100)
    email = models.EmailField(unique=True)
    skills = models.TextField(blank=True, null=True)
    education = models.TextField(blank=True, null=True)
    experience = models.CharField(max_length=50, blank=True, null=True)
    resume_path = models.CharField(max_length=225, blank=True, null=True)
    phone = models.CharField(max_length=15, blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    deleted_at = models.DateTimeField(blank=True, null=True)

    class Meta:
        db_table = 'candidates'

    def __str__(self):
        return self.name
    
# Interview Invitation Table
class InterviewInvitation(models.Model):    
    STATUS_CHOICES = [
        ("Scheduled", "Scheduled"),
        ("Completed", "Completed")
    ]

    candidate = models.ForeignKey(
        Candidate,
        on_delete=models.CASCADE,
        related_name="interview_invitation",
    )

    role = models.CharField(max_length=100)
    question_limit = models.PositiveIntegerField()
    sender_email = models.EmailField()
    status = models.CharField(max_length=50, choices=STATUS_CHOICES)
    created_at = models.DateTimeField(auto_now_add=True)
    deleted_at = models.DateTimeField(blank=True, null=True)
    interview_scheduling_time = models.DateTimeField()

    class Meta:
        db_table = 'interview_invitation'

    def __str__(self):
        return f"{self.role} - {self.status}"

# Interview Details Table
class InterviewDetail(models.Model):
    invitation = models.ForeignKey(
        InterviewInvitation,
        on_delete=models.CASCADE,
        related_name='details'
    )

    candidate = models.ForeignKey(
        Candidate,
        on_delete=models.CASCADE,
        related_name="interview_details"
    )

    questions = models.TextField(blank=True, null=True)
    answers = models.TextField(blank=True, null=True)
    achieved_score = models.PositiveIntegerField(blank=True, null=True)
    total_score = models.PositiveIntegerField(blank=True, null=True)
    feedback = models.TextField(blank=True, null=True)
    summary = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    deleted_at = models.DateTimeField(blank=True, null=True)
    recommendation = models.TextField(null=True, blank=True)
    skills = models.TextField(null=True, blank=True)

    class Meta:
        db_table = 'interview_details'

    def __str__(self):
        return f"Result: {self.achieved_score}/{self.total_score} for {self.candidate.name}"
