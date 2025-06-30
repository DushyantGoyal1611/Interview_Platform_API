from rest_framework.decorators import api_view, parser_classes
from rest_framework.response import Response
from rest_framework import status
from Helper.interviewer_chatbot import schedule_interview, track_candidate, list_all_scheduled_roles, TrackCandidateInput, ScheduleInterviewInput
from pydantic import ValidationError
from rest_framework.parsers import MultiPartParser, FormParser
from django.core.files.storage import default_storage

# # Schedule Interview
# @api_view(["POST"])
# def schedule_interview_view(request):
#     try:
#         validated_data = ScheduleInterviewInput(**request.data)

#         resume = request.FILES.get('resume_path')
#         print(resume)
#         path = "../Uploaded_Resumes/"


#         result = schedule_interview(
#             validated_data.role,
#             # validated_data.resume_path,
#             validated_data.question_limit,
#             validated_data.sender_email
#         )
#         return Response({"message": result}, status=status.HTTP_200_OK)
#     except ValidationError as ve:
#         return Response({"error": ve.errors()}, status=status.HTTP_422_UNPROCESSABLE_ENTITY)

#     except Exception as e:
#         return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
    
    
# Schedule Interview
@api_view(["POST"])
@parser_classes([MultiPartParser, FormParser])
def schedule_interview_view(request):
    try:
        role = request.data.get("role", "NA")
        resume_file = request.FILES.get("resume")
        question_limit = int(request.data.get("question_limit", 5))
        sender_email = request.data.get("sender_email", "NA")

        if not all([role, question_limit, sender_email, resume_file]):
            return Response({"error": "All fields including file are required"}, status=status.HTTP_400_BAD_REQUEST)
        
        # Save uploaded file
        resume_path = default_storage.save(f"Uploaded_Resumes/{resume_file.name}", resume_file)
        resume_full_path = default_storage.path(resume_path)  # Absolute path for processing

        result = schedule_interview(role, resume_full_path, question_limit, sender_email)

        return Response({"message": result}, status=status.HTTP_200_OK)

    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)


# Track Candidate
@api_view(["POST"])
def track_candidate_view(request):
    try:
        filters = TrackCandidateInput(**request.data)
        result = track_candidate(filters)
        return Response({"data":result}, status=status.HTTP_200_OK)
    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
    
# List all Scheduled Interviews
@api_view(["GET"])
def list_scheduled_roles_view(request):
    try:
        roles = list_all_scheduled_roles()
        
        # If the function returns a string, it's likely an error message
        if isinstance(roles, str):
            return Response({"error": roles}, status=status.HTTP_404_NOT_FOUND)
        
        return Response({"roles": roles}, status=status.HTTP_200_OK)

    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)