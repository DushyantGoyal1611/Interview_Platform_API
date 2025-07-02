import json
from rest_framework.response import Response
from rest_framework import status
from rest_framework.views import APIView
from Helper.candidate_interview import get_llm, generate_next_question, generate_and_save_feedback
from interviewer_api.models import Candidate, InterviewInvitation  
from .interview_session import InterviewSession

# Model
llm = get_llm()

# Get Scheduled Roles
class ScheduledRolesView(APIView):
    def post(self, request):
        email = request.data.get("email", "").strip().lower()
        if not email:
            return Response({"error": "Email is required"}, status=400)
        
        try:
            candidate = Candidate.objects.get(email__iexact=email)
        except Candidate.DoesNotExist:
            return Response({"error": "Candidate not found"}, status=404)
        
        scheduled = InterviewInvitation.objects.filter(candidate=candidate, status__iexact="Scheduled")
        if not scheduled.exists():
            return Response({"roles": []})
        
        roles = [{"role": i.role, "invitation_id": i.id} for i in scheduled]
        return Response({"roles": roles})

# Start Interview
class StartInterviewView(APIView):
    def post(self, request):
        email = request.data.get("email", "").strip().lower()
        invitation_id = request.query_params.get("invitation_id")

        if not invitation_id:
            return Response({"error": "Missing invitation_id in query params"}, status=400)

        try:
            candidate = Candidate.objects.get(email__iexact=email)
        except Candidate.DoesNotExist:
            return Response({"error": "Candidate not found"}, status=404)

        # Step 2: Get Interview Invitation
        try:
            invitation = InterviewInvitation.objects.get(
                id=invitation_id,
                candidate=candidate,
                status__iexact="Scheduled"
            )   
        except InterviewInvitation.DoesNotExist:
            return Response({"error": f"No scheduled interview found for invitation: '{invitation_id}'"}, status=404)

        InterviewSession.start(email, candidate, invitation)
        return self.generate_question(email)
    
    def generate_question(self, email):
        session = InterviewSession.get(email)
        if not session:
            return Response({"error": "Session not found"}, status=404)
        
        experience = session['candidate'].experience
        role = session['invitation'].role
        asked = set(session['asked_questions'])

        try:
            question = generate_next_question(experience, role, asked)
            if question and question not in asked:
                asked.add(question)
                session['asked_questions'] = list(asked)
                return Response({"question": question})
        except Exception as e:
            return Response({"error": str(e)}, status=500)
        
        return Response({"message": "All unique questions exhausted."})

# Submit Answer
class SubmitAnswerView(APIView):
    def post(self, request):
        print("Raw request body:", request.body)
        print("Parsed data:", request.data)
        email = request.data.get("email", "").strip().lower()
        question = request.data.get("question", "").strip()
        answer = request.data.get("answer", "").strip()

        session = InterviewSession.get(email)
        if not session:
            return Response({"error": "Session not found"}, status=404)
        
        # Save Q&A to session
        InterviewSession.answer(email,  question, answer)

        # Check if interview should continue
        if len(session["qa_pairs"]) < session["invitation"].question_limit:
            return StartInterviewView().generate_question(email)

        # Interview Done → generate feedback
        return self.complete_interview(email)

    def complete_interview(self, email):
        session = InterviewSession.end(email)
        qa_pairs = session["qa_pairs"]
        invitation = session["invitation"]
        candidate = session["candidate"]
        question_limit = invitation.question_limit
        try:
            skills = json.loads(candidate.skills) if candidate.skills else []
        except json.JSONDecodeError:
            skills = []

        try:
            result = generate_and_save_feedback(
                qa_pairs=qa_pairs,
                question_limit=question_limit,
                invitation_id=invitation.id,
                candidate_id=candidate.id,
                skills=skills
            )
        except Exception as e:
            return Response({"error": str(e)}, status=500)

        return Response({
            "message": "Interview completed",
            "summary": result.get("summary", ""),
            "recommendation": result.get("recommendation", ""),
            "feedback": result.get("feedback", []),
            "achieved_score": result.get("achieved_score"),
            "total_score": result.get("total_score")
        })