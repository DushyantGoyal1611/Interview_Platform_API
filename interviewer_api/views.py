from rest_framework.response import Response
from rest_framework import status
from rest_framework.views import APIView
from Helper.interviewer_chatbot import get_llm, schedule_interview, create_rag_chain, extract_filters, track_candidate, intent_detect, list_all_scheduled_roles, TrackCandidateInput
from rest_framework.parsers import MultiPartParser, FormParser
from django.core.files.storage import default_storage    
from langchain_core.output_parsers import StrOutputParser
    

# Model
llm = get_llm()
# Parser
parser = StrOutputParser()

# View of Chatbot
class AskAIView(APIView):
    parser_classes = [MultiPartParser, FormParser]  # Enable file uploads
    def post(self, request):
        try:
            user_input = request.data.get('user_message', '')
            if not user_input:
                return Response({'error': 'No input provided'}, status=status.HTTP_400_BAD_REQUEST)
            
            intent = intent_detect(user_input)

            if intent == 'greet':
                greet_reponse = llm.invoke(user_input)
                return Response({'response' : greet_reponse})
            
            elif intent == 'bye':
                bye_reponse = llm.invoke(user_input)
                return Response({'response' : bye_reponse})
            
            elif intent == 'help':
                rag_chain = create_rag_chain("Necessary_Documents/formatted_QA.txt", parser)
                rag_response = rag_chain.invoke(user_input)
                return Response({'response': rag_response})
            
            elif intent == 'list_roles':
                roles = list_all_scheduled_roles()
                return Response({'response': roles})
            
            elif intent == 'track_candidate':
                filters = extract_filters(user_input)
                result = track_candidate(TrackCandidateInput(**filters))
                return Response({"response":result})
            
            elif intent == 'schedule_interview':
                role = request.data.get('role', 'NA')
                resume_file = request.FILES.get('resume')
                question_limit = int(request.data.get('question_limit', 5))
                sender_email = request.data.get('sender_email', 'NA')

                if not all([role, resume_file, sender_email]):
                    return Response({"error": "All fields including file are required"}, status=status.HTTP_400_BAD_REQUEST)
                
                resume_path = default_storage.save(resume_file.name, resume_file)
                resume_full_path = default_storage.path(resume_path)

                result = schedule_interview(role, resume_full_path, question_limit, sender_email)
                return Response({"response": result})
            else:
                return Response({'response': "I'm sorry, I can only help with interview-related queries."})
            
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


