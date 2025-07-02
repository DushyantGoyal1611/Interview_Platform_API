import os
import re
import json
import warnings
from dotenv import load_dotenv
from functools import lru_cache
# LangChain related libraries
# Gemini
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# SQL and ORM
import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'JOBMA_API.settings')  # change to your actual path
django.setup()
from interviewer_api.models import InterviewInvitation, InterviewDetail

# Code Starts from here --------------------------------------------->
warnings.filterwarnings('ignore')
load_dotenv()

# LLM
@lru_cache(maxsize=1)
def get_llm():
    try:
        return ChatGoogleGenerativeAI(
            model='gemini-2.0-flash',
            temperature=0.5,
            max_retries=3,
            request_timeout=30
        )
    except Exception as e:
        print(f"Failed to initialize LLM: {e}")
        return None

llm = get_llm()
if not llm:
    print("Critical error: Could not initialize LLM. Please try again later.")
    exit()

# Role-based Question-Generation
def generate_next_question(experience, target_role, asked_questions:set, max_retries=3):
    question_prompt = f"""
        You are an AI Interviewer conducting a structured interview for the role of "{target_role}".

        Candidate Profile:
        - Experience: {experience} years

        Instructions:
        - Generate one unique and relevant interview question tailored to the candidate’s experience and the role.
        - Use a mix of technical, behavioral, and situational styles if appropriate.
        - Do not repeat previous questions or themes.
        - Make sure the question is meaningful and precise.

        Only output the interview question. No explanations or extra text.
    """
    
    for attempt in range(max_retries):
        try:
            response = llm.invoke(question_prompt)
            question = response.content.strip() if hasattr(response, "content") else str(response)
            print(f"[Retry {attempt+1}] Question: {question}")
            
            if question and question not in asked_questions:
                # asked_questions.add(question)
                return question
        except Exception as e:
            print(f"Error generating question: {e}")
            continue
    return "All unique questions have been exhausted."

# Chatbot
def interview_chatbot(experience, question_limit, selected_role, invitation_id, candidate_id, skills):
    print("Interview Started! \nType 'exit' to quit.")

    # Chat History
    qna_history = [
        SystemMessage(content="You are an AI Interviewer. Ask one question at a time based on candidate's resume. After the interviewee answers, ask the next one. Keep it conversational.")
    ]

    counter = 0
    qa_pairs = []
    asked_questions = set()

    while counter < question_limit:
        # Generate and print next question
        response = generate_next_question(experience, selected_role, asked_questions)
        if response == "All unique questions have been exhausted.":
            print("AI: All unique questions have been exhausted.")
            break
        question = response.strip()
        print(f"AI: {question}")

        qna_history.append(AIMessage(content=question))

        # Wait for candidate's response
        interviewee_response = input("You: ")
        if interviewee_response.lower() == 'exit':
            print("Interview Ended. \nThank you!")
            break
        
        qna_history.append(HumanMessage(content=interviewee_response))
        qa_pairs.append({
            "Question": question,
            "Answer": interviewee_response
        })
        counter += 1

    generate_and_save_feedback(
        qa_pairs=qa_pairs,
        question_limit=question_limit,
        invitation_id=invitation_id,
        candidate_id=candidate_id,
        skills=skills
    )

# Feedback Generation
def generate_and_save_feedback(qa_pairs, question_limit, invitation_id, candidate_id, skills):
    feedback_prompt = f"""
        You are an AI Interview Assessor. Based on the following Q&A from an interview, provide:
        1. A score out of 10 for each answer.
        2. A brief feedback for each answer.
        3. An overall performance summary.
        4. A final recommendation: "Recommended", "Borderline", or "Not Recommended".
        5. Score Candidate got.
        6. Total Score

        Output in JSON format with:
        - "Feedback": List of {{"Skill", "Question", "Answer", "Score", "Comment", "Achieved Score"}}
        - "Summary": A short paragraph summarizing the candidate's overall performance.
        - "Recommendation": One of "Recommended", "Borderline", or "Not Recommended".

        Q&A:
        {json.dumps(qa_pairs, indent=2)}
    """

    def extract_json(text):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            json_match = re.search(r'\{[\s\S]+\}', text)
            if json_match:
                return json.loads(json_match.group(0))
            raise ValueError("Could not extract valid JSON.")
    
    # Step 1: Get feedback from LLM
    feedback_response = llm.invoke(feedback_prompt)
    try:
        feedback_data = extract_json(feedback_response.content)
    except Exception as e:
        print("Invalid JSON from Gemini:\n", feedback_response.content)
        feedback_data = {
            "Feedback": [],
            "Summary": "Feedback could not be generated.",
            "Recommendation": "Not Available"
        }

    summary = feedback_data.get("Summary", "No Summary Provided")
    recommendation = feedback_data.get("Recommendation", "No Recommendation Provided")

    achieved_score = sum(
        int(item.get("Achieved Score", 0))
        for item in feedback_data.get("Feedback", [])
        if isinstance(item, dict)
    )
    total_score = question_limit * 10

    # Prepare JSON strings
    skills_str = json.dumps(skills, ensure_ascii=False)

    # Separate questions and answers
    questions_list = [pair["Question"] for pair in qa_pairs]
    answers_list = [pair["Answer"] for pair in qa_pairs]

    questions_json = json.dumps(questions_list, ensure_ascii=False)
    answers_json = json.dumps(answers_list, ensure_ascii=False)
    feedback_json = json.dumps(feedback_data.get("Feedback", []), ensure_ascii=False)

    # Save to DB
    try:
        InterviewDetail.objects.create(
            invitation_id=invitation_id,
            candidate_id=candidate_id,
            questions=questions_json,
            answers=answers_json,
            achieved_score=achieved_score,
            total_score=total_score,
            feedback=feedback_json,
            summary=summary,
            recommendation=recommendation,
            skills=skills_str
        )
        InterviewInvitation.objects.filter(id=invitation_id).update(status="Completed")
        print("Feedback saved and interview marked as completed.")
    except Exception as e:
        print("Error saving feedback to database:", e)
        return {"error": str(e)}
    
    return {
        "summary": summary,
        "recommendation": recommendation,
        "feedback": feedback_data.get("Feedback", []),
        "achieved_score": achieved_score,
        "total_score": total_score
    }