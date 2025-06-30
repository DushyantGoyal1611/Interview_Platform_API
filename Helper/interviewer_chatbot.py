import os
import warnings
from dotenv import load_dotenv
from functools import lru_cache
from datetime import datetime, timedelta
from typing import Optional, Union, Literal

# LangChain & Gemini
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, EmailStr

# SQL and ORM
import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'JOBMA_API.settings')  # change to your actual path
django.setup()
from interviewer_api.models import Candidate, InterviewInvitation, InterviewDetail

# Code Starts from here --------------------------------------------->
warnings.filterwarnings('ignore')
load_dotenv()

# Input Schema Using Pydantic
    # For Interview Scheduling
class ScheduleInterviewInput(BaseModel):
    role:str = Field(description="Target Job Role")
    resume_path:str = Field(description="Path to resume file (PDF/DOCX/TXT)")
    question_limit:int = Field(description="Number of interview questions to generate")
    sender_email:EmailStr = Field(description="Sender's email address")

    # For Tracking Candidate
class TrackCandidateInput(BaseModel):
    name: Optional[str] = Field(None, description="Full name of the candidate")
    email: Optional[EmailStr] = Field(None, description="Email address of the candidate")
    role: Optional[str] = Field(None, description="Role applied for, e.g., 'frontend', 'backend'")
    date_filter: Optional[str] = Field(
        None,
        description="Optional date filter: 'today', 'recent', or 'last_week'"
    )
    status: Optional[Literal["Scheduled", "Completed"]] = None

# Model
@lru_cache(maxsize=1)
def get_llm():
    return ChatGoogleGenerativeAI(model='gemini-2.0-flash', temperature=0)

llm = get_llm()

# Document Loader
def extract_document(file_path):
    try:
        ext = os.path.splitext(file_path)[-1].lower()

        if ext == '.pdf':
            loader = PyPDFLoader(file_path)
        elif ext == '.docx':
            loader = UnstructuredWordDocumentLoader(file_path)
        elif ext == '.txt':
            loader = TextLoader(file_path, encoding="utf-8")
        else:
            raise ValueError("Unsupported file format. Please upload a PDF, DOCX or TXT file.")

        docs = loader.load()
        print(f'Loading {file_path} ......')
        print('Loading Successful')
        return docs

    except FileNotFoundError as fe:
        print(f"File not found: {fe}")
    except Exception as e:
        print(f"Error loading document: {e}")
    
    return []

# RAG Workflow
def create_rag_chain(doc, parser, score_threshold=1.0, resume_text=False):
    # Prompt
    prompt = PromptTemplate(
        template="""
    You are an intelligent assistant that only answers questions based on the provided document content.

    The document may include:
    - Headings, paragraphs, subheadings
    - Lists or bullet points
    - Tables or structured data
    - Text from PDF, DOCX, or TXT formats

    Your responsibilities:
    1. Use ONLY the content in the document to answer.
    2. If the user greets (e.g., "hi", "hello"), respond with a friendly greeting.
    3. Otherwise, provide a concise and accurate answer using only the document content.

    Document Content:
    {context}

    User Question:
    {question}

    Answer:
    """,
        input_variables=["context", "question"]
    )

    # Document Loader
    docs = extract_document(doc)
    if not docs:
        raise ValueError("Document could not be loaded.")
    # Text Splitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    # Model Embeddings and Vector Store
    embedding_model = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    vector_store = FAISS.from_documents(chunks, embedding_model)
    # Retriever for Confidence Score
    retriever = vector_store.similarity_search_with_score

    def retrieve_using_confidence(query):
        results = retriever(query)
        filtered = [doc for doc, score in results if score <= score_threshold]
        return filtered

    def format_docs(retrieved_docs):
        if not retrieved_docs:
            return "INSUFFICIENT CONTEXT"
        return "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    parallel_chain = RunnableParallel({
        'context': RunnableLambda(lambda q: retrieve_using_confidence(q)) | RunnableLambda(format_docs),
        'question': RunnablePassthrough()
    })

    main_chain = parallel_chain | prompt | llm | parser
    if resume_text:
        return main_chain, "\n\n".join([doc.page_content for doc in docs])
    return main_chain


# Function to Schedule Interview
def schedule_interview(role:str|dict, resume_path:str, question_limit:int, sender_email:str) -> str:
    # Skills and Experience fetching prompt and JSON Parser
    resume_parser = JsonOutputParser()

    # For Current Day
    current_month_year = datetime.now().strftime("%B %Y")

    resume_prompt = PromptTemplate(
        template="""
    You are an AI Resume Analyzer. Analyze the resume text below and extract **only** information relevant to the given job role.

    Your output **must** be in the following JSON format:
    {format_instruction}

    **Instructions:**
    1. **Name**:
    - Extract the candidate's full name from the **first few lines** of the resume.
    - It is usually the **first large bold text** or line that is **not an address, email, or phone number**.
    - Exclude words like "Resume", "Curriculum Vitae", "AI", or job titles.
    - If the name appears to be broken across lines, reconstruct it (e.g., "Abhis" and "hek" should be "Abhishek").
    - If no clear name is found, return: `"Name": "NA"`.

    2. **Skills**:
    - Extract technical and soft skills relevant to the **target role**.
    - Exclude generic or irrelevant skills (e.g., MS Word, Internet Browsing).
    - If **no skills are relevant**, return an empty list: `"Skills": []`.

    3. **Experience**:
    - Calculate the **cumulative time spent at each company** to get total professional experience.
    - Include only non-overlapping, clearly dated experiences (internships, jobs).
    - If a role ends in "Present" or "Current", treat it as ending in **{current_month_year}**.
    - Example: 
        - Google: Jan 2023 - Mar 2023 = 2 months  
        - Jobma: Feb 2025 - May 2025 = 3 months  
        - Total: 5 months = `"Experience": "0.42 years"`
    - Round the final answer to **2 decimal places**.
    - If durations are missing or unclear, return: `"Experience": "NA"`.

    4. Fetch email id from the document
    - Extract the first valid email address ending with `@gmail.com` from the text.
    - If not found, return `"Email": "NA"`.

    5. **Phone**:
    - Extract the first 10-digit Indian mobile number (starting with 6-9) from the resume.
    - You can allow formats with or without `+91`, spaces, or dashes.
    - Examples: `9876543210`, `+91-9876543210`, `+91 98765 43210`.
    - If no valid number is found, return `"Phone": "NA"`.

    6. **Education**:
    - Extract **highest qualification** (e.g., B.Tech, M.Tech, MCA, MBA, PhD).
    - Include the **degree name**, **specialization** (if available), and **university/institute name**.
    - Example: `"Education": "MCA in Computer Applications from VIPS, GGSIPU"`
    - If not found, return `"Education": "NA"`.
    ---

    **Target Role**: {role}

    **Resume Text**:
    {context}
    """,
        input_variables=["context", "role"],
        partial_variables={
            "format_instruction": resume_parser.get_format_instructions(),
            "current_month_year": current_month_year
        }
    )

    if not isinstance(resume_path, str):
        raise ValueError("resume_path must be a valid string")
    
    if not os.path.isfile(resume_path):
        raise FileNotFoundError(f"Resume file not found at path: {resume_path}")
    
    # Create the chain with JsonOutputParser instead of StrOutputParser
    resume_chain = resume_prompt | llm | resume_parser
    resume_result = resume_chain.invoke({'context': extract_document(resume_path), 'role': role})

    name = resume_result.get("Name", "NA")
    email = resume_result.get("Email", "NA")
    experience = resume_result.get("Experience", "NA")
    skills = ", ".join(resume_result.get("Skills", []))
    education = resume_result.get("Education", "NA")
    phone = resume_result.get("Phone", "NA")
    current_time = datetime.now()

    try:
        candidate = Candidate.objects.get(email=email)
        candidate.resume_path = resume_path
        candidate.save()
    except Candidate.DoesNotExist:
        candidate = Candidate.objects.create(
            name=name,
            email=email,
            skills=skills,
            education=education,
            experience=experience,
            resume_path=resume_path,
            phone=phone,
            created_at=current_time
        )


    InterviewInvitation.objects.create(
        candidate=candidate,
        role=role,
        question_limit=question_limit,
        sender_email=sender_email,
        status="Scheduled",
        created_at=current_time,
        interview_scheduling_time=current_time
    )

    return f"Interview scheduled for '{name}' for role: {role}"

# Function to Track Candidate's Details
def track_candidate(filter: TrackCandidateInput) -> Union[list[dict], str]: 
    """Flexible candidate tracker. Filter by name, email, role, date, and interview status."""
    try:
        queryset = InterviewInvitation.objects.select_related('candidate').prefetch_related('details')
        if filter.name:
                queryset = queryset.filter(candidate__name__icontains=filter.name.strip())

        if filter.email:
            queryset = queryset.filter(candidate__email=filter.email.strip().lower())

        if filter.role:
            queryset = queryset.filter(role__icontains=filter.role.strip())

        if filter.status:
            queryset = queryset.filter(status__iexact=filter.status.strip())

        if filter.date_filter:
            today = datetime.today()
            if filter.date_filter == "last_week":
                start = today - timedelta(days=today.weekday() + 7)
                end = start + timedelta(days=6)
            elif filter.date_filter == "recent":
                start = today - timedelta(days=3)
                end = today
            elif filter.date_filter == "today":
                start = today.replace(hour=0, minute=0, second=0, microsecond=0)
                end = today
            else:
                start = None

            if start:
                queryset = queryset.filter(interview_scheduling_time__range=(start, end))

        queryset = queryset.order_by("-candidate__created_at")

        if not queryset.exists():
            return "No matching candidate records found."

        result = []
        for invite in queryset:
            details = invite.details.first()  # Assuming OneToMany, get latest/first
            result.append({
                "candidate_id": invite.candidate.id,
                "name": invite.candidate.name,
                "email": invite.candidate.email,
                "phone": invite.candidate.phone,
                "role": invite.role,
                "sender_email": invite.sender_email,
                "status": invite.status,
                "interview_scheduling_time": invite.interview_scheduling_time,
                "achieved_score": getattr(details, 'achieved_score', None),
                "total_score": getattr(details, 'total_score', None),
                "summary": getattr(details, 'summary', None),
                "recommendation": getattr(details, 'recommendation', None),
                "skills": getattr(details, 'skills', None)
            })

        return result

    except Exception as e:
        return f"Error in tracking candidates: {str(e)}"
    
# To check available roles
def list_all_scheduled_roles() -> Union[list[str], str]:
    """Returns a list of all distinct roles for which interviews are scheduled."""
    roles = InterviewInvitation.objects.exclude(role__isnull=True).values_list('role', flat=True).distinct()
    return list(roles)

# Parsing part of Track Candidate
def extract_filters(user_input:str) -> dict:
    # Parsing Prompt
    parsing_prompt = PromptTemplate(
        template="""
    You are a helpful assistant that extracts filters to track a candidate's interview information.
    Based on the user's request, extract and return a JSON object with the following keys:

    - name: Candidate's name (if mentioned, like "Priya Sharma", "Dushyant Goyal")
    - email: Candidate's email (e.g., "abc@example.com", "SinghDeepanshu1233@gmail.com")
    - role: Role mentioned (like "backend", "frontend", "data analyst", "AI associate", etc.)
    - date_filter: One of: "today", "recent", "last_week", or null if not mentioned
    - status: "Scheduled" or "Completed" if mentioned (e.g., "show scheduled interviews" → "Scheduled")

    Special cases:
    - If user asks for "scheduled" or "upcoming" interviews, set status to "Scheduled"
    - If user asks for "completed" or "past" interviews, set status to "Completed"

    Only include relevant values. If a value is not mentioned, return null.

    Input: {input}
    Output:
    """,
        input_variables=["input"]
    )

    parsing_chain = parsing_prompt | llm | JsonOutputParser()
    parsing_result = parsing_chain.invoke({"input": user_input})

    return parsing_result

# Intent Classification
def intent_detect(user_input:str):
     # Intent Prompt
    intent_prompt = PromptTemplate(
        template="""You are an AI Intent Classifier for the Jobma Interviewing Platform. Based on the user input, identify their intent from the list of predefined intents.

    Possible Intents:
    - **schedule_interview**: The user wants to schedule an interview, usually mentions a role, resume, job title, or similar context.
    - **track_candidate**: The user wants to check or track candidate interview details. This may include:
    - Asking for a specific candidate's status using an email or name.
    - Requesting a summary, details or list of all candidates interviewed.
    - Asking how many interviews have been conducted or who has been interviewed.
    - Asking for scheduled interviews (phrases like "show scheduled", "track scheduled", "upcoming interviews")
    - Asking for completed interviews (phrases like "show completed", "past interviews")
    - **greet**: The user says hello, hi, good morning, or other greeting-like phrases.
    - **help**: The user is asking for help or support about using the Jobma platform.
    - **list_roles**: The user wants to view a list of roles interviews are scheduled for.
    - **bye**: The user says goodbye or ends the conversation.
    - **irrelevant**: The user input is unrelated to the Jobma platform or job interviews, such as asking about food, weather, sports, or general unrelated queries (e.g., "I want to make a pizza").

    Classify the following user input strictly as one of the intents above. Your response must be a **single word** from the list: `schedule_interview`, `track_candidate`, `list_roles`, `greet`, `help`, `bye`, or `irrelevant`.

    User Input:
    "{user_input}"

    Intent:
    """,
        input_variables=["user_input"]
    )

    intent_chain = intent_prompt | llm | StrOutputParser()
    intent = intent_chain.invoke({"user_input": user_input})
    return intent.strip()



# Chatbot using Intent
def ask_ai():

    # Tools   
    interview_tool = StructuredTool.from_function(
        func=schedule_interview,
        name='schedule_interview',
        description="Extracts resume information and schedules interview. Input should be a dictionary with keys: role, resume_path, question_limit, sender_email",
        args_schema=ScheduleInterviewInput
    )   

    status_tool = StructuredTool.from_function(
        func=list_all_scheduled_roles,
        name="list_all_scheduled_roles",
        description="Returns a list of all distinct roles for which interviews are scheduled."
    )

    memory = ConversationBufferMemory(k=20, memory_key="chat_history", return_messages=True)
    parser = StrOutputParser()
    rag_chain = create_rag_chain("formatted_QA.txt", parser)
    

    tools = [interview_tool, status_tool]
    # Agent
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False,
        memory=memory,
        handle_parsing_errors=True,
        max_iterations=3
    )

    while True:
        try:
            user_input = input("You: ")

            response = intent_detect(user_input)

            if response == 'bye':
                print(f"{response}: Exiting Chat \nGoodbye!")
                break
            elif response == 'schedule_interview':
                try:
                    print("AI: Let's schedule an interview. I'll need some details:")
                    role = input("Enter Target role: ").strip()
                    resume_path = input("Enter resume file path: ").strip()
                    if not os.path.exists(resume_path):
                        print(f"Error: File not found at {resume_path}")
                        continue
                    question_limit = int(input("How many questions to generate? "))
                    sender_email = input("Enter sender's email: ").strip()

                    response = agent.run(
                        f"Schedule an interview for {role} using resume at {resume_path}. "
                        f"Generate {question_limit} questions and send confirmation to {sender_email}."
                    )
                    print("AI (Agent):", response)
                except Exception as e:
                    print("Error during Scheduling:", str(e))
                continue
            elif response == 'track_candidate':
                filters = extract_filters(user_input)
                track_result = track_candidate(TrackCandidateInput(**filters))
                print(track_result)
                continue
            elif response == 'list_roles':
                agent_response = agent.invoke({'input': user_input})
                print(f"Agent({response}): {agent_response['output']}")
                continue
            elif response == 'greet':
                greet_reponse = llm.invoke(user_input)
                print(f"AI({response}):, {greet_reponse.content}")
                continue
            elif response == 'help':
                rag_response = rag_chain.invoke(user_input)
                print(f"Rag AI({response}): {rag_response}")
            else:
                print(f"{response}: Sorry! This Question is Irrelevant")
        except Exception as e:
            print(f"Error: {e}")

# ask_ai()
if __name__ == "__main__":
    ask_ai()