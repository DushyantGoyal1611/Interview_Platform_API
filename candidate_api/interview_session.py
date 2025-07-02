from datetime import datetime

class InterviewSession:
    sessions = {}

    @classmethod
    def start(cls, email, candidate, invitation):
        cls.sessions[email] = {
            "candidate": candidate,
            "invitation": invitation,
            "asked_questions": [],
            "qa_pairs": [],
            "start_time": datetime.now()
        }
        
    @classmethod
    def get(cls, email):
        return cls.sessions.get(email)
    
    @classmethod
    def answer(cls, email, question, answer):
        cls.sessions[email]["asked_questions"].append(question)
        cls.sessions[email]["qa_pairs"].append({"Question": question, "Answer": answer})

    @classmethod
    def end(cls, email):
        return cls.sessions.pop(email, None)