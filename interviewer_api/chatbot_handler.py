from Helper.interviewer_chatbot import schedule_interview, track_candidate, intent_detect, list_all_scheduled_roles

def handle_chat_message(user_input: str) -> str:
    """
        Main chatbot handler: routes user input to correct function.
    """
    intent = intent_detect(user_input)

    if intent == "schedule_interview":
        return schedule_interview()
    elif intent == "track_candidate":
        return track_candidate()
    elif intent == "list_roles":
        return list_all_scheduled_roles()