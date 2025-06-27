from django.http import JsonResponse

def test_interviewer(request):
    return JsonResponse({"message": "Interviewer API is working!"})