from django.http import JsonResponse

def test_candidate(request):
    return JsonResponse({"message": "Candidate API is working!"})