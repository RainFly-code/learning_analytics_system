from django.utils.deprecation import MiddlewareMixin
from django.http import HttpResponse

class SimpleCORSMiddleware(MiddlewareMixin):
    def process_response(self, request, response):
        response['Access-Control-Allow-Origin'] = '*'
        response['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        response['Access-Control-Allow-Headers'] = 'Content-Type'
        return response

    def process_request(self, request):
        if request.method == 'OPTIONS':
            return HttpResponse(status=204)
        return None