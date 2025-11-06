from django.urls import path, include
from . import views

urlpatterns = [
    path('api/', include('api.urls')),
    path('', views.index, name='index'),
    path('processed/videos/<path:path>', views.serve_processed, name='serve_processed'),
]