from django.urls import path
from . import views

urlpatterns = [
    path('courses/upload', views.upload_course, name='upload_course'),
    path('jobs/<str:job_id>/status', views.job_status, name='job_status'),
    path('jobs/<str:job_id>/result', views.job_result, name='job_result'),
]