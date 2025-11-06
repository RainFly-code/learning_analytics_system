import json
from pathlib import Path
from django.conf import settings
from django.http import JsonResponse, HttpResponseBadRequest
from django.views.decorators.csrf import csrf_exempt

from .tasks import create_job
from .storage import Storage

storage = Storage(Path(settings.BASE_DIR) / 'data')


@csrf_exempt
def upload_course(request):
    if request.method != 'POST':
        return HttpResponseBadRequest('Only POST supported')

    course_name = request.POST.get('course_name', '').strip()
    course_desc = request.POST.get('course_desc', '').strip()
    students_json = request.POST.get('students', '[]')
    try:
        students = json.loads(students_json)
    except Exception:
        students = []

    video = request.FILES.get('video')
    if not video:
        return HttpResponseBadRequest('Missing video')

    video_path = Path(settings.UPLOAD_DIR) / 'videos' / f"{course_name or 'course'}_{video.name}"
    video_path.parent.mkdir(parents=True, exist_ok=True)
    with open(video_path, 'wb') as f:
        for chunk in video.chunks():
            f.write(chunk)

    job_id = create_job(video_path, students)

    return JsonResponse({'job_id': job_id})


def job_status(request, job_id: str):
    job = storage.get_job(job_id)
    if not job:
        return JsonResponse({'error': 'job not found'}, status=404)
    return JsonResponse({'status': job.get('status'), 'progress': job.get('progress'), 'message': job.get('message')})


def job_result(request, job_id: str):
    job = storage.get_job(job_id)
    if not job:
        return JsonResponse({'error': 'job not found'}, status=404)
    if job.get('status') != 'completed':
        return JsonResponse({'error': 'job not completed'}, status=400)
    return JsonResponse(job.get('result', {}))