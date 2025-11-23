import json
from pathlib import Path
from django.conf import settings
import os
import urllib.request
import urllib.error
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


@csrf_exempt
def suggestions(request):
    if request.method != 'POST':
        return HttpResponseBadRequest('Only POST supported')
    try:
        body = request.body.decode('utf-8') if request.body else '{}'
        data = json.loads(body)
    except Exception:
        data = {}
    prompt = (
        '请根据下面提供的课堂原始统计数据，按照固定模板用中文输出建议，总长度≤300字；需结合课程名称与课程简介的整体特征，不涉及具体知识点：\n'
        '【课堂整体评价】\n'
        '（两到三句话评价课堂表现，结合出勤率、专注度、不良行为比例；可点名课程名称但不涉及具体知识点）\n'
        '【课堂亮点】\n'
        '- （亮点1，简洁但信息更丰富）\n'
        '- （亮点2，简洁但信息更丰富）\n'
        '【需改进的方面】\n'
        '- （问题1，简洁但信息更丰富）\n'
        '- （问题2，简洁但信息更丰富）\n'
        '【教学建议】\n'
        '1. （建议1，结合课程整体特征、可执行）\n'
        '2. （建议2，结合课程整体特征、可执行）\n'
        '3. （建议3，结合课程整体特征、可执行）\n'
        '【综合判断】\n'
        '（一句话或两句话收尾，保持概括性）\n'
        '要求：仅输出上述中文模板内容，不要JSON或解释；建议部分以 1、2、3 编号；避免涉及具体知识点，强调课程主题与整体特征。\n'
        '以下为原始统计数据：\n' +
        json.dumps(data, ensure_ascii=False)
    )

    def call_deepseek(p):
        key = getattr(settings, 'DEEPSEEK_API_KEY', None) or os.environ.get('DEEPSEEK_API_KEY') or os.environ.get('DEEPSEEK_KEY')
        base = getattr(settings, 'DEEPSEEK_API_BASE', None) or os.environ.get('DEEPSEEK_API_BASE') or 'https://api.deepseek.com'
        model = getattr(settings, 'DEEPSEEK_MODEL', None) or os.environ.get('DEEPSEEK_MODEL') or 'deepseek-chat'
        if not key:
            return None
        url = (base.rstrip('/') + '/v1/chat/completions').encode('utf-8').decode('utf-8')
        payload = {
            'model': model,
            'messages': [
                {'role': 'system', 'content': '你是教学质量分析助手，请用中文输出，整体长度≤300字，遵循用户模板与编号规则。'},
                {'role': 'user', 'content': p}
            ],
            'temperature': 0.7
        }
        try:
            req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'), headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=20) as resp:
                raw = resp.read().decode('utf-8', errors='ignore')
                j = json.loads(raw)
                c = j.get('choices') or []
                if c:
                    m = c[0].get('message') or {}
                    return str(m.get('content') or '').strip()
        except Exception:
            return None
        return None

    llm_text = call_deepseek(prompt)
    if not llm_text:
        return JsonResponse({'suggestions': '生成失败'})
    return JsonResponse({'suggestions': llm_text})
