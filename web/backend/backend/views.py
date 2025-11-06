from pathlib import Path
from django.conf import settings
from django.shortcuts import render
from django.http import HttpResponse
from django.views.static import serve as static_serve


def index(request):
    return render(request, 'index.html')


def serve_processed(request, path):
    # 仅用于演示环境，生产建议由Nginx等直接静态托管
    root = Path(settings.PROCESSED_DIR)
    return static_serve(request, path, document_root=str(root))