import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# 确保顶层项目目录加入PYTHONPATH，便于导入 behavior_classification / face_recognition
PROJECT_ROOT = BASE_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SECRET_KEY = 'demo-secret-key-do-not-use-in-prod'
DEBUG = True
ALLOWED_HOSTS = ['*']

INSTALLED_APPS = [
    'django.contrib.contenttypes',
    'django.contrib.staticfiles',
    'api',
]

MIDDLEWARE = [
    'django.middleware.common.CommonMiddleware',
    'api.middleware.SimpleCORSMiddleware',
]

ROOT_URLCONF = 'backend.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [],
        },
    },
]

WSGI_APPLICATION = 'backend.wsgi.application'

# 简化演示：不使用数据库模型，避免迁移复杂度
DATABASES = { 'default': { 'ENGINE': 'django.db.backends.sqlite3', 'NAME': BASE_DIR / 'db.sqlite3' } }

LANGUAGE_CODE = 'zh-hans'
TIME_ZONE = 'Asia/Shanghai'
USE_I18N = True
USE_TZ = True

STATIC_URL = '/static/'
STATIC_ROOT = BASE_DIR / 'static'

# 文件存储根目录（按需求放在 web/ 下）
UPLOAD_DIR = Path(__file__).resolve().parents[1] / 'upload' / 'videos'
PROCESSED_DIR = Path(__file__).resolve().parents[1] / 'processed' / 'videos'

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)