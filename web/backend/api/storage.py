import json
import threading
from pathlib import Path


class Storage:
    """轻量级JSON存储，用于演示不依赖数据库的任务状态保存"""
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.jobs_file = self.base_dir / 'jobs.json'
        self.lock = threading.Lock()
        if not self.jobs_file.exists():
            self.jobs_file.write_text(json.dumps({}), encoding='utf-8')

    def _load(self):
        try:
            return json.loads(self.jobs_file.read_text(encoding='utf-8'))
        except Exception:
            return {}

    def _save(self, data):
        self.jobs_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')

    def set_job(self, job_id: str, payload: dict):
        with self.lock:
            data = self._load()
            data[job_id] = payload
            self._save(data)

    def get_job(self, job_id: str):
        data = self._load()
        return data.get(job_id)