import uuid
import time
import threading
from pathlib import Path
import subprocess
import traceback
from django.conf import settings
from .storage import Storage

# 复用现有行为与人脸模块（若权重缺失则回退占位）
try:
    from behavior_classification.get_pose import process_video as pose_process_video
except Exception:
    pose_process_video = None

try:
    from behavior_classification.predict import SkeletonPredictor as BC_SkeletonPredictor
except Exception:
    BC_SkeletonPredictor = None
try:
    from behavior_classification.real_time_detection import RealTimeActionDetector
except Exception:
    RealTimeActionDetector = None

try:
    from face_recognition.video_face_recognition import VideoFaceRecognition
except Exception:
    # 回退：直接将 face_recognition 目录加入 sys.path 并按脚本名导入
    try:
        import sys, importlib
        fr_dir = Path(settings.PROJECT_ROOT) / 'face_recognition'
        if str(fr_dir) not in sys.path:
            sys.path.insert(0, str(fr_dir))
        video_face_recognition_mod = importlib.import_module('video_face_recognition')
        VideoFaceRecognition = getattr(video_face_recognition_mod, 'VideoFaceRecognition', None)
    except Exception:
        VideoFaceRecognition = None


storage = Storage(Path(settings.BASE_DIR) / 'data')


def _safe_path(p: Path) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _find_action_weights() -> Path:
    """尝试在 checkpoints/yolopose 中查找行为分类模型权重，找不到则回退到 weights/stgcn.pth"""
    base = Path(settings.PROJECT_ROOT) / 'behavior_classification'
    candidates = [
        # 以项目根目录为基准的路径优先
        base / 'checkpoints' / 'yolopose' / 'spatial' / 'best_model.pth',
        base / 'checkpoints' / 'yolopose' / 'best.pth',
        base / 'checkpoints' / 'yolopose' / 'latest.pth',
        base / 'checkpoints' / 'yolopose' / 'model.pth',
        base / 'checkpoints' / 'yolopose' / 'spatial' / 'best.pth',
        base / 'checkpoints' / 'yolopose' / 'spatial' / 'latest.pth',
        base / 'checkpoints' / 'yolopose' / 'spatial' / 'model.pth',
        base / 'weights' / 'stgcn.pth',
        # 用户提供的绝对路径（兼容保留）
        Path('d:/Python_Project/YOLOv11-POSE-STGCN/behavior_classification/checkpoints/yolopose/spatial/best_model.pth'),
        Path('D:/Python_Project/YOLOv11-POSE-STGCN/behavior_classification/checkpoints/yolopose/spatial/best_model.pth'),
    ]
    for p in candidates:
        if p.exists():
            return p
    return base / 'weights' / 'stgcn.pth'


def start_processing_job(job_id: str, video_path: Path, students: list):
    """后台线程：执行行为检测与人脸识别，并更新状态"""
    def run():
        storage.set_job(job_id, {
            'status': 'processing',
            'progress': 0,
            'message': '正在准备模型与数据...'
        })

        processed_video = _safe_path(Path(settings.PROCESSED_DIR) / f'{job_id}.mp4')
        tmp_video = processed_video.with_name(processed_video.stem + '_tmp.mp4')
        behavior_stats = { 
            'Normal Listening': 0,
            'Raising Hand': 0,
            'Standing': 0,
            'Passing Objects': 0,
            'Turning Around': 0,
            'Sleeping': 0,
            'Looking Down': 0
        }
        attendance = []

        pipeline_mode = 'unknown'
        last_error_text = ''
        try:
            # 行为检测：优先使用实时检测器输出带标注视频
            storage.set_job(job_id, {
                'status': 'processing', 'progress': 10, 'message': '启动行为检测...'
            })

            def transcode_to_h264(src: Path, dst: Path):
                """使用FFmpeg转码到浏览器友好的H.264/AAC MP4，确保尺寸为偶数、快速启动、音视频兼容（Baseline+CFR）"""
                try:
                    # 关键点：
                    # -pix_fmt yuv420p 保证兼容；Baseline@3.1 对旧设备更友好
                    # scale=trunc(iw/2)*2:trunc(ih/2)*2 保证宽高是偶数（H.264要求）
                    # -r 30 强制 CFR，避免可变帧率造成播放问题
                    # -movflags +faststart 将 moov 移到文件前，提高网页首帧时间
                    # -map 选择第一路音视频，音频可选（若原视频无音轨也能成功）
                    # -vsync cfr 输出恒定帧率
                    cmd = [
                        'ffmpeg', '-y', '-i', str(src),
                        '-map', '0:v:0', '-map', '0:a:0?',
                        '-c:v', 'libx264', '-preset', 'veryfast',
                        '-profile:v', 'baseline', '-level', '3.1',
                        '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
                        '-pix_fmt', 'yuv420p',
                        '-r', '30',
                        '-x264-params', 'keyint=60:min-keyint=60:no-scenecut=1',
                        '-vsync', 'cfr',
                        '-movflags', '+faststart',
                        '-tag:v', 'avc1',
                        '-c:a', 'aac', '-b:a', '128k', '-ac', '2', '-ar', '48000',
                        str(dst)
                    ]
                    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                except subprocess.CalledProcessError as e:
                    # 记录ffmpeg错误并回退复制原始文件
                    try:
                        err = (e.stderr or b'').decode('utf-8', errors='ignore')
                        storage.set_job(job_id, {
                            'status': 'processing', 'progress': 38,
                            'message': f'转码失败，保留原视频：{e}\n{err[:500]}'
                        })
                    except Exception:
                        pass
                    try:
                        import shutil
                        shutil.copyfile(str(src), str(dst))
                    except Exception:
                        pass
                except FileNotFoundError as e:
                    # 未安装或未配置 FFmpeg：记录提示并复制原视频
                    try:
                        storage.set_job(job_id, {
                            'status': 'processing', 'progress': 36,
                            'message': '未检测到 FFmpeg，可执行文件未找到，已复制原视频（原编码可能不兼容浏览器）'
                        })
                    except Exception:
                        pass
                    try:
                        import shutil
                        shutil.copyfile(str(src), str(dst))
                    except Exception:
                        pass
                except Exception:
                    # 如果ffmpeg不可用或其他失败，保留原始文件以避免任务失败
                    try:
                        import shutil
                        shutil.copyfile(str(src), str(dst))
                    except Exception:
                        pass
                finally:
                    # 转码后清理临时源文件
                    try:
                        if src.exists():
                            src.unlink(missing_ok=True)
                    except Exception:
                        pass

            try:
                if RealTimeActionDetector is None:
                    raise RuntimeError('RealTimeActionDetector unavailable')
                detector = RealTimeActionDetector(
                    config_path=str(Path(settings.PROJECT_ROOT) / 'behavior_classification' / 'config.json'),
                    model_path=str(_find_action_weights()),
                    onnx_path=str(Path(settings.PROJECT_ROOT) / 'behavior_classification' / 'weights' / 'yolo11s-pose.onnx'),
                    window_size=50,
                    conf_threshold=0.4,
                )
                # 先输出到临时文件，再统一转码到H.264
                stats_rt = detector.run_video(str(video_path), str(tmp_video))
                if isinstance(stats_rt, dict):
                    # 将实时统计结果写入行为统计
                    for k in behavior_stats.keys():
                        behavior_stats[k] = int(stats_rt.get(k, 0))
                transcode_to_h264(tmp_video, processed_video)
                pipeline_mode = 'realtime'
            except Exception as e:
                # 回退1：仅做骨骼绘制并复制输出视频
                try:
                    if pose_process_video is None:
                        raise RuntimeError('pose_process_video unavailable')
                    # 如果可用，则创建预测器以在回退视频上叠加类别与置信度
                    predictor = None
                    weights_path = _find_action_weights()
                    config_path = Path(settings.PROJECT_ROOT) / 'behavior_classification' / 'config.json'
                    if BC_SkeletonPredictor and weights_path.exists() and config_path.exists():
                        predictor = BC_SkeletonPredictor(str(config_path), str(weights_path), device=None)
                    action_mapping_local = {
                        0: "Normal Listening", 1: "Normal Listening", 2: "Normal Listening", 8: "Normal Listening",
                        3: "Raising Hand", 4: "Raising Hand", 5: "Raising Hand", 6: "Raising Hand",
                        7: "Standing",
                        9: "Passing Objects", 10: "Passing Objects", 11: "Passing Objects", 12: "Passing Objects",
                        13: "Turning Around", 14: "Turning Around", 15: "Turning Around", 16: "Turning Around",
                        17: "Sleeping",
                        18: "Looking Down"
                    }
                    csv_out = _safe_path(Path(settings.PROCESSED_DIR) / f'{job_id}.csv')
                    pose_process_video(
                        model_path=str(Path(settings.PROJECT_ROOT) / 'behavior_classification' / 'weights' / 'yolo11s-pose.onnx'),
                        input_video_path=str(video_path),
                        output_csv_path=str(csv_out),
                        output_video_path=str(tmp_video),
                        conf_threshold=0.4,
                        predictor=predictor,
                        action_mapping=action_mapping_local,
                        window_size=50
                    )
                    transcode_to_h264(tmp_video, processed_video)
                    # 使用窗口滑动对CSV进行真实统计（课堂维度）
                    try:
                        if predictor is not None:
                            # 读取原CSV头以便重用
                            import csv as _csv
                            rows = []
                            with open(str(csv_out), 'r', encoding='utf-8') as f:
                                reader = _csv.reader(f)
                                header = next(reader, None)
                                for row in reader:
                                    rows.append(row)
                            # 创建复用的窗口文件
                            tmp_window = _safe_path(Path(settings.PROCESSED_DIR) / f'{job_id}_window.csv')
                            step = 5
                            win = 50
                            action_map_local = action_mapping_local
                            # 累计统计
                            for start in range(0, max(0, len(rows) - 1), step):
                                end = min(start + win, len(rows))
                                if end - start < 10:
                                    break
                                try:
                                    with open(str(tmp_window), 'w', encoding='utf-8', newline='') as wf:
                                        w = _csv.writer(wf)
                                        if header:
                                            w.writerow(header)
                                        for r in rows[start:end]:
                                            w.writerow(r)
                                    pred = predictor.predict(str(tmp_window))
                                    cls_name = pred.get('class_name') if isinstance(pred, dict) else None
                                    # 统一名称（预测器返回英文名称，与本地映射一致）
                                    if cls_name in behavior_stats:
                                        behavior_stats[cls_name] += 1
                                    else:
                                        # 若返回ID字符串映射
                                        cls_id = pred.get('class') if isinstance(pred, dict) else None
                                        if cls_id is not None:
                                            nm = action_map_local.get(int(cls_id), None)
                                            if nm and nm in behavior_stats:
                                                behavior_stats[nm] += 1
                                except Exception:
                                    continue
                    except Exception:
                        pass
                    pipeline_mode = 'fallback_pose'
                    storage.set_job(job_id, {
                        'status': 'processing', 'progress': 40,
                        'message': f'实时检测失败，已回退骨骼绘制：{e}\n{traceback.format_exc()}'
                    })
                    last_error_text = f'{e}'
                except Exception as e2:
                    # 回退2：直接复制原始视频到processed目录
                    import shutil
                    shutil.copyfile(str(video_path), str(tmp_video))
                    transcode_to_h264(tmp_video, processed_video)
                    pipeline_mode = 'copy_only'
                    storage.set_job(job_id, {
                        'status': 'processing', 'progress': 50,
                        'message': f'骨骼绘制失败，已直接复制原始视频：{e2}\n{traceback.format_exc()}'
                    })
                    last_error_text = f'{e2}'

            storage.set_job(job_id, {
                'status': 'processing', 'progress': 60, 'message': '进行人脸识别签到...'
            })

            # 人脸识别与签到：从视频中实际识别已注册人员
            recognized_names = set()
            try:
                if VideoFaceRecognition is None:
                    raise RuntimeError('VideoFaceRecognition unavailable')
                face_recog = VideoFaceRecognition(
                    arcface_model_path=str(Path(settings.PROJECT_ROOT) / 'face_recognition' / 'weight' / 'arcface_iresnet50.onnx'),
                    database_path=str(Path(settings.PROJECT_ROOT) / 'face_recognition' / 'face_database'),
                    recognition_threshold=0.4
                )
                import cv2
                cap = cv2.VideoCapture(str(video_path))
                if cap.isOpened():
                    fps = cap.get(cv2.CAP_PROP_FPS) or 25
                    max_seconds = 15
                    max_frames = int(fps * max_seconds)
                    frame_idx = 0
                    detected_faces_total = 0
                    recognized_faces_total = 0
                    max_similarity_observed = 0.0
                    while frame_idx < max_frames:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        # 每帧识别，尽量提高召回率
                        results = face_recog.process_frame(frame)
                        detected_faces_total += len(results)
                        recognized_faces_total += sum(1 for r in results if r.get('recognized'))
                        for r in results:
                            sim = float(r.get('similarity') or 0.0)
                            if sim > max_similarity_observed:
                                max_similarity_observed = sim
                            if r.get('recognized') and r.get('person_name'):
                                recognized_names.add(str(r['person_name']).strip())
                        # 若已全部识别到课程名单中的人员，则提前结束
                        if recognized_names and students:
                            if all(((stu.get('name') if isinstance(stu, dict) else str(stu)).strip() in recognized_names) for stu in students):
                                break
                        frame_idx += 1
                    cap.release()
                else:
                    storage.set_job(job_id, {
                        'status': 'processing', 'progress': 62,
                        'message': '人脸识别：无法打开视频，跳过签到统计'
                    })
            except Exception as e:
                storage.set_job(job_id, {
                    'status': 'processing', 'progress': 62,
                    'message': f'人脸识别失败：{e}'
                })

            # 统一名字格式，避免空格/大小写差异导致匹配失败
            normalized_recognized = {str(n).strip() for n in recognized_names}
            for stu in students:
                name = (stu.get('name') if isinstance(stu, dict) else str(stu))
                name = str(name).strip()
                attendance.append({
                    'name': name,
                    'status': 'Present' if name in normalized_recognized else 'Unknown'
                })

            final_msg = f'处理完成（{pipeline_mode}）'
            if pipeline_mode != 'realtime' and last_error_text:
                final_msg += f'：{last_error_text}'
            # 附加识别统计，便于排查为何显示 Unknown
            try:
                final_msg += f"；人脸检测/识别统计：检测到 {detected_faces_total} 张人脸，识别成功 {recognized_faces_total} 张，最高相似度 {max_similarity_observed:.2f}；识别到：{', '.join(sorted(normalized_recognized)) or '无'}"
            except Exception:
                pass
            storage.set_job(job_id, {
                'status': 'completed',
                'progress': 100,
                'message': final_msg,
                'result': {
                    'processed_video': f"/processed/videos/{job_id}.mp4",
                    'behavior_stats': behavior_stats,
                    'attendance': attendance,
                    'pipeline': pipeline_mode
                }
            })
        except Exception as e:
            storage.set_job(job_id, {
                'status': 'error',
                'progress': 100,
                'message': f'处理失败: {e}'
            })

    threading.Thread(target=run, daemon=True).start()


def create_job(video_file_path: Path, students: list) -> str:
    job_id = uuid.uuid4().hex
    storage.set_job(job_id, {
        'status': 'queued',
        'progress': 0,
        'message': '任务已创建'
    })
    start_processing_job(job_id, video_file_path, students)
    return job_id