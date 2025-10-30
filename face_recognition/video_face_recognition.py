import cv2
import numpy as np
import argparse
import time
import os
from typing import List, Dict, Optional
from face_detector import FaceRecognitionSystem
from face_database import FaceDatabase


class VideoFaceRecognition:
    """
    视频人脸识别系统
    """
    
    def __init__(self, arcface_model_path: str, database_path: str = "face_database",
                 confidence_threshold: float = 0.5, recognition_threshold: float = 0.6):
        """
        初始化视频人脸识别系统
        
        参数:
            arcface_model_path: ArcFace模型路径
            database_path: 人脸数据库路径
            confidence_threshold: 人脸检测置信度阈值
            recognition_threshold: 人脸识别相似度阈值
        """
        self.face_system = FaceRecognitionSystem(arcface_model_path, confidence_threshold)
        self.face_db = FaceDatabase(database_path)
        self.recognition_threshold = recognition_threshold
        
        # 性能统计
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0.0
        
        # 识别结果缓存
        self.recognition_cache = {}
        self.cache_timeout = 1.0  # 缓存超时时间（秒）
    
    def update_fps(self):
        """
        更新FPS统计
        """
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def process_frame(self, frame: np.ndarray) -> List[Dict]:
        """
        处理单帧图像，进行人脸识别
        
        参数:
            frame: 输入帧
            
        返回:
            识别结果列表
        """
        results = []
        
        # 检测人脸并提取特征
        face_results = self.face_system.process_image(frame)
        
        for face_result in face_results:
            bbox = face_result['bbox']
            features = face_result['features']
            
            # 人脸识别
            recognition_result = self.face_db.recognize_face(
                features, self.recognition_threshold
            )
            
            result = {
                'bbox': bbox,
                'features': features,
                'face_roi': face_result['face_roi']
            }
            
            if recognition_result:
                person_id, person_name, similarity = recognition_result
                result.update({
                    'person_id': person_id,
                    'person_name': person_name,
                    'similarity': similarity,
                    'recognized': True
                })
            else:
                result.update({
                    'person_id': None,
                    'person_name': 'Unknown',
                    'similarity': 0.0,
                    'recognized': False
                })
            
            results.append(result)
        
        return results
    
    def draw_results(self, frame: np.ndarray, results: List[Dict]) -> np.ndarray:
        """
        在帧上绘制识别结果
        
        参数:
            frame: 输入帧
            results: 识别结果列表
            
        返回:
            绘制结果的帧
        """
        result_frame = frame.copy()
        
        for result in results:
            bbox = result['bbox']
            x, y, w, h = bbox
            
            # 确定颜色和标签
            if result['recognized']:
                color = (0, 255, 0)  # 绿色 - 已识别
                label = f"{result['person_name']}"
                confidence_text = f"Similarity: {result['similarity']:.2f}"
            else:
                color = (0, 0, 255)  # 红色 - 未识别
                label = "Unknown"
                confidence_text = "Not recognized"
            
            # 绘制人脸框
            cv2.rectangle(result_frame, (x, y), (x + w, y + h), color, 2)
            
            # 准备文本
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            
            # 计算文本尺寸
            (text_width1, text_height1), _ = cv2.getTextSize(label, font, font_scale, thickness)
            (text_width2, text_height2), _ = cv2.getTextSize(confidence_text, font, font_scale, thickness)
            
            # 背景框尺寸
            bg_width = max(text_width1, text_width2) + 10
            bg_height = text_height1 + text_height2 + 15
            
            # 确保文本框在图像范围内
            text_x = max(0, min(x, result_frame.shape[1] - bg_width))
            text_y = max(bg_height, y)
            
            # 绘制背景框
            cv2.rectangle(result_frame, 
                         (text_x, text_y - bg_height), 
                         (text_x + bg_width, text_y), 
                         (0, 0, 0), -1)
            cv2.rectangle(result_frame, 
                         (text_x, text_y - bg_height), 
                         (text_x + bg_width, text_y), 
                         color, 2)
            
            # 绘制文本
            cv2.putText(result_frame, label, 
                       (text_x + 5, text_y - text_height2 - 8), 
                       font, font_scale, color, thickness)
            cv2.putText(result_frame, confidence_text, 
                       (text_x + 5, text_y - 3), 
                       font, font_scale, color, thickness)
        
        # 绘制系统信息
        info_lines = [
            f"FPS: {self.current_fps:.1f}",
            f"Faces: {len(results)}",
            f"Recognized: {sum(1 for r in results if r['recognized'])}"
        ]
        
        for i, line in enumerate(info_lines):
            y_pos = 30 + i * 25
            cv2.putText(result_frame, line, (result_frame.shape[1] - 200, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return result_frame
    
    def run_camera(self, camera_id: int = 0, show_window: bool = True):
        """
        运行摄像头实时人脸识别
        
        参数:
            camera_id: 摄像头ID
            show_window: 是否显示窗口
        """
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"无法打开摄像头 {camera_id}")
            return
        
        print("实时人脸识别已启动，按 'q' 退出")
        print("按 'r' 进入注册模式")
        
        registration_mode = False
        registration_name = ""
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("无法读取摄像头帧")
                break
            
            # 处理帧
            results = self.process_frame(frame)
            
            # 绘制结果
            result_frame = self.draw_results(frame, results)
            
            # 注册模式提示
            if registration_mode:
                cv2.putText(result_frame, f"Registration Mode: {registration_name}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(result_frame, "Press SPACE to register, ESC to cancel", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # 更新FPS
            self.update_fps()
            
            # 显示结果
            if show_window:
                cv2.imshow('Face Recognition', result_frame)
            
            # 处理按键
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r') and not registration_mode:
                # 进入注册模式
                registration_name = input("请输入要注册的人员姓名: ").strip()
                if registration_name:
                    registration_mode = True
                    print(f"进入注册模式，准备注册: {registration_name}")
            elif key == ord(' ') and registration_mode:
                # 注册人脸
                if results:
                    # 使用第一个检测到的人脸进行注册
                    face_result = results[0]
                    person_id = f"person_{int(time.time())}"
                    
                    success = self.face_db.register_person(
                        person_id, registration_name, 
                        face_result['features'], face_result['face_roi']
                    )
                    
                    if success:
                        print(f"成功注册: {registration_name}")
                    else:
                        print("注册失败")
                    
                    registration_mode = False
                    registration_name = ""
                else:
                    print("未检测到人脸，请重试")
            elif key == 27 and registration_mode:  # ESC键
                # 取消注册
                registration_mode = False
                registration_name = ""
                print("取消注册")
        
        cap.release()
        if show_window:
            cv2.destroyAllWindows()
    
    def run_video(self, video_path: str, output_path: str = None):
        """
        运行视频文件人脸识别
        
        参数:
            video_path: 输入视频路径
            output_path: 输出视频路径（可选）
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"无法打开视频文件: {video_path}")
            return
        
        # 获取视频信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"处理视频: {video_path}")
        print(f"视频尺寸: {width}x{height}, 总帧数: {total_frames}, FPS: {fps}")
        
        # 创建视频写入对象
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if not out.isOpened():
                print(f"无法创建输出视频文件: {output_path}")
                out = None
        
        frame_count = 0
        recognition_stats = {'total_faces': 0, 'recognized_faces': 0}
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"视频处理完成，共处理 {frame_count} 帧")
                break
            
            try:
                # 处理帧
                results = self.process_frame(frame)
                
                # 统计识别结果
                recognition_stats['total_faces'] += len(results)
                recognition_stats['recognized_faces'] += sum(1 for r in results if r['recognized'])
                
                # 绘制结果
                result_frame = self.draw_results(frame, results)
                
                # 保存或显示结果
                if out and out.isOpened():
                    out.write(result_frame)
                else:
                    if not output_path:
                        cv2.imshow('Video Face Recognition', result_frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            print("用户中断处理")
                            break
                
            except Exception as e:
                print(f"处理第 {frame_count} 帧时出错: {e}")
            
            frame_count += 1
            if frame_count % 100 == 0:
                print(f"已处理 {frame_count}/{total_frames} 帧 ({frame_count/total_frames*100:.1f}%)")
        
        cap.release()
        if out and out.isOpened():
            out.release()
        cv2.destroyAllWindows()
        
        # 输出统计信息
        print("\n识别统计:")
        print(f"总检测人脸数: {recognition_stats['total_faces']}")
        print(f"成功识别人脸数: {recognition_stats['recognized_faces']}")
        if recognition_stats['total_faces'] > 0:
            recognition_rate = recognition_stats['recognized_faces'] / recognition_stats['total_faces'] * 100
            print(f"识别率: {recognition_rate:.1f}%")
        
        print("视频处理完成")


def main():
    parser = argparse.ArgumentParser(description='视频人脸识别系统')
    parser.add_argument('--model', type=str, default='weight/arcface_iresnet50.onnx', 
                       help='ArcFace模型路径')
    parser.add_argument('--database', type=str, default='face_database', 
                       help='人脸数据库路径')
    parser.add_argument('--conf_threshold', type=float, default=0.5, 
                       help='人脸检测置信度阈值')
    parser.add_argument('--recognition_threshold', type=float, default=0.6, 
                       help='人脸识别相似度阈值')
    parser.add_argument('--mode', type=str, choices=['camera', 'video'], default='camera', 
                       help='运行模式')
    parser.add_argument('--input', type=str, help='输入视频路径（video模式）或摄像头ID（camera模式）')
    parser.add_argument('--output', type=str, help='输出视频路径（可选）')
    
    args = parser.parse_args()
    
    # 检查模型文件
    if not os.path.exists(args.model):
        print(f"模型文件不存在: {args.model}")
        return
    
    # 初始化人脸识别系统
    face_recognition = VideoFaceRecognition(
        arcface_model_path=args.model,
        database_path=args.database,
        confidence_threshold=args.conf_threshold,
        recognition_threshold=args.recognition_threshold
    )
    
    # 显示数据库信息
    stats = face_recognition.face_db.get_database_stats()
    print(f"人脸数据库统计: {stats['total_persons']} 人, {stats['total_features']} 个特征")
    
    # 运行识别
    if args.mode == 'camera':
        camera_id = int(args.input) if args.input else 0
        face_recognition.run_camera(camera_id)
    elif args.mode == 'video':
        if not args.input:
            print("视频模式需要指定输入视频路径")
            return
        face_recognition.run_video(args.input, args.output)


if __name__ == "__main__":
    main()