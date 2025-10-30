import cv2
import numpy as np
import argparse
import os
import time
from typing import List, Dict
from face_detector import FaceRecognitionSystem
from face_database import FaceDatabase


class FaceRegistrationTool:
    """
    人脸注册工具，用于批量注册人脸数据
    """
    
    def __init__(self, arcface_model_path: str, database_path: str = "face_database"):
        """
        初始化人脸注册工具
        
        参数:
            arcface_model_path: ArcFace模型路径
            database_path: 人脸数据库路径
        """
        self.face_system = FaceRecognitionSystem(arcface_model_path)
        self.face_db = FaceDatabase(database_path)
    
    def register_from_camera(self, person_name: str, person_id: str = None, 
                           samples_count: int = 5):
        """
        从摄像头注册人脸
        
        参数:
            person_name: 人员姓名
            person_id: 人员ID（可选，默认自动生成）
            samples_count: 采集样本数量
        """
        if person_id is None:
            person_id = f"person_{int(time.time())}"
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("无法打开摄像头")
            return False
        
        print(f"开始注册 {person_name} (ID: {person_id})")
        print(f"需要采集 {samples_count} 个样本")
        print("按空格键采集样本，按 'q' 退出")
        
        collected_samples = 0
        
        while collected_samples < samples_count:
            ret, frame = cap.read()
            if not ret:
                print("无法读取摄像头帧")
                break
            
            # 检测人脸
            results = self.face_system.process_image(frame)
            
            # 绘制检测结果
            display_frame = frame.copy()
            for result in results:
                bbox = result['bbox']
                x, y, w, h = bbox
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # 显示进度信息
            info_text = f"Samples: {collected_samples}/{samples_count}"
            cv2.putText(display_frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, "Press SPACE to capture, 'q' to quit", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('Face Registration', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("用户取消注册")
                break
            elif key == ord(' '):
                # 采集样本
                if results:
                    face_result = results[0]  # 使用第一个检测到的人脸
                    
                    if collected_samples == 0:
                        # 第一次注册
                        success = self.face_db.register_person(
                            person_id, person_name, 
                            face_result['features'], face_result['face_roi']
                        )
                    else:
                        # 添加额外样本
                        success = self.face_db.add_face_sample(
                            person_id, face_result['features'], face_result['face_roi']
                        )
                    
                    if success:
                        collected_samples += 1
                        print(f"采集样本 {collected_samples}/{samples_count}")
                    else:
                        print("样本采集失败")
                else:
                    print("未检测到人脸，请重试")
        
        cap.release()
        cv2.destroyAllWindows()
        
        if collected_samples == samples_count:
            print(f"成功注册 {person_name}，共采集 {collected_samples} 个样本")
            return True
        else:
            print(f"注册未完成，仅采集 {collected_samples} 个样本")
            return False
    
    def register_from_image(self, image_path: str, person_name: str, person_id: str = None):
        """
        从图像文件注册人脸
        
        参数:
            image_path: 图像文件路径
            person_name: 人员姓名
            person_id: 人员ID（可选，默认自动生成）
        """
        if not os.path.exists(image_path):
            print(f"图像文件不存在: {image_path}")
            return False
        
        if person_id is None:
            person_id = f"person_{int(time.time())}"
        
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法读取图像: {image_path}")
            return False
        
        # 检测人脸
        results = self.face_system.process_image(image)
        
        if not results:
            print("图像中未检测到人脸")
            return False
        
        if len(results) > 1:
            print(f"检测到 {len(results)} 个人脸，将使用第一个")
        
        # 注册第一个检测到的人脸
        face_result = results[0]
        success = self.face_db.register_person(
            person_id, person_name, 
            face_result['features'], face_result['face_roi']
        )
        
        if success:
            print(f"成功从图像注册 {person_name} (ID: {person_id})")
            return True
        else:
            print("注册失败")
            return False
    
    def register_from_folder(self, folder_path: str, person_name: str, person_id: str = None):
        """
        从文件夹批量注册人脸
        
        参数:
            folder_path: 包含人脸图像的文件夹路径
            person_name: 人员姓名
            person_id: 人员ID（可选，默认自动生成）
        """
        if not os.path.exists(folder_path):
            print(f"文件夹不存在: {folder_path}")
            return False
        
        if person_id is None:
            person_id = f"person_{int(time.time())}"
        
        # 支持的图像格式
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        # 获取所有图像文件
        image_files = []
        for filename in os.listdir(folder_path):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(folder_path, filename))
        
        if not image_files:
            print(f"文件夹中未找到图像文件: {folder_path}")
            return False
        
        print(f"找到 {len(image_files)} 个图像文件")
        
        registered_count = 0
        first_registration = True
        
        for i, image_path in enumerate(image_files):
            print(f"处理图像 {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
            
            # 读取图像
            image = cv2.imread(image_path)
            if image is None:
                print(f"  无法读取图像，跳过")
                continue
            
            # 检测人脸
            results = self.face_system.process_image(image)
            
            if not results:
                print(f"  未检测到人脸，跳过")
                continue
            
            # 使用第一个检测到的人脸
            face_result = results[0]
            
            if first_registration:
                # 第一次注册
                success = self.face_db.register_person(
                    person_id, person_name, 
                    face_result['features'], face_result['face_roi']
                )
                first_registration = False
            else:
                # 添加额外样本
                success = self.face_db.add_face_sample(
                    person_id, face_result['features'], face_result['face_roi']
                )
            
            if success:
                registered_count += 1
                print(f"  成功注册样本 {registered_count}")
            else:
                print(f"  注册失败")
        
        if registered_count > 0:
            print(f"成功注册 {person_name}，共 {registered_count} 个样本")
            return True
        else:
            print("注册失败，未成功注册任何样本")
            return False
    
    def list_registered_persons(self):
        """
        列出所有已注册的人员
        """
        persons = self.face_db.get_all_persons()
        
        if not persons:
            print("数据库中没有注册的人员")
            return
        
        print(f"\n已注册人员 ({len(persons)} 人):")
        print("-" * 80)
        print(f"{'ID':<15} {'姓名':<20} {'样本数':<8} {'注册时间':<20}")
        print("-" * 80)
        
        for person in persons:
            print(f"{person['id']:<15} {person['name']:<20} {person['feature_count']:<8} {person['register_time'][:19]}")
    
    def delete_person(self, person_id: str):
        """
        删除指定人员
        
        参数:
            person_id: 人员ID
        """
        person_info = self.face_db.get_person_info(person_id)
        if person_info is None:
            print(f"人员 {person_id} 不存在")
            return False
        
        confirm = input(f"确认删除人员 {person_info['name']} (ID: {person_id})? (y/N): ")
        if confirm.lower() == 'y':
            success = self.face_db.delete_person(person_id)
            if success:
                print(f"成功删除人员 {person_id}")
                return True
            else:
                print("删除失败")
                return False
        else:
            print("取消删除")
            return False


def main():
    parser = argparse.ArgumentParser(description='人脸注册工具')
    parser.add_argument('--model', type=str, default='weight/arcface_iresnet50.onnx', 
                       help='ArcFace模型路径')
    parser.add_argument('--database', type=str, default='face_database', 
                       help='人脸数据库路径')
    parser.add_argument('--mode', type=str, 
                       choices=['camera', 'image', 'folder', 'list', 'delete'], 
                       required=True, help='注册模式')
    parser.add_argument('--name', type=str, help='人员姓名')
    parser.add_argument('--id', type=str, help='人员ID（可选）')
    parser.add_argument('--input', type=str, help='输入路径（图像文件或文件夹）')
    parser.add_argument('--samples', type=int, default=5, help='摄像头模式采集样本数量')
    
    args = parser.parse_args()
    
    # 检查模型文件
    if not os.path.exists(args.model):
        print(f"模型文件不存在: {args.model}")
        return
    
    # 初始化注册工具
    registration_tool = FaceRegistrationTool(args.model, args.database)
    
    if args.mode == 'list':
        # 列出已注册人员
        registration_tool.list_registered_persons()
        
    elif args.mode == 'delete':
        # 删除人员
        if not args.id:
            print("删除模式需要指定人员ID (--id)")
            return
        registration_tool.delete_person(args.id)
        
    elif args.mode == 'camera':
        # 摄像头注册
        if not args.name:
            print("摄像头模式需要指定人员姓名 (--name)")
            return
        registration_tool.register_from_camera(args.name, args.id, args.samples)
        
    elif args.mode == 'image':
        # 图像文件注册
        if not args.name or not args.input:
            print("图像模式需要指定人员姓名 (--name) 和图像路径 (--input)")
            return
        registration_tool.register_from_image(args.input, args.name, args.id)
        
    elif args.mode == 'folder':
        # 文件夹批量注册
        if not args.name or not args.input:
            print("文件夹模式需要指定人员姓名 (--name) 和文件夹路径 (--input)")
            return
        registration_tool.register_from_folder(args.input, args.name, args.id)


if __name__ == "__main__":
    main()