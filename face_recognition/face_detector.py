import cv2
import numpy as np
import onnxruntime
import os
from typing import List, Tuple, Optional
import warnings

warnings.filterwarnings("ignore")


class FaceDetector:
    """
    人脸检测器，使用OpenCV的DNN人脸检测模型
    """
    
    def __init__(self, confidence_threshold=0.5):
        """
        初始化人脸检测器
        
        参数:
            confidence_threshold: 人脸检测置信度阈值
        """
        self.confidence_threshold = confidence_threshold
        
        # 使用OpenCV内置的DNN人脸检测模型
        # 这里使用Caffe模型，也可以替换为其他模型
        self.net = None
        self._load_face_detection_model()
    
    def _load_face_detection_model(self):
        """
        加载人脸检测模型
        """
        try:
            # 使用OpenCV的Haar级联分类器作为备选方案
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            print("使用Haar级联分类器进行人脸检测")
        except Exception as e:
            print(f"加载人脸检测模型失败: {e}")
            raise
    
    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        检测图像中的人脸
        
        参数:
            image: 输入图像 (BGR格式)
            
        返回:
            人脸边界框列表 [(x, y, w, h), ...]
        """
        if image is None or image.size == 0:
            return []
        
        # 转换为灰度图像
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 使用Haar级联分类器检测人脸
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # 将numpy数组转换为元组列表
        if len(faces) > 0:
            return [tuple(face) for face in faces]
        else:
            return []
    
    def extract_face_roi(self, image: np.ndarray, bbox: Tuple[int, int, int, int], 
                        margin: float = 0.2) -> Optional[np.ndarray]:
        """
        从图像中提取人脸区域
        
        参数:
            image: 输入图像
            bbox: 人脸边界框 (x, y, w, h)
            margin: 边界扩展比例
            
        返回:
            人脸区域图像
        """
        if image is None or len(bbox) != 4:
            return None
        
        x, y, w, h = bbox
        
        # 添加边界扩展
        margin_x = int(w * margin)
        margin_y = int(h * margin)
        
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(image.shape[1], x + w + margin_x)
        y2 = min(image.shape[0], y + h + margin_y)
        
        face_roi = image[y1:y2, x1:x2]
        
        return face_roi if face_roi.size > 0 else None


class ArcFaceFeatureExtractor:
    """
    ArcFace特征提取器，使用预训练的ArcFace模型提取人脸特征
    """
    
    def __init__(self, model_path: str):
        """
        初始化ArcFace特征提取器
        
        参数:
            model_path: ArcFace ONNX模型路径
        """
        self.model_path = model_path
        self.session = None
        self.input_size = (112, 112)  # ArcFace标准输入尺寸
        self._load_model()
    
    def _load_model(self):
        """
        加载ArcFace模型
        """
        try:
            # 设置ONNX Runtime提供者
            providers = ['CPUExecutionProvider']
            if onnxruntime.get_device() == 'GPU':
                providers.insert(0, 'CUDAExecutionProvider')
            
            self.session = onnxruntime.InferenceSession(
                self.model_path, 
                providers=providers
            )
            
            # 获取输入输出信息
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            
            print(f"ArcFace模型加载成功: {self.model_path}")
            
        except Exception as e:
            print(f"加载ArcFace模型失败: {e}")
            raise
    
    def preprocess_face(self, face_image: np.ndarray) -> np.ndarray:
        """
        预处理人脸图像
        
        参数:
            face_image: 人脸图像 (BGR格式)
            
        返回:
            预处理后的图像数组
        """
        if face_image is None or face_image.size == 0:
            return None
        
        # 调整尺寸到模型输入要求
        face_resized = cv2.resize(face_image, self.input_size)
        
        # 转换为RGB格式
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        
        # 归一化到[-1, 1]
        face_normalized = (face_rgb.astype(np.float32) - 127.5) / 127.5
        
        # 转换维度 (H, W, C) -> (1, C, H, W)
        face_input = np.transpose(face_normalized, (2, 0, 1))
        face_input = np.expand_dims(face_input, axis=0)
        
        return face_input
    
    def extract_features(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """
        提取人脸特征
        
        参数:
            face_image: 人脸图像 (BGR格式)
            
        返回:
            512维特征向量
        """
        if self.session is None:
            print("模型未加载")
            return None
        
        # 预处理
        face_input = self.preprocess_face(face_image)
        if face_input is None:
            return None
        
        try:
            # 模型推理
            features = self.session.run(
                [self.output_name], 
                {self.input_name: face_input}
            )[0]
            
            # L2归一化
            features = features / np.linalg.norm(features, axis=1, keepdims=True)
            
            return features.flatten()
            
        except Exception as e:
            print(f"特征提取失败: {e}")
            return None


class FaceRecognitionSystem:
    """
    人脸识别系统，整合人脸检测和特征提取
    """
    
    def __init__(self, arcface_model_path: str, confidence_threshold: float = 0.5):
        """
        初始化人脸识别系统
        
        参数:
            arcface_model_path: ArcFace模型路径
            confidence_threshold: 人脸检测置信度阈值
        """
        self.face_detector = FaceDetector(confidence_threshold)
        self.feature_extractor = ArcFaceFeatureExtractor(arcface_model_path)
    
    def process_image(self, image: np.ndarray) -> List[dict]:
        """
        处理图像，检测人脸并提取特征
        
        参数:
            image: 输入图像 (BGR格式)
            
        返回:
            人脸信息列表，每个元素包含 {'bbox': (x,y,w,h), 'features': np.ndarray}
        """
        results = []
        
        # 检测人脸
        faces = self.face_detector.detect_faces(image)
        
        for bbox in faces:
            # 提取人脸区域
            face_roi = self.face_detector.extract_face_roi(image, bbox)
            
            if face_roi is not None:
                # 提取特征
                features = self.feature_extractor.extract_features(face_roi)
                
                if features is not None:
                    results.append({
                        'bbox': bbox,
                        'features': features,
                        'face_roi': face_roi
                    })
        
        return results
    
    def calculate_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """
        计算两个特征向量的相似度
        
        参数:
            features1: 特征向量1
            features2: 特征向量2
            
        返回:
            余弦相似度 (0-1之间，越大越相似)
        """
        if features1 is None or features2 is None:
            return 0.0
        
        # 计算余弦相似度
        similarity = np.dot(features1, features2)
        return float(similarity)


if __name__ == "__main__":
    # 测试代码
    model_path = "weight/arcface_iresnet50.onnx"
    
    if os.path.exists(model_path):
        # 初始化人脸识别系统
        face_system = FaceRecognitionSystem(model_path)
        
        # 测试摄像头
        cap = cv2.VideoCapture(0)
        
        print("人脸识别系统测试启动，按 'q' 退出")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 处理帧
            results = face_system.process_image(frame)
            
            # 绘制结果
            for result in results:
                bbox = result['bbox']
                x, y, w, h = bbox
                
                # 绘制人脸框
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # 显示特征维度信息
                features = result['features']
                cv2.putText(frame, f"Features: {len(features)}", 
                           (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            cv2.imshow('Face Recognition Test', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    else:
        print(f"模型文件不存在: {model_path}")