import os
import json
import numpy as np
import cv2
import pickle
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import sqlite3


class FaceDatabase:
    """
    人脸数据库管理类，用于存储和管理人脸特征数据
    """
    
    def __init__(self, db_path: str = "face_database"):
        """
        初始化人脸数据库
        
        参数:
            db_path: 数据库存储路径
        """
        self.db_path = db_path
        self.features_file = os.path.join(db_path, "face_features.pkl")
        self.metadata_file = os.path.join(db_path, "metadata.json")
        self.images_dir = os.path.join(db_path, "face_images")
        
        # 创建必要的目录
        os.makedirs(db_path, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        
        # 加载现有数据
        self.face_data = self._load_face_data()
        self.metadata = self._load_metadata()
    
    def _load_face_data(self) -> Dict:
        """
        加载人脸特征数据
        
        返回:
            人脸数据字典
        """
        if os.path.exists(self.features_file):
            try:
                with open(self.features_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"加载人脸特征数据失败: {e}")
        
        return {}
    
    def _load_metadata(self) -> Dict:
        """
        加载元数据
        
        返回:
            元数据字典
        """
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"加载元数据失败: {e}")
        
        return {}
    
    def _save_face_data(self):
        """
        保存人脸特征数据
        """
        try:
            with open(self.features_file, 'wb') as f:
                pickle.dump(self.face_data, f)
        except Exception as e:
            print(f"保存人脸特征数据失败: {e}")
    
    def _save_metadata(self):
        """
        保存元数据
        """
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存元数据失败: {e}")
    
    def register_person(self, person_id: str, person_name: str, 
                       face_features: np.ndarray, face_image: np.ndarray = None) -> bool:
        """
        注册新人员
        
        参数:
            person_id: 人员ID
            person_name: 人员姓名
            face_features: 人脸特征向量
            face_image: 人脸图像（可选）
            
        返回:
            是否注册成功
        """
        try:
            # 检查是否已存在
            if person_id in self.face_data:
                print(f"人员 {person_id} 已存在，将更新特征")
            
            # 存储特征
            if person_id not in self.face_data:
                self.face_data[person_id] = []
            
            self.face_data[person_id].append(face_features)
            
            # 更新元数据
            self.metadata[person_id] = {
                'name': person_name,
                'register_time': datetime.now().isoformat(),
                'feature_count': len(self.face_data[person_id])
            }
            
            # 保存人脸图像
            if face_image is not None:
                image_path = os.path.join(self.images_dir, f"{person_id}_{len(self.face_data[person_id])}.jpg")
                cv2.imwrite(image_path, face_image)
                self.metadata[person_id]['last_image'] = image_path
            
            # 保存数据
            self._save_face_data()
            self._save_metadata()
            
            print(f"成功注册人员: {person_name} (ID: {person_id})")
            return True
            
        except Exception as e:
            print(f"注册人员失败: {e}")
            return False
    
    def add_face_sample(self, person_id: str, face_features: np.ndarray, 
                       face_image: np.ndarray = None) -> bool:
        """
        为已存在的人员添加新的人脸样本
        
        参数:
            person_id: 人员ID
            face_features: 人脸特征向量
            face_image: 人脸图像（可选）
            
        返回:
            是否添加成功
        """
        if person_id not in self.face_data:
            print(f"人员 {person_id} 不存在，请先注册")
            return False
        
        try:
            # 添加特征
            self.face_data[person_id].append(face_features)
            
            # 更新元数据
            self.metadata[person_id]['feature_count'] = len(self.face_data[person_id])
            self.metadata[person_id]['last_update'] = datetime.now().isoformat()
            
            # 保存人脸图像
            if face_image is not None:
                image_path = os.path.join(self.images_dir, f"{person_id}_{len(self.face_data[person_id])}.jpg")
                cv2.imwrite(image_path, face_image)
                self.metadata[person_id]['last_image'] = image_path
            
            # 保存数据
            self._save_face_data()
            self._save_metadata()
            
            print(f"成功为 {person_id} 添加新的人脸样本")
            return True
            
        except Exception as e:
            print(f"添加人脸样本失败: {e}")
            return False
    
    def recognize_face(self, face_features: np.ndarray, 
                      threshold: float = 0.6) -> Optional[Tuple[str, str, float]]:
        """
        识别人脸
        
        参数:
            face_features: 待识别的人脸特征
            threshold: 相似度阈值
            
        返回:
            (person_id, person_name, similarity) 或 None
        """
        if not self.face_data:
            return None
        
        best_match = None
        best_similarity = 0.0
        
        for person_id, stored_features_list in self.face_data.items():
            # 计算与该人员所有特征的相似度
            similarities = []
            for stored_features in stored_features_list:
                similarity = np.dot(face_features, stored_features)
                similarities.append(similarity)
            
            # 取最高相似度
            max_similarity = max(similarities)
            
            if max_similarity > best_similarity and max_similarity >= threshold:
                best_similarity = max_similarity
                person_name = self.metadata.get(person_id, {}).get('name', 'Unknown')
                best_match = (person_id, person_name, max_similarity)
        
        return best_match
    
    def delete_person(self, person_id: str) -> bool:
        """
        删除人员数据
        
        参数:
            person_id: 人员ID
            
        返回:
            是否删除成功
        """
        try:
            if person_id in self.face_data:
                del self.face_data[person_id]
            
            if person_id in self.metadata:
                del self.metadata[person_id]
            
            # 删除相关图像文件
            for filename in os.listdir(self.images_dir):
                if filename.startswith(f"{person_id}_"):
                    os.remove(os.path.join(self.images_dir, filename))
            
            # 保存数据
            self._save_face_data()
            self._save_metadata()
            
            print(f"成功删除人员: {person_id}")
            return True
            
        except Exception as e:
            print(f"删除人员失败: {e}")
            return False
    
    def get_all_persons(self) -> List[Dict]:
        """
        获取所有注册人员信息
        
        返回:
            人员信息列表
        """
        persons = []
        for person_id, metadata in self.metadata.items():
            person_info = {
                'id': person_id,
                'name': metadata.get('name', 'Unknown'),
                'register_time': metadata.get('register_time', ''),
                'feature_count': metadata.get('feature_count', 0),
                'last_update': metadata.get('last_update', ''),
                'last_image': metadata.get('last_image', '')
            }
            persons.append(person_info)
        
        return persons
    
    def get_person_info(self, person_id: str) -> Optional[Dict]:
        """
        获取指定人员信息
        
        参数:
            person_id: 人员ID
            
        返回:
            人员信息字典或None
        """
        if person_id not in self.metadata:
            return None
        
        metadata = self.metadata[person_id]
        return {
            'id': person_id,
            'name': metadata.get('name', 'Unknown'),
            'register_time': metadata.get('register_time', ''),
            'feature_count': metadata.get('feature_count', 0),
            'last_update': metadata.get('last_update', ''),
            'last_image': metadata.get('last_image', '')
        }
    
    def get_database_stats(self) -> Dict:
        """
        获取数据库统计信息
        
        返回:
            统计信息字典
        """
        total_persons = len(self.face_data)
        total_features = sum(len(features) for features in self.face_data.values())
        
        return {
            'total_persons': total_persons,
            'total_features': total_features,
            'database_path': self.db_path,
            'last_modified': datetime.now().isoformat()
        }


if __name__ == "__main__":
    # 测试代码
    db = FaceDatabase("test_face_db")
    
    # 显示数据库统计
    stats = db.get_database_stats()
    print("数据库统计信息:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # 显示所有注册人员
    persons = db.get_all_persons()
    print(f"\n已注册人员 ({len(persons)} 人):")
    for person in persons:
        print(f"  ID: {person['id']}, 姓名: {person['name']}, 特征数: {person['feature_count']}")