import numpy as np
import pandas as pd
import os
import random
import math
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
import argparse
from tqdm import tqdm

class SkeletonDataAugmentation:
    """骨骼关键点数据增强类"""
    
    def __init__(self):
        # COCO 17关键点格式 (0-16)
        self.num_keypoints = 17
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
        # 对称关键点对 (用于水平翻转)
        self.symmetric_pairs = [
            (1, 2),   # left_eye <-> right_eye
            (3, 4),   # left_ear <-> right_ear
            (5, 6),   # left_shoulder <-> right_shoulder
            (7, 8),   # left_elbow <-> right_elbow
            (9, 10),  # left_wrist <-> right_wrist
            (11, 12), # left_hip <-> right_hip
            (13, 14), # left_knee <-> right_knee
            (15, 16)  # left_ankle <-> right_ankle
        ]
    
    def load_csv_data(self, csv_path):
        """加载CSV数据"""
        df = pd.read_csv(csv_path)
        return df
    
    def extract_keypoints(self, df):
        """从DataFrame中提取关键点坐标和置信度"""
        frames = []
        for _, row in df.iterrows():
            frame_data = {
                'frame': row['frame'],
                'keypoints': []
            }
            
            # 提取每个关键点的x, y, conf
            for i in range(self.num_keypoints):
                x = row[f'kp_{i}_x']
                y = row[f'kp_{i}_y']
                conf = row[f'kp_{i}_conf']
                frame_data['keypoints'].append([x, y, conf])
            
            frames.append(frame_data)
        
        return frames
    
    def keypoints_to_dataframe(self, frames, original_df):
        """将关键点数据转换回DataFrame格式"""
        data = []
        
        for frame_data in frames:
            row = {'frame': frame_data['frame']}
            
            # 添加关键点数据
            for i, (x, y, conf) in enumerate(frame_data['keypoints']):
                row[f'kp_{i}_x'] = x
                row[f'kp_{i}_y'] = y
                row[f'kp_{i}_conf'] = conf
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def horizontal_flip(self, frames, image_width=1280):
        """水平翻转增强"""
        augmented_frames = []
        
        for frame_data in frames:
            new_frame = {
                'frame': frame_data['frame'],
                'keypoints': []
            }
            
            keypoints = np.array(frame_data['keypoints'])
            
            # 翻转x坐标
            keypoints[:, 0] = image_width - keypoints[:, 0]
            
            # 交换对称关键点
            for left_idx, right_idx in self.symmetric_pairs:
                keypoints[[left_idx, right_idx]] = keypoints[[right_idx, left_idx]]
            
            new_frame['keypoints'] = keypoints.tolist()
            augmented_frames.append(new_frame)
        
        return augmented_frames
    
    def rotation_augmentation(self, frames, angle_range=(-15, 15), center=None):
        """旋转增强"""
        angle = random.uniform(angle_range[0], angle_range[1])
        angle_rad = math.radians(angle)
        
        augmented_frames = []
        
        for frame_data in frames:
            new_frame = {
                'frame': frame_data['frame'],
                'keypoints': []
            }
            
            keypoints = np.array(frame_data['keypoints'])
            
            # 如果没有指定中心点，使用所有关键点的中心
            if center is None:
                valid_points = keypoints[keypoints[:, 2] > 0.5]  # 只考虑置信度高的点
                if len(valid_points) > 0:
                    cx = np.mean(valid_points[:, 0])
                    cy = np.mean(valid_points[:, 1])
                else:
                    cx, cy = 640, 360  # 默认图像中心
            else:
                cx, cy = center
            
            # 旋转变换
            cos_angle = math.cos(angle_rad)
            sin_angle = math.sin(angle_rad)
            
            for i, (x, y, conf) in enumerate(keypoints):
                if conf > 0.1:  # 只旋转有效的关键点
                    # 平移到原点
                    x_centered = x - cx
                    y_centered = y - cy
                    
                    # 旋转
                    x_rotated = x_centered * cos_angle - y_centered * sin_angle
                    y_rotated = x_centered * sin_angle + y_centered * cos_angle
                    
                    # 平移回去
                    x_new = x_rotated + cx
                    y_new = y_rotated + cy
                    
                    keypoints[i] = [x_new, y_new, conf]
            
            new_frame['keypoints'] = keypoints.tolist()
            augmented_frames.append(new_frame)
        
        return augmented_frames
    
    def scaling_augmentation(self, frames, scale_range=(0.8, 1.2), center=None):
        """缩放增强"""
        scale = random.uniform(scale_range[0], scale_range[1])
        
        augmented_frames = []
        
        for frame_data in frames:
            new_frame = {
                'frame': frame_data['frame'],
                'keypoints': []
            }
            
            keypoints = np.array(frame_data['keypoints'])
            
            # 如果没有指定中心点，使用所有关键点的中心
            if center is None:
                valid_points = keypoints[keypoints[:, 2] > 0.5]
                if len(valid_points) > 0:
                    cx = np.mean(valid_points[:, 0])
                    cy = np.mean(valid_points[:, 1])
                else:
                    cx, cy = 640, 360
            else:
                cx, cy = center
            
            # 缩放变换
            for i, (x, y, conf) in enumerate(keypoints):
                if conf > 0.1:
                    x_new = cx + (x - cx) * scale
                    y_new = cy + (y - cy) * scale
                    keypoints[i] = [x_new, y_new, conf]
            
            new_frame['keypoints'] = keypoints.tolist()
            augmented_frames.append(new_frame)
        
        return augmented_frames
    
    def noise_augmentation(self, frames, noise_std=2.0):
        """添加高斯噪声"""
        augmented_frames = []
        
        for frame_data in frames:
            new_frame = {
                'frame': frame_data['frame'],
                'keypoints': []
            }
            
            keypoints = np.array(frame_data['keypoints'])
            
            # 为x和y坐标添加高斯噪声
            for i, (x, y, conf) in enumerate(keypoints):
                if conf > 0.1:  # 只对有效关键点添加噪声
                    noise_x = np.random.normal(0, noise_std)
                    noise_y = np.random.normal(0, noise_std)
                    keypoints[i] = [x + noise_x, y + noise_y, conf]
            
            new_frame['keypoints'] = keypoints.tolist()
            augmented_frames.append(new_frame)
        
        return augmented_frames
    
    def temporal_warping(self, frames, warp_factor=0.1):
        """时间扭曲增强"""
        if len(frames) < 10:  # 序列太短不进行时间扭曲
            return frames
        
        original_length = len(frames)
        
        # 创建扭曲的时间索引
        warp_strength = random.uniform(-warp_factor, warp_factor)
        
        # 生成非线性时间映射
        original_indices = np.arange(original_length)
        warped_indices = original_indices + warp_strength * np.sin(2 * np.pi * original_indices / original_length) * original_length / 10
        
        # 确保索引在有效范围内
        warped_indices = np.clip(warped_indices, 0, original_length - 1)
        
        augmented_frames = []
        
        # 对每个关键点进行插值
        for kp_idx in range(self.num_keypoints):
            x_coords = [frame['keypoints'][kp_idx][0] for frame in frames]
            y_coords = [frame['keypoints'][kp_idx][1] for frame in frames]
            confs = [frame['keypoints'][kp_idx][2] for frame in frames]
            
            # 创建插值函数
            if len(set(x_coords)) > 1:  # 确保有变化
                interp_x = interp1d(original_indices, x_coords, kind='linear', bounds_error=False, fill_value='extrapolate')
                interp_y = interp1d(original_indices, y_coords, kind='linear', bounds_error=False, fill_value='extrapolate')
                interp_conf = interp1d(original_indices, confs, kind='linear', bounds_error=False, fill_value='extrapolate')
                
                # 应用扭曲
                new_x = interp_x(warped_indices)
                new_y = interp_y(warped_indices)
                new_conf = interp_conf(warped_indices)
                
                # 更新帧数据
                for i, frame in enumerate(frames):
                    if i >= len(augmented_frames):
                        augmented_frames.append({
                            'frame': frame['frame'],
                            'keypoints': [kp[:] for kp in frame['keypoints']]
                        })
                    
                    augmented_frames[i]['keypoints'][kp_idx] = [new_x[i], new_y[i], new_conf[i]]
        
        return augmented_frames
    
    def smooth_augmentation(self, frames, sigma=1.0):
        """平滑增强（高斯滤波）"""
        if len(frames) < 5:
            return frames
        
        augmented_frames = []
        
        # 对每个关键点应用高斯平滑
        for kp_idx in range(self.num_keypoints):
            x_coords = np.array([frame['keypoints'][kp_idx][0] for frame in frames])
            y_coords = np.array([frame['keypoints'][kp_idx][1] for frame in frames])
            confs = np.array([frame['keypoints'][kp_idx][2] for frame in frames])
            
            # 应用高斯滤波
            smoothed_x = gaussian_filter1d(x_coords, sigma=sigma)
            smoothed_y = gaussian_filter1d(y_coords, sigma=sigma)
            
            # 更新帧数据
            for i, frame in enumerate(frames):
                if i >= len(augmented_frames):
                    augmented_frames.append({
                        'frame': frame['frame'],
                        'keypoints': [kp[:] for kp in frame['keypoints']]
                    })
                
                augmented_frames[i]['keypoints'][kp_idx] = [smoothed_x[i], smoothed_y[i], confs[i]]
        
        return augmented_frames
    
    def augment_sequence(self, frames, augmentation_type='random', **kwargs):
        """对序列应用指定的增强方法"""
        if augmentation_type == 'random':
            # 随机选择增强方法
            methods = ['horizontal_flip', 'rotation', 'scaling', 'noise', 'temporal_warp', 'smooth']
            augmentation_type = random.choice(methods)
        
        if augmentation_type == 'horizontal_flip':
            return self.horizontal_flip(frames, **kwargs)
        elif augmentation_type == 'rotation':
            return self.rotation_augmentation(frames, **kwargs)
        elif augmentation_type == 'scaling':
            return self.scaling_augmentation(frames, **kwargs)
        elif augmentation_type == 'noise':
            return self.noise_augmentation(frames, **kwargs)
        elif augmentation_type == 'temporal_warp':
            return self.temporal_warping(frames, **kwargs)
        elif augmentation_type == 'smooth':
            return self.smooth_augmentation(frames, **kwargs)
        else:
            return frames
    
    def process_csv_file(self, input_path, output_dir, num_augmentations=5):
        """处理单个CSV文件，生成增强数据"""
        # 加载原始数据
        df = self.load_csv_data(input_path)
        frames = self.extract_keypoints(df)
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存原始数据
        original_output_path = os.path.join(output_dir, f"original_{os.path.basename(input_path)}")
        df.to_csv(original_output_path, index=False)
        
        augmented_files = [original_output_path]
        
        # 生成增强数据
        for i in range(num_augmentations):
            # 随机选择增强方法
            augmentation_methods = [
                ('horizontal_flip', {}),
                ('rotation', {'angle_range': (-20, 20)}),
                ('scaling', {'scale_range': (0.8, 1.2)}),
                ('noise', {'noise_std': 3.0}),
                ('temporal_warp', {'warp_factor': 0.15}),
                ('smooth', {'sigma': 1.5})
            ]
            
            method, params = random.choice(augmentation_methods)
            
            # 应用增强
            augmented_frames = self.augment_sequence(frames, method, **params)
            
            # 转换回DataFrame并保存
            augmented_df = self.keypoints_to_dataframe(augmented_frames, df)
            output_path = os.path.join(output_dir, f"aug_{method}_{i}_{os.path.basename(input_path)}")
            augmented_df.to_csv(output_path, index=False)
            augmented_files.append(output_path)
        
        return augmented_files

def main():
    parser = argparse.ArgumentParser(description='骨骼关键点数据增强工具')
    parser.add_argument('--input_dir', type=str, default='csv', help='输入CSV目录')
    parser.add_argument('--output_dir', type=str, default='csv_augmented', help='输出目录')
    parser.add_argument('--num_augmentations', type=int, default=5, help='每个文件生成的增强样本数量')
    parser.add_argument('--actions', type=str, nargs='+', help='指定要处理的行为类别，如 action_0 action_1')
    
    args = parser.parse_args()
    
    augmenter = SkeletonDataAugmentation()
    
    # 获取所有行为目录
    if args.actions:
        action_dirs = [os.path.join(args.input_dir, action) for action in args.actions if os.path.exists(os.path.join(args.input_dir, action))]
    else:
        action_dirs = [os.path.join(args.input_dir, d) for d in os.listdir(args.input_dir) 
                      if os.path.isdir(os.path.join(args.input_dir, d)) and d.startswith('action_')]
    
    print(f"找到 {len(action_dirs)} 个行为类别目录")
    
    total_original = 0
    total_augmented = 0
    
    # 处理每个行为类别
    for action_dir in tqdm(action_dirs, desc="处理行为类别"):
        action_name = os.path.basename(action_dir)
        print(f"\n处理 {action_name}...")
        
        # 创建输出目录
        output_action_dir = os.path.join(args.output_dir, action_name)
        
        # 查找CSV文件
        csv_files = [f for f in os.listdir(action_dir) if f.endswith('.csv')]
        
        if not csv_files:
            print(f"  警告: {action_name} 目录中没有找到CSV文件")
            continue
        
        action_augmented = 0
        
        for csv_file in csv_files:
            input_path = os.path.join(action_dir, csv_file)
            
            try:
                # 处理文件
                augmented_files = augmenter.process_csv_file(
                    input_path, 
                    output_action_dir, 
                    args.num_augmentations
                )
                
                action_augmented += len(augmented_files)
                print(f"  {csv_file}: 生成了 {len(augmented_files)} 个文件")
                
            except Exception as e:
                print(f"  错误: 处理 {csv_file} 时出错: {e}")
        
        total_original += len(csv_files)
        total_augmented += action_augmented
        
        print(f"  {action_name} 完成: {len(csv_files)} -> {action_augmented} 个文件")
    
    print(f"\n数据增强完成!")
    print(f"原始文件: {total_original}")
    print(f"增强后文件: {total_augmented}")
    print(f"增强倍数: {total_augmented/total_original:.1f}x")
    print(f"输出目录: {args.output_dir}")

if __name__ == "__main__":
    main()