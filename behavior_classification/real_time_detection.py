import cv2
import numpy as np
import torch
import onnxruntime
import json
import time
from collections import deque
import argparse
try:
    from .predict import SkeletonPredictor
except Exception:
    from predict import SkeletonPredictor
try:
    from .get_pose import initialize_session, process_frame, plot_skeleton_kpts
except Exception:
    from get_pose import initialize_session, process_frame, plot_skeleton_kpts


class RealTimeActionDetector:
    def __init__(self, config_path, model_path, onnx_path, device=None, window_size=50, conf_threshold=0.6):
        """
        实时行为检测器
        
        参数:
            config_path: 配置文件路径
            model_path: ST-GCN模型权重路径
            onnx_path: YOLO姿态检测ONNX模型路径
            device: 指定设备 (None则自动选择)
            window_size: 滑动窗口大小（帧数）
            conf_threshold: 姿态检测置信度阈值
        """
        # 行为分类映射
        self.action_mapping = {
            0: "Normal Listening", 1: "Normal Listening", 2: "Normal Listening", 8: "Normal Listening",
            3: "Raising Hand", 4: "Raising Hand", 5: "Raising Hand", 6: "Raising Hand",
            7: "Standing",
            9: "Passing Objects", 10: "Passing Objects", 11: "Passing Objects", 12: "Passing Objects",
            13: "Turning Around", 14: "Turning Around", 15: "Turning Around", 16: "Turning Around",
            17: "Sleeping",
            18: "Looking Down"
        }
        
        # 初始化姿态检测模型
        self.pose_session = initialize_session(onnx_path)
        self.conf_threshold = conf_threshold
        
        # 初始化行为识别预测器
        self.action_predictor = SkeletonPredictor(config_path, model_path, device)
        
        # 滑动窗口设置
        self.window_size = window_size
        self.skeleton_buffer = deque(maxlen=window_size)
        
        # 预测结果缓存
        self.last_prediction = None
        self.prediction_confidence = 0.0
        self.prediction_history = deque(maxlen=10)  # 保存最近10次预测结果
        self.prediction_buffer = deque(maxlen=5)  # 存储5帧的预测结果用于平滑
        self.last_stable_prediction = None  # 最后一个稳定的预测结果
        self.stable_count = 0  # 连续稳定预测的计数
        
        # 性能统计
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
    def process_skeleton_data(self, kpts):
        """
        处理单帧的骨骼关键点数据
        
        参数:
            kpts: 关键点数据 [17, 3] (x, y, confidence)
            
        返回:
            是否有足够数据进行预测
        """
        if kpts is None:
            # 如果没有检测到人体，添加零数据
            zero_kpts = np.zeros((17, 3))
            self.skeleton_buffer.append(zero_kpts)
        else:
            # 重新整理关键点数据格式
            kpts_reshaped = kpts.reshape(-1, 3)  # [17, 3]
            self.skeleton_buffer.append(kpts_reshaped)
        
        # 检查是否有足够的帧数进行预测
        return len(self.skeleton_buffer) >= self.window_size
    
    def create_temp_csv(self, skeleton_data):
        """
        将滑动窗口中的骨骼数据转换为临时CSV格式
        
        参数:
            skeleton_data: 骨骼数据列表
            
        返回:
            CSV格式的字符串
        """
        import io
        import pandas as pd
        
        # 创建CSV数据
        csv_data = []
        for frame_idx, frame_kpts in enumerate(skeleton_data):
            xs = frame_kpts[:, 0]  # x坐标
            ys = frame_kpts[:, 1]  # y坐标
            confs = frame_kpts[:, 2]  # 置信度
            
            row_data = [frame_idx] + list(xs) + list(ys) + list(confs)
            csv_data.append(row_data)
        
        # 创建DataFrame
        header = ['frame'] + [f'kp_{i}_x' for i in range(17)] + [f'kp_{i}_y' for i in range(17)] + [f'kp_{i}_conf' for i in range(17)]
        df = pd.DataFrame(csv_data, columns=header)
        
        # 保存为临时CSV文件
        temp_csv_path = "temp_skeleton_data.csv"
        df.to_csv(temp_csv_path, index=False)
        
        return temp_csv_path
    
    def predict_action(self):
        """
        基于当前滑动窗口中的数据预测行为
        
        返回:
            预测结果字典
        """
        if len(self.skeleton_buffer) < self.window_size:
            return None
        
        try:
            # 将滑动窗口数据转换为CSV格式
            skeleton_data = list(self.skeleton_buffer)
            temp_csv_path = self.create_temp_csv(skeleton_data)
            
            # 使用行为识别模型进行预测
            result = self.action_predictor.predict(temp_csv_path)
            
            # 更新预测历史
            self.prediction_history.append(result)
            
            # 使用多数投票或置信度加权来稳定预测结果
            self.last_prediction = result
            self.prediction_confidence = result['probability']
            
            # 添加到预测缓冲区
            self.prediction_buffer.append(result)
            
            # 预测结果平滑处理
            return self.smooth_predictions()
            
        except Exception as e:
            print(f"预测错误: {e}")
            return None
    
    def smooth_predictions(self):
        """平滑预测结果，减少跳动"""
        if len(self.prediction_buffer) < 3:
            return self.prediction_buffer[-1] if self.prediction_buffer else None
        
        # 统计最近几帧的预测结果
        class_counts = {}
        total_confidence = {}
        
        for pred in self.prediction_buffer:
            class_id = pred['class']
            if class_id not in class_counts:
                class_counts[class_id] = 0
                total_confidence[class_id] = 0
            class_counts[class_id] += 1
            total_confidence[class_id] += pred['probability']
        
        # 找到出现次数最多的类别
        most_frequent_class = max(class_counts, key=class_counts.get)
        avg_confidence = total_confidence[most_frequent_class] / class_counts[most_frequent_class]
        
        # 获取对应的英文行为名称
        action_name = self.action_mapping.get(most_frequent_class, f"Action {most_frequent_class}")
        
        # 创建平滑后的预测结果
        smoothed_prediction = {
            'class': most_frequent_class,
            'probability': avg_confidence,
            'class_name': action_name
        }
        
        # 检查预测稳定性
        if (self.last_stable_prediction is None or 
            self.last_stable_prediction['class'] == most_frequent_class):
            self.stable_count += 1
        else:
            self.stable_count = 1
        
        # 只有当预测足够稳定时才更新显示结果
        if self.stable_count >= 2 or avg_confidence > 0.8:
            self.last_stable_prediction = smoothed_prediction
            return smoothed_prediction
        else:
            # 返回上一个稳定的预测结果
            return self.last_stable_prediction
    
    def get_stable_prediction(self):
        """
        获取稳定的预测结果（基于历史预测的多数投票）
        
        返回:
            稳定的预测结果
        """
        if len(self.prediction_history) < 3:
            return self.last_prediction
        
        # 统计最近几次预测的类别
        recent_predictions = list(self.prediction_history)[-5:]  # 最近5次预测
        class_votes = {}
        
        for pred in recent_predictions:
            if pred:
                class_id = pred['class']
                if class_id not in class_votes:
                    class_votes[class_id] = []
                class_votes[class_id].append(pred['probability'])
        
        if not class_votes:
            return self.last_prediction
        
        # 选择投票最多且平均置信度最高的类别
        best_class = max(class_votes.keys(), 
                        key=lambda x: (len(class_votes[x]), np.mean(class_votes[x])))
        
        avg_confidence = np.mean(class_votes[best_class])
        
        # 获取对应的英文行为名称
        action_name = self.action_mapping.get(best_class, f"Action {best_class}")
        
        return {
            'class': best_class,
            'probability': avg_confidence,
            'class_name': action_name,
            'vote_count': len(class_votes[best_class])
        }
    
    def draw_results(self, frame, kpts, bbox, prediction):
        """
        在帧上绘制检测结果
        
        Args:
            frame: 原始帧
            kpts: 关键点数据
            bbox: 检测框数据 [x, y, w, h]
            prediction: 预测结果
            
        Returns:
            绘制结果的帧
        """
        result_frame = frame.copy()
        
        # 绘制骨骼关键点
        if kpts is not None:
            plot_skeleton_kpts(result_frame, kpts)  # 该函数直接在图像上绘制，无返回值
        
        # 绘制人物检测框
        if bbox is not None and kpts is not None:
            # 基于关键点计算更紧凑的边界框
            kpts_reshaped = kpts.reshape(-1, 3)
            valid_kpts = kpts_reshaped[kpts_reshaped[:, 2] > 0.5]  # 只考虑置信度高的关键点
            
            if len(valid_kpts) > 0:
                # 计算关键点的边界
                min_x = int(np.min(valid_kpts[:, 0]))
                max_x = int(np.max(valid_kpts[:, 0]))
                min_y = int(np.min(valid_kpts[:, 1]))
                max_y = int(np.max(valid_kpts[:, 1]))
                
                # 添加适当的边距
                margin_x = int((max_x - min_x) * 0.1)
                margin_y = int((max_y - min_y) * 0.1)
                
                # 确保边距不会太小
                margin_x = max(margin_x, 20)
                margin_y = max(margin_y, 20)
                
                # 计算最终的边界框
                x = max(0, min_x - margin_x)
                y = max(0, min_y - margin_y)
                w = min(result_frame.shape[1] - x, max_x - min_x + 2 * margin_x)
                h = min(result_frame.shape[0] - y, max_y - min_y + 2 * margin_y)
                
                # 绘制矩形框
                cv2.rectangle(result_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # 在矩形框左上角绘制预测结果
                if prediction:
                    # 准备文本信息
                    action_text = f"{prediction['class_name']}"
                    confidence_text = f"{prediction['probability']:.1%}"
                    
                    # 计算文本尺寸
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.6
                    thickness = 2
                    
                    (text_width1, text_height1), _ = cv2.getTextSize(action_text, font, font_scale, thickness)
                    (text_width2, text_height2), _ = cv2.getTextSize(confidence_text, font, font_scale, thickness)
                    
                    # 计算背景框尺寸
                    bg_width = max(text_width1, text_width2) + 10
                    bg_height = text_height1 + text_height2 + 15
                    
                    # 确保文本框不超出图像边界
                    text_x = max(0, x)
                    text_y = max(bg_height, y)
                    
                    # 绘制背景框
                    cv2.rectangle(result_frame, 
                                (text_x, text_y - bg_height), 
                                (text_x + bg_width, text_y), 
                                (0, 0, 0), -1)
                    cv2.rectangle(result_frame, 
                                (text_x, text_y - bg_height), 
                                (text_x + bg_width, text_y), 
                                (0, 255, 0), 2)
                    
                    # 绘制文本
                    cv2.putText(result_frame, action_text, 
                              (text_x + 5, text_y - text_height2 - 8), 
                              font, font_scale, (0, 255, 0), thickness)
                    cv2.putText(result_frame, confidence_text, 
                              (text_x + 5, text_y - 3), 
                              font, font_scale, (0, 255, 0), thickness)
        
        # 绘制系统信息（右上角）
        if prediction:
            info_lines = [
                f"FPS: {self.current_fps:.1f}",
                f"Buffer: {len(self.skeleton_buffer)}/{self.window_size}"
            ]
            
            for i, line in enumerate(info_lines):
                y_pos = 30 + i * 25
                cv2.putText(result_frame, line, (frame.shape[1] - 150, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return result_frame
    
    def update_fps(self):
        """更新FPS计算"""
        self.fps_counter += 1
        if self.fps_counter >= 30:  # 每30帧更新一次FPS
            current_time = time.time()
            self.current_fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def run_camera(self, camera_id=0):
        """
        运行摄像头实时检测
        
        参数:
            camera_id: 摄像头ID
        """
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"无法打开摄像头 {camera_id}")
            return
        
        print("实时行为检测已启动，按 'q' 退出")
        print("等待收集足够的帧数据进行预测...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("无法读取摄像头帧")
                break
            
            # 姿态检测
            processed_frame, kpts, bbox = process_frame(self.pose_session, frame, self.conf_threshold)
            
            # 处理骨骼数据
            can_predict = self.process_skeleton_data(kpts)
            
            # 行为预测
            prediction = None
            if can_predict:
                if len(self.skeleton_buffer) % 10 == 0:  # 每10帧预测一次，减少计算负担
                    prediction = self.predict_action()
                else:
                    prediction = self.get_stable_prediction()
            
            # 绘制结果
            result_frame = self.draw_results(frame, kpts, bbox, prediction)
            
            # 更新FPS
            self.update_fps()
            
            # 显示结果
            cv2.imshow('Real-time Action Detection', result_frame)
            
            # 检查退出条件
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def run_video(self, video_path, output_path=None):
        """
        运行视频文件检测
        
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
        
        # 检查视频尺寸是否有效
        if width <= 0 or height <= 0:
            print(f"错误: 视频尺寸无效 ({width}x{height})")
            cap.release()
            return
        
        # 创建视频写入对象
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if not out.isOpened():
                print(f"无法创建输出视频文件: {output_path}")
                out = None
        else:
            out = None
        
        # 课堂行为统计（基于平滑后的预测结果按帧累计）
        behavior_counts = {
            "Normal Listening": 0,
            "Raising Hand": 0,
            "Standing": 0,
            "Passing Objects": 0,
            "Turning Around": 0,
            "Sleeping": 0,
            "Looking Down": 0,
        }

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"视频读取完成，共处理 {frame_count} 帧")
                break
            
            # 检查帧是否有效
            if frame is None or frame.size == 0:
                print(f"警告: 第 {frame_count} 帧无效，跳过")
                frame_count += 1
                continue
            
            try:
                # 姿态检测
                processed_frame, kpts, bbox = process_frame(self.pose_session, frame, self.conf_threshold)
                
                # 处理骨骼数据
                can_predict = self.process_skeleton_data(kpts)
                
                # 行为预测
                prediction = None
                if can_predict and frame_count % 5 == 0:  # 每5帧预测一次
                    prediction = self.predict_action()
                else:
                    prediction = self.get_stable_prediction()
                
                # 绘制结果
                result_frame = self.draw_results(frame, kpts, bbox, prediction)

                # 累计课堂行为统计（按帧计数）
                try:
                    if prediction and isinstance(prediction, dict):
                        name = prediction.get('class_name')
                        if name in behavior_counts:
                            behavior_counts[name] += 1
                except Exception:
                    pass
                
                # 检查结果帧是否有效
                if result_frame is None or result_frame.size == 0:
                    print(f"警告: 第 {frame_count} 帧处理结果无效，跳过显示")
                    frame_count += 1
                    continue
                
                # 保存或显示结果
                if out and out.isOpened():
                    out.write(result_frame)
                else:
                    # 只有在没有输出文件时才显示窗口
                    if not output_path:
                        cv2.imshow('Video Action Detection', result_frame)
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
        print("视频处理完成")
        # 返回统计结果供后端使用
        return behavior_counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='实时行为检测系统')
    parser.add_argument('--config', type=str, default='config.json', help='配置文件路径')
    parser.add_argument('--model', type=str, required=True, help='ST-GCN模型权重路径')
    parser.add_argument('--onnx', type=str, default='weights/yolo11s-pose.onnx', help='YOLO姿态检测ONNX模型路径')
    parser.add_argument('--conf_threshold', type=float, default=0.6, help='姿态检测置信度阈值')
    parser.add_argument('--window_size', type=int, default=50, help='滑动窗口大小')
    parser.add_argument('--mode', type=str, choices=['camera', 'video'], default='video', help='运行模式')
    parser.add_argument('--input', type=str, help='输入视频路径（video模式）或摄像头ID（camera模式）')
    parser.add_argument('--output', type=str, help='输出视频路径（可选）')
    
    args = parser.parse_args()
    
    # 初始化检测器
    detector = RealTimeActionDetector(
        config_path=args.config,
        model_path=args.model,
        onnx_path=args.onnx,
        window_size=args.window_size,
        conf_threshold=args.conf_threshold
    )
    
    # 运行检测
    if args.mode == 'camera':
        camera_id = int(args.input) if args.input else 0
        detector.run_camera(camera_id)
    elif args.mode == 'video':
        if not args.input:
            print("视频模式需要指定输入视频路径")
            exit(1)
        detector.run_video(args.input, args.output)