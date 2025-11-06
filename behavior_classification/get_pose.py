import os
import json
import time
import numpy as np
import cv2
import onnxruntime
import warnings
import argparse
from collections import deque
try:
    from .predict import SkeletonPredictor
except Exception:
    try:
        from predict import SkeletonPredictor
    except Exception:
        SkeletonPredictor = None

warnings.filterwarnings("ignore")

# 颜色定义
palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                    [230, 230, 0], [255, 153, 255], [153, 204, 255],
                    [255, 102, 255], [255, 51, 255], [102, 178, 255],
                    [51, 153, 255], [255, 153, 153], [255, 102, 102],
                    [255, 51, 51], [153, 255, 153], [102, 255, 102],
                    [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
                    [255, 255, 255]])

# 骨架连接定义 (COCO格式) - 使用0-based索引
# COCO 17个关键点: 0-鼻子, 1-左眼, 2-右眼, 3-左耳, 4-右耳, 5-左肩, 6-右肩, 
# 7-左肘, 8-右肘, 9-左腕, 10-右腕, 11-左髋, 12-右髋, 13-左膝, 14-右膝, 15-左踝, 16-右踝
# 正确的COCO骨骼连接定义
skeleton = [
    [15, 13],  # 左踝 -> 左膝
    [13, 11],  # 左膝 -> 左髋
    [16, 14],  # 右踝 -> 右膝
    [14, 12],  # 右膝 -> 右髋
    [11, 12],  # 左髋 -> 右髋
    [5, 11],   # 左肩 -> 左髋
    [6, 12],   # 右肩 -> 右髋
    [5, 6],    # 左肩 -> 右肩
    [5, 7],    # 左肩 -> 左肘
    [6, 8],    # 右肩 -> 右肘
    [7, 9],    # 左肘 -> 左腕
    [8, 10],   # 右肘 -> 右腕
    [1, 2],    # 左眼 -> 右眼
    [0, 1],    # 鼻子 -> 左眼
    [0, 2],    # 鼻子 -> 右眼
    [1, 3],    # 左眼 -> 左耳
    [2, 4],    # 右眼 -> 右耳
    [3, 5],    # 左耳 -> 左肩
    [4, 6]     # 右耳 -> 右肩
]

pose_limb_color = palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
pose_kpt_color = palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), scaleup=True):
    ''' 调整图像大小和两边灰条填充 '''
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im


def xyxy2xywh(x):
    y = np.copy(x)
    y[:, 2] = x[:, 2] - x[:, 0]  # w
    y[:, 3] = x[:, 3] - x[:, 1]  # h
    return y


def scale_boxes(img1_shape, boxes, img0_shape):
    ''' 将预测的坐标信息转换回原图尺度 '''
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
    pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    
    # 处理检测框坐标 (前4列: x, y, w, h)
    boxes[:, 0] -= pad[0]
    boxes[:, 1] -= pad[1]
    boxes[:, :4] /= gain
    
    # 处理关键点坐标 (从第5列开始，每3列为一个关键点: x, y, conf)
    # YOLOv11-pose输出格式: [x, y, w, h, conf, kpt0_x, kpt0_y, kpt0_conf, kpt1_x, kpt1_y, kpt1_conf, ...]
    num_kpts = (boxes.shape[1] - 5) // 3  # 减去前5列(检测框+置信度)，除以3得到关键点数量
    for kid in range(num_kpts):
        kpt_x_idx = 5 + kid * 3      # 关键点x坐标的索引
        kpt_y_idx = 5 + kid * 3 + 1  # 关键点y坐标的索引
        boxes[:, kpt_x_idx] = (boxes[:, kpt_x_idx] - pad[0]) / gain
        boxes[:, kpt_y_idx] = (boxes[:, kpt_y_idx] - pad[1]) / gain
    
    clip_boxes(boxes, img0_shape)
    return boxes


def clip_boxes(boxes, shape):
    top_left_x = boxes[:, 0].clip(0, shape[1])
    top_left_y = boxes[:, 1].clip(0, shape[0])
    bottom_right_x = (boxes[:, 0] + boxes[:, 2]).clip(0, shape[1])
    bottom_right_y = (boxes[:, 1] + boxes[:, 3]).clip(0, shape[0])
    boxes[:, 0] = top_left_x
    boxes[:, 1] = top_left_y
    boxes[:, 2] = bottom_right_x
    boxes[:, 3] = bottom_right_y


def read_img(img, img_mean=127.5, img_scale=1 / 127.5):
    img = (img - img_mean) * img_scale
    img = np.asarray(img, dtype=np.float32)
    img = np.expand_dims(img, 0)
    img = img.transpose(0, 3, 1, 2)
    return img


def plot_skeleton_kpts(im, kpts, steps=3):
    ''' 在图像上绘制关键点和骨架 '''
    num_kpts = len(kpts) // steps
    # 限制关键点数量为17个（COCO格式）
    num_kpts = min(num_kpts, 17)
    
    for kid in range(num_kpts):
        # 确保不超出颜色数组边界
        if kid < len(pose_kpt_color):
            r, g, b = pose_kpt_color[kid]
            x_coord, y_coord = kpts[steps * kid], kpts[steps * kid + 1]
            conf = kpts[steps * kid + 2]
            if conf > 0.5:
                cv2.circle(im, (int(x_coord), int(y_coord)), 3, (int(r), int(g), int(b)), -1)
    
    for sk_id, sk in enumerate(skeleton):
        # 确保骨架索引在有效范围内 (使用0-based索引)
        sk0_idx = sk[0]  # 直接使用0-based索引
        sk1_idx = sk[1]  # 直接使用0-based索引
        
        if sk0_idx * steps + 2 < len(kpts) and sk1_idx * steps + 2 < len(kpts):
            r, g, b = pose_limb_color[sk_id]
            pos1 = (int(kpts[sk0_idx * steps]), int(kpts[sk0_idx * steps + 1]))
            pos2 = (int(kpts[sk1_idx * steps]), int(kpts[sk1_idx * steps + 1]))
            conf1 = kpts[sk0_idx * steps + 2]
            conf2 = kpts[sk1_idx * steps + 2]
            if conf1 > 0.5 and conf2 > 0.5:
                cv2.line(im, pos1, pos2, (int(r), int(g), int(b)), thickness=2)


def initialize_session(model_path):
    ''' 初始化ONNX Runtime会话 '''
    session_options = onnxruntime.SessionOptions()
    session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    ort_session = onnxruntime.InferenceSession(model_path,
                                               session_options=session_options,
                                               providers=['CPUExecutionProvider'])
    return ort_session


def process_frame(ort_session, img, conf_threshold=0.1):
    ''' 处理单帧图像 '''
    # 图像预处理
    image1 = letterbox(img)
    input = read_img(image1, 0.0, 0.00392156862745098)
    input_name = ort_session.get_inputs()[0].name

    # 模型推理
    output = ort_session.run([], {input_name: input})[0]
    
    # 转置输出: (1, 56, 8400) -> (1, 8400, 56)
    output = output.transpose(0, 2, 1)
    
    # 置信度过滤
    output = output[0][output[0, :, 4] > conf_threshold]
    if len(output) == 0:
        return img, None, None  # 没有检测到任何目标，返回原图、空关键点和空检测框

    # 坐标转换
    det_box = xyxy2xywh(output)
    output = scale_boxes(image1.shape, det_box, img.shape)

    # 提取检测框和关键点
    # YOLOv11-pose输出格式: [x, y, w, h, conf, kpt0_x, kpt0_y, kpt0_conf, kpt1_x, kpt1_y, kpt1_conf, ...]
    det_bboxes, det_scores, kpts = output[:, 0:4], output[:, 4], output[:, 5:]

    # 只绘制第一个人的骨骼关键点（避免重复绘制）
    if len(det_bboxes) > 0:
        kpt = kpts[0]  # 只取第一个人的关键点
        plot_skeleton_kpts(img, kpt)

    # 返回处理后的图像、第一个人的关键点和第一个人的检测框
    # 每个人有17个关键点，每个关键点3个值(x, y, conf)，所以取前51个值
    first_person_kpts = kpts[0][:51] if len(kpts[0]) >= 51 else kpts[0]
    return img, first_person_kpts, det_bboxes[0]


def process_video(model_path, input_video_path, output_csv_path, output_video_path=None, conf_threshold=0.1,
                  predictor: SkeletonPredictor = None, action_mapping: dict = None, window_size: int = 50):
    """ 处理整个视频并保存骨骼数据到CSV，并在可用时叠加行为类别与置信度 """
    # 初始化模型
    ort_session = initialize_session(model_path)

    # 打开输入视频
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"无法打开视频文件: {input_video_path}")
        return

    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 创建视频写入对象（如果需要输出视频）
    if output_video_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    else:
        out = None

    # 准备CSV文件
    csv_file = open(output_csv_path, 'w')
    csv_header = ['frame'] + [f'kp_{i}_x' for i in range(17)] + [f'kp_{i}_y' for i in range(17)] + [f'kp_{i}_conf' for i
                                                                                                    in range(17)]
    csv_file.write(','.join(csv_header) + '\n')

    # 滑动窗口缓存（用于行为预测）
    skeleton_buffer = deque(maxlen=window_size)

    def create_temp_csv(skeleton_data):
        import pandas as pd
        rows = []
        for frame_idx, frame_kpts in enumerate(skeleton_data):
            xs = frame_kpts[:, 0]
            ys = frame_kpts[:, 1]
            confs = frame_kpts[:, 2]
            rows.append([frame_idx] + list(xs) + list(ys) + list(confs))
        header = ['frame'] + [f'kp_{i}_x' for i in range(17)] + [f'kp_{i}_y' for i in range(17)] + [f'kp_{i}_conf' for i in range(17)]
        import os
        import pandas as pd
        df = pd.DataFrame(rows, columns=header)
        temp_csv_path = os.path.join(os.path.dirname(output_csv_path), 'temp_window.csv')
        df.to_csv(temp_csv_path, index=False)
        return temp_csv_path

    # 处理每一帧
    frame_count = 0
    start_time = time.time()
    last_pred = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 处理帧
        processed_frame, kpts, det_bbox = process_frame(ort_session, frame, conf_threshold=conf_threshold)

        # 写入CSV
        if kpts is not None:
            # 重新组织关键点数据: x0,y0,conf0, x1,y1,conf1, ... → x0,x1,..., y0,y1,..., conf0,conf1,...
            kpts_reshaped = kpts.reshape(-1, 3)
            xs = kpts_reshaped[:, 0]
            ys = kpts_reshaped[:, 1]
            confs = kpts_reshaped[:, 2]
            row_data = [str(frame_count)] + [str(x) for x in xs] + [str(y) for y in ys] + [str(c) for c in confs]
            csv_file.write(','.join(row_data) + '\n')
            skeleton_buffer.append(kpts_reshaped)
        else:
            # 如果没有检测到关键点，写入空数据
            empty_data = [str(frame_count)] + ['0'] * 51
            csv_file.write(','.join(empty_data) + '\n')
            skeleton_buffer.append(np.zeros((17, 3)))

        # 始终绘制人物紧凑矩形框（基于关键点），确保无论是否有预测都能看到边框
        if kpts is not None:
            try:
                k = kpts.reshape(-1, 3)
                v = k[k[:, 2] > 0.5]
                if len(v) > 0:
                    min_x, max_x = int(np.min(v[:, 0])), int(np.max(v[:, 0]))
                    min_y, max_y = int(np.min(v[:, 1])), int(np.max(v[:, 1]))
                    mx = max(int((max_x - min_x) * 0.1), 20)
                    my = max(int((max_y - min_y) * 0.1), 20)
                    x, y = max(0, min_x - mx), max(0, min_y - my)
                    w = min(processed_frame.shape[1] - x, max_x - min_x + 2 * mx)
                    h = min(processed_frame.shape[0] - y, max_y - min_y + 2 * my)
                    cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # 若有预测器与映射，则在框旁显示类别与置信度
                    if predictor is not None and action_mapping:
                        if len(skeleton_buffer) >= window_size and frame_count % 5 == 0:
                            temp_csv = create_temp_csv(list(skeleton_buffer))
                            last_pred = predictor.predict(temp_csv)
                        pred = last_pred
                        if pred:
                            cls_id = pred.get('class')
                            prob = float(pred.get('probability', 0.0))
                            cls_name = action_mapping.get(cls_id, f"Action {cls_id}")
                            action_text = f"{cls_name}"
                            conf_text = f"{prob:.1%}"
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            fs, th = 0.6, 2
                            (tw1, th1), _ = cv2.getTextSize(action_text, font, fs, th)
                            (tw2, th2), _ = cv2.getTextSize(conf_text, font, fs, th)
                            bg_w = max(tw1, tw2) + 10
                            bg_h = th1 + th2 + 15
                            tx, ty = max(0, x), max(bg_h, y)
                            cv2.rectangle(processed_frame, (tx, ty - bg_h), (tx + bg_w, ty), (0, 0, 0), -1)
                            cv2.rectangle(processed_frame, (tx, ty - bg_h), (tx + bg_w, ty), (0, 255, 0), 2)
                            cv2.putText(processed_frame, action_text, (tx + 5, ty - th2 - 8), font, fs, (0, 255, 0), th)
                            cv2.putText(processed_frame, conf_text, (tx + 5, ty - 3), font, fs, (0, 255, 0), th)
            except Exception:
                pass

        # 写入输出视频（如果需要）
        if out:
            out.write(processed_frame)

        # 显示进度
        frame_count += 1
        if frame_count % 10 == 0:
            elapsed = time.time() - start_time
            remaining = (total_frames - frame_count) * (elapsed / frame_count)
            print(f"处理进度: {frame_count}/{total_frames} | 已用时间: {elapsed:.1f}s | 剩余时间: {remaining:.1f}s")

    # 释放资源
    cap.release()
    if out:
        out.release()
    csv_file.close()
    cv2.destroyAllWindows()

    print(f"视频处理完成，骨骼数据保存到: {output_csv_path}")
    if output_video_path:
        print(f"处理后的视频保存到: {output_video_path}")


if __name__ == '__main__':
    # 根据config文件从video_folder的视频中提取对应行为的关节点信息并输出同名csv文件到data_folder中,设置output_video_dir后同步输出绘制关节点视频
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path',default='config.json',help='config file path')
    parser.add_argument('--onnx_path',default='yolo11s-pose.onnx',help='onnx file path')
    parser.add_argument('--conf_threshold',default=0.6,type=float,help='confidence threshold')
    parser.add_argument('--output_video_dir',default='output',help='output video dir')
    args = parser.parse_args()

    config_path = args.config_path
    model_path = args.onnx_path
    output_video_dir = args.output_video_dir
    conf_threshold = args.conf_threshold
    with open(config_path, 'r') as f:
        config = json.load(f)

    # 处理每个行为类别
    for action in config['actions']:
        action_name = action['name']
        action_id = action['action']
        video_folder = action['video_folder']
        data_folder = action['data_folder']

        # 确保数据目录存在
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)

        # 处理该类别下指定的视频文件
        if 'video_files' in action:
            video_files = action['video_files']
        else:
            # 如果没有指定video_files，则处理所有视频
            video_files = [f for f in os.listdir(video_folder) if f.endswith(('.mp4', '.avi', '.mov'))]

        for video_file in video_files:
            video_name = os.path.splitext(video_file)[0]
            input_video_path = os.path.join(video_folder, video_file)
            output_csv_path = os.path.join(data_folder, f"{video_name}.csv")
            
            # 检查视频文件是否存在
            if not os.path.exists(input_video_path):
                print(f"警告: 视频文件 {input_video_path} 不存在，跳过处理")
                continue
                
            if output_video_dir is None:
                output_video_path = None
            else:
                os.makedirs(output_video_dir, exist_ok=True)
                output_video_path = os.path.join(output_video_dir, video_name+'_pose.mp4')

            print(f"\n处理行为 {action_id}: {video_file}")
            process_video(model_path, input_video_path, output_csv_path, output_video_path,conf_threshold)