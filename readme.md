# 课堂行为分析与人脸识别系统 (Classroom Behavior Analysis & Face Recognition System)

本项目是一个集成了 **YOLOv11-Pose**（姿态估计）、**ST-GCN**（时空图卷积网络，用于动作识别）以及 **ArcFace**（人脸识别）的综合性课堂行为分析系统。支持实时摄像头检测、视频文件分析以及 Web 端管理界面。

## 🛠️ 功能特性

- **多模态行为识别**：能够识别举手 (Raising Hand)、起立 (Standing)、睡觉 (Sleeping)、玩手机 (Looking Down) 等多种课堂行为。
- **实时人脸识别**：基于 ArcFace 实现的高精度人脸检测与识别，支持人员库管理。
- **Web 管理平台**：基于 Django 的后台管理系统，支持视频上传、任务处理、结果可视化及 DeepSeek AI 辅助分析。
- **实时监控**：支持调用本地摄像头进行实时的行为与人脸检测。

## 📦 环境部署

本项目建议使用 Conda 进行环境管理。
请先安装anaconda或者miniconda，
还需要安装cuda 11.8，以支持pytorch的cuda版本。
然后按照以下步骤进行环境部署。
**注意：CUDA 版本要求为 11.8**

### 1. 创建 Conda 环境

```bash

conda env create -f environment.yml

### 2. 安装 PyTorch (CUDA 11.8)

如果使用 `environment.yml` 未能正确安装 PyTorch，请手动执行以下命令安装适配 CUDA 11.8 的版本：

```bash
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 --index-url https://download.pytorch.org/whl/cu118
```

_(注：您也可以根据需要选择其他兼容 CUDA 11.8 的 PyTorch 版本)_

## 📂 模型文件准备

请确保以下模型文件放置在正确的位置：

1.  **行为识别模型 (ST-GCN)**

    - 路径：`behavior_classification/weights/stgcn.pth`
    - 或者：`behavior_classification/checkpoints/yolopose/spatial/best_model.pth`

2.  **人脸识别模型 (ArcFace)**

    - 路径：`face_recognition/weight/arcface_iresnet50.onnx`

3.  **姿态估计模型 (YOLOv11-Pose)**
    - 确保 YOLO ONNX 模型已在 `behavior_classification` 目录下或代码指定路径中。

## 🚀 运行指南

### 1. 启动 Web 管理平台

Web 平台提供了完整的视频处理流程和结果展示。

```bash
cd web/backend
python manage.py runserver
```

- 访问地址：`http://127.0.0.1:8000/`
- 功能：上传课堂视频，系统后台自动进行转码、骨架提取、动作识别，并生成分析报告。

### 2. 运行实时检测 (摄像头)

直接调用摄像头进行实时的行为和人脸检测。

```bash
python behavior_classification/real_time_detection.py
```

- 按 `q` 键退出。

### 3. 人脸库管理

使用人脸注册工具录入学生信息。

```bash
cd face_recognition

# 摄像头注册模式
python face_registration_tool.py --mode camera --name "学生姓名"

# 批量图片注册
python face_registration_tool.py --mode folder --name "学生姓名" --input "path/to/images"
```

## 📂 目录结构详细说明

```text
YOLOv11-POSE-STGCN/
├── behavior_classification/           # 行为识别核心模块
│   ├── net/                           # ST-GCN 网络模型定义
│   │   ├── st_gcn.py                  # 模型主体结构
│   │   └── utils/graph.py             # 人体骨架图结构定义
│   ├── csv/ & csv_augmented/          # 训练数据及增强数据
│   ├── checkpoints/                   # 模型训练权重保存目录
│   ├── output/                        # 行为检测结果视频输出
│   ├── train/                         # 原始训练视频样本
│   ├── config.json                    # 动作类别与数据路径配置文件
│   ├── real_time_detection.py         # [核心] 实时检测与可视化脚本
│   ├── get_pose.py                    # YOLOv11-Pose 姿态提取接口
│   ├── predict.py                     # 动作分类推理接口
│   ├── train.py                       # 模型训练脚本
│   ├── data_loader.py                 # 数据加载与预处理工具
│   ├── data_augmentation.py           # 数据增强工具
│   └── draw_pose.py                   # 骨架绘制辅助工具
├── face_recognition/                  # 人脸识别核心模块
│   ├── face_database/                 # 本地人脸库
│   │   ├── face_images/               # 注册人员原图
│   │   ├── face_features.pkl          # 特征向量序列化文件
│   │   └── metadata.json              # 人员ID与姓名映射
│   ├── video_face_recognition.py      # [核心] 视频/摄像头人脸识别脚本
│   ├── face_registration_tool.py      # 人脸注册录入工具
│   ├── face_detector.py               # 检测与特征提取底层封装
│   └── face_database.py               # 数据库管理类
├── web/backend/                       # Web 管理平台 (Django)
│   ├── api/                           # 核心业务 App
│   │   ├── views.py                   # API 接口定义
│   │   ├── tasks.py                   # 异步任务处理 (调用识别模块)
│   │   └── storage.py                 # 简易 JSON 数据存储
│   ├── backend/                       # 项目配置 (settings.py 等)
│   ├── templates/                     # 前端 HTML 模板
│   ├── data/                          # 运行时任务数据 (jobs.json)
│   ├── upload/ & processed/           # 视频上传与处理结果目录
│   └── manage.py                      # Django 启动脚本
├── environment.yml                    # Conda 环境配置文件
└── README.md                          # 项目说明文档
```

## 📝 常见问题

- **RuntimeError: CUDA out of memory**: 请尝试减小 `batch_size` 或使用更小的模型。
- **模型加载失败**: 请检查 `config.json` 中的模型路径是否与实际文件位置一致。
- **DeepSeek API 错误**: 请在 `web/backend/backend/settings.py` 中配置有效的 `DEEPSEEK_API_KEY`。
