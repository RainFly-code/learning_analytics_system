# 人脸识别系统

基于 ArcFace 模型的实时人脸识别系统，支持视频文件和摄像头输入。

## 功能特性

- 实时人脸检测和识别
- 人脸特征提取和匹配
- 人脸数据库管理
- 支持摄像头和视频文件输入
- 多种人脸注册方式
- 可调节的识别阈值

## 文件结构

```
face_recognition/
├── weight/
│   └── arcface_iresnet50.onnx      # ArcFace 模型文件
├── face_detector.py                # 人脸检测和特征提取
├── face_database.py               # 人脸数据库管理
├── video_face_recognition.py      # 主要识别程序
├── face_registration_tool.py      # 人脸注册工具
└── README.md                      # 使用说明
```

## 安装依赖

```bash
pip install opencv-python numpy onnxruntime
```

## 使用方法

### 1. 人脸注册

#### 从摄像头注册
```bash
python face_registration_tool.py --mode camera --name "张三" --samples 5
```

#### 从图像文件注册
```bash
python face_registration_tool.py --mode image --name "李四" --input "path/to/image.jpg"
```

#### 从文件夹批量注册
```bash
python face_registration_tool.py --mode folder --name "王五" --input "path/to/folder"
```

#### 查看已注册人员
```bash
python face_registration_tool.py --mode list
```

#### 删除人员
```bash
python face_registration_tool.py --mode delete --id "person_id"
```

### 2. 实时人脸识别

#### 使用摄像头
```bash
python video_face_recognition.py --mode camera
```

#### 处理视频文件
```bash
python video_face_recognition.py --mode video --input "path/to/video.mp4"
```

#### 注册模式（边识别边注册）
```bash
python video_face_recognition.py --mode camera --register
```

### 3. 参数说明

#### video_face_recognition.py 参数
- `--model`: ArcFace 模型路径（默认: weight/arcface_iresnet50.onnx）
- `--database`: 人脸数据库路径（默认: face_database）
- `--mode`: 运行模式（camera/video）
- `--input`: 视频文件路径（video 模式必需）
- `--output`: 输出视频路径（可选）
- `--threshold`: 识别阈值（默认: 0.6）
- `--register`: 启用注册模式

#### face_registration_tool.py 参数
- `--model`: ArcFace 模型路径
- `--database`: 人脸数据库路径
- `--mode`: 注册模式（camera/image/folder/list/delete）
- `--name`: 人员姓名
- `--id`: 人员ID（可选，默认自动生成）
- `--input`: 输入路径（图像文件或文件夹）
- `--samples`: 摄像头模式采集样本数量（默认: 5）

## 使用示例

### 完整工作流程

1. **注册人脸**
   ```bash
   # 从摄像头注册
   python face_registration_tool.py --mode camera --name "员工A" --samples 3
   
   # 从图像注册
   python face_registration_tool.py --mode image --name "员工B" --input "employee_b.jpg"
   ```

2. **查看注册结果**
   ```bash
   python face_registration_tool.py --mode list
   ```

3. **开始识别**
   ```bash
   # 实时识别
   python video_face_recognition.py --mode camera --threshold 0.7
   
   # 处理视频文件
   python video_face_recognition.py --mode video --input "meeting.mp4" --output "result.mp4"
   ```

### 高级用法

#### 批量注册多个人员
```bash
# 为每个人创建文件夹，包含多张照片
python face_registration_tool.py --mode folder --name "张三" --input "photos/zhangsan/"
python face_registration_tool.py --mode folder --name "李四" --input "photos/lisi/"
```

#### 调整识别精度
```bash
# 高精度模式（较少误识别）
python video_face_recognition.py --mode camera --threshold 0.8

# 高召回模式（更容易识别）
python video_face_recognition.py --mode camera --threshold 0.5
```

## 数据库结构

人脸数据库存储在指定目录下，包含以下文件：
- `features.pkl`: 人脸特征向量
- `metadata.json`: 人员信息和元数据
- `images/`: 人脸图像文件

## 性能优化建议

1. **硬件要求**
   - 推荐使用 GPU 加速（需要 onnxruntime-gpu）
   - 摄像头分辨率建议 640x480 或 1280x720

2. **参数调优**
   - 识别阈值：0.6-0.8 之间效果较好
   - 每人注册 3-5 个样本即可
   - 注册时确保光照充足、角度多样

3. **数据库管理**
   - 定期清理无效人员数据
   - 避免数据库过大影响性能
   - 建议单个数据库不超过 1000 人

## 常见问题

### Q: 识别准确率不高怎么办？
A: 
1. 增加注册样本数量
2. 确保注册时光照条件良好
3. 调整识别阈值
4. 重新注册质量更好的人脸图像

### Q: 程序运行缓慢怎么办？
A: 
1. 降低视频分辨率
2. 使用 GPU 加速
3. 减少数据库中的人员数量
4. 优化人脸检测参数

### Q: 无法检测到人脸怎么办？
A: 
1. 检查光照条件
2. 确保人脸清晰可见
3. 调整摄像头角度和距离
4. 检查 OpenCV 安装是否正确

## 技术细节

- **人脸检测**: 使用 OpenCV Haar 级联分类器
- **特征提取**: ArcFace (ResNet50) 模型
- **特征匹配**: 余弦相似度计算
- **数据存储**: Pickle + JSON 格式

## 许可证

本项目仅供学习和研究使用。