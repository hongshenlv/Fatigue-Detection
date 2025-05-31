# Fatigue-Detection（基于 YOLOv8 的眨眼 + 打哈欠疲劳监测）

## 项目背景 

现如今学生和上班族每天面对电脑的时间越来越长，容易产生眼睛干涩、视疲劳，甚至长期伏案可能导致注意力下降、工作效率降低。在驾驶、操作机械等高风险场景下，疲劳更会带来严重的安全隐患。


**为何选择眨眼和打哈欠作为疲劳监测信号？**

- **眨眼频率过低**：人在高度集中用眼时眨眼次数会大幅减少，导致泪液分布不足，眼睛容易干涩发痒，久而久之出现视疲劳。  
- **打哈欠频率过高**：频繁打哈欠往往与大脑缺氧、警觉性下降相关，也是在身体疲倦时的一种生理表现。  

因此，通过我们电脑的摄像头实时检测用户的眨眼和打哈欠行为，在一定时间窗口内结合阈值进行判断，就能及时检测并提醒用户休息。


## 解决方案

本项目通过以下思路实现“疲劳监测与告警”：


1. **改造自 [Anukriti2512/Eye-Strain-Detection](https://github.com/Anukriti2512/Eye-Strain-Detection)为我们提供了很好的基础。**  
   - 原项目在 macOS 下基于 OpenCV + dlib （68 点人脸关键点）进行眨眼检测，使用 EAR（Eye Aspect Ratio）阈值判断闭眼。  
   - 我们将其改造为 **两个独立的 YOLOv8 分类模型**：  
     - **眨眼分类模型**（Open Eye vs Close Eye）  
     - **打哈欠分类模型**（No Yawn vs Yawn）  

2. **为什么使用 YOLOv8 分类**  
   - 仅需为训练数据标注“开眼/闭眼”或“打哈欠/未打哈欠”分类标签，无需在每张图片上画检测框；  
   - 基于 YOLOv8 预训练权重，只需微调分类头，训练速度快、效率高；  
   - 对多样化人脸、光照、佩戴眼镜等情况具有更好泛化能力。

3. **实时疲劳判断逻辑**  
   - 通过 MediaPipe FaceMesh 先定位左右眼和嘴部 ROI，分别送入两个已训练好的 YOLOv8 分类模型进行推理，得到“左右眼开/闭”和“嘴部是否在打哈欠”的信息；  
   - 统计 **每 60 秒**（以眨眼判定时长为基准）窗口内的眨眼次数 `window_blinks` 和打哈欠次数 `window_yawns`；  
   - 如果 `window_blinks < BLINK_ALERT_THRESHOLD（默认 12 次）` 或 `window_yawns > YAWN_ALERT_THRESHOLD（默认 3 次）`，立即判定为疲劳：  
     1. 发送系统通知（弹窗 + 声音）  
     2. 在画面上显示 “WARNING: FATIGUE”  
     3. 立即将 `window_blinks = 0`、`window_yawns = 0` 并重置窗口计时，进入下一轮监测  
   - 否则保持 “STATUS: NORMAL”，等待 60 秒后再次根据窗口值判定是否疲劳。  

4. **告警与提示**  
   - 前端视频帧上实时叠加：  
     - 累计眨眼次数 `Blinks`  
     - 累计打哈欠次数 `Yawns`  
     - 实时“Eye: Open/Closed” 和 “Mouth: Yawning/No Yawn”  
     - 疲劳状态文本：`WARNING: FATIGUE` 或 `STATUS: NORMAL`  
   - 使用 macOS 通知（`pync` 调用 Notification Center）在疲劳判定时弹窗告警，并发出声音提醒。

## 关键特性
双模型分类：分别用 YOLOv8 模型训练“开/闭眼”和“打哈欠/未打哈欠”分类头

高效训练：只需分类标签，无须标注检测框，基于 YOLOv8 预训练模型微调

实时监测：MediaPipe FaceMesh 定位 ROI，Ultralytics YOLOv8 推理，Flask 推流

疲劳综合判断：结合眨眼与打哈欠信号，60 秒内一旦触发条件 → 立即告警并重置窗口

跨平台扩展：原生支持 macOS 通知，用户可根据需求替换为 Windows/Linux 通知方案

## 项目结构

```bash
Eye-Strain-Detection/
├── Blink_detection/                   
│   ├── best.pt                       # 眨眼分类模型权重（Open/Close Eye）
│   ├── best_yawn.pt                  # 打哈欠分类模型权重（No Yawn/Yawn）
│   └── blink_detection.py            # 核心推理函数：classify_eyes() + classify_yawn()
│
├── dataset_blink.yaml                # 眨眼分类数据集配置
├── dataset_yawn.yaml                 # 打哈欠分类数据集配置
├── train_blink.py                    # 训练眨眼分类模型脚本
├── train_yawn.py                     # 训练打哈欠分类模型脚本
├── webstreaming.py                   # Flask 后端：实时调用分类函数，叠加信息并推视频流
├── Notifier.py                       # macOS 通知模块：疲劳告警弹窗 + 声音
├── shape_predictor_68_face_landmarks.dat  # （原项目依赖，可选）dlib 关键点模型
├── requirements.txt                  # Python 依赖列表
├── Procfile                          # 部署示例（Heroku 等，可选）
├── README.md                         # 项目说明文档（当前文件）
├── static/                           # 前端静态资源（CSS）
│   └── css/
│       ├── bootstrap.min.css
│       └── templatemo-style.css
└── templates/                        
    └── index.html                    # 实时视频流 + 介绍文字的前端模板
```


### 核心文件

- **`Blink_detection/blink_detection.py`**  
  - `classify_eyes(frame, conf_thres)`  
    1. 使用 MediaPipe FaceMesh 定位左右眼 6 个关键点，裁剪出左右眼 ROI  
    2. 以分类模式调用 `blink_model.predict(roi, task='classify')`，得到开/闭眼概率 `probs`  
    3. 根据闭眼类别置信度（`probs[_closed_class_id]`）和阈值判断 `is_closed`  
    4. 在原图画出左右眼边框及置信度标签，返回 `(left_closed: bool, right_closed: bool)`  

  - `classify_yawn(frame, conf_thres)`  
    1. 使用 MediaPipe FaceMesh 提取嘴部关键点 `MOUTH_IDXS = [61,291,0,17]`，裁剪出嘴部 ROI  
    2. 以分类模式调用 `yawn_model.predict(roi, task='classify')`，得到打哈欠/未打哈欠概率 `probs`  
    3. 根据打哈欠类别置信度（`probs[_yawn_class_id]`）和阈值判断 `is_yawn`  
    4. 在原图画出嘴部边框及置信度标签，返回 `is_yawn: bool`  

- **`train_blink.py` / `train_yawn.py`**  
  - 分别对眨眼与打哈欠数据集进行 YOLOv8 分类训练，将数据集配置 `*.yaml` 传给 `model.train(task="classify")`  
  - 训练日志、最优权重保存在 `runs/classify/train*/weights/best.pt` 下，可将其拷贝到 `Blink_detection/` 作为推理模型。  

- **`webstreaming.py`**  
  - 打开摄像头、启动 Flask 应用；后台线程不断读取帧，执行：  
    1. `classify_eyes(frame)` → 更新 `total_blinks`, `window_blinks`，判断眨眼事件  
    2. `classify_yawn(frame)` → 更新 `total_yawns`, `window_yawns`，判断打哈欠事件  
    3. 每帧叠加四组信息：累计眨眼次数、累计打哈欠次数、当前“Eye: …” / “Mouth: …”  
    4. 判断疲劳并实时更新“WARNING: FATIGUE” 或 “STATUS: NORMAL”  
    5. 将叠加完文本的 `frame` 存入全局变量 `outputFrame`  
  - Flask 路由 `/video_feed` 调用 `generate()`，从 `outputFrame` 中读取 JPEG 数据并以 `multipart/x-mixed-replace` 形式推送给前端 `<img>`，实现实时预览。  

- **`Notifier.py`**（仅限 macOS）  
  - 通过 `pync.Notifier.notify(...)` 向 macOS 发送系统通知，并附带声音和点击打开监测页面的链接。  
  - 使用前需确保安装了 `terminal-notifier`：  
    ```bash
    brew install terminal-notifier
    ```

## 安装与使用

> **注意**：本项目在 macOS 下使用 `pync` 调用系统通知；如果在 Windows 或 Linux 平台使用，请自行替换通知模块（例如 `win10toast`、`notify-send` 等）。

### 1. 克隆仓库

```bash
git clone git@github.com:hongshenlv/Fatigue-Detection.git
cd Fatigue-Detection
```

### 2. 创建并激活虚拟环境
```bash
python3 -m venv venv
source venv/bin/activate       # macOS/Linux
# Windows PowerShell: .\venv\Scripts\Activate.ps1
```

### 3. 安装依赖
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. 准备数据集

#### 4.1 眨眼分类数据集（Eyes Dataset）

1. 在[Kaggle Eyes Dataset](https://www.kaggle.com/datasets/charunisa/eyes-dataset)下载数据

2. 将数据解压并按以下结构组织：
```bash
dataset_blink/
├── train/
│   ├── open eye/
│   └── close eye/
└── val/
    ├── open eye/
    └── close eye/
```

3. 确保项目根目录下存在 dataset_blink.yaml，并内容正确。例如：
```bash
path: ./dataset_blink
train: train
val: val
nc: 2
names:
  0: open eye
  1: close eye
```

#### 4.2 打哈欠分类数据集（Yawning Dataset）

1. 在[Kaggle Yawning Dataset](https://www.kaggle.com/datasets/deepankarvarma/yawning-dataset-classification)下载数据

2. 将数据解压并按以下结构组织（需手动划分训练/验证子集）：
```bash
dataset_yawn/
├── train/
│   ├── no_yawn/
│   └── yawn/
└── val/
    ├── no_yawn/
    └── yawn/
```

3. 确保项目根目录下存在 dataset_blink.yaml，并内容正确。例如：
```bash
path: ./dataset_yawn
train: train
val: val
nc: 2
names:
  0: no_yawn
  1: yawn
```

### 5. 训练模型

请分别运行：
```bash
# 训练眨眼分类模型
python3 train_blink.py

# 训练打哈欠分类模型
python3 train_yawn.py
```

### 6. 运行疲劳监测

```bash
python3 webstreaming.py
```

- 打开浏览器访问 http://127.0.0.1:5000/ ，即可看到实时摄像头画面及叠加的：

   - 累计眨眼次数

   - 累计打哈欠次数

   - 当前“Eye: Open/Closed”

   - 当前“Mouth: Yawning/No Yawn”

   - 疲劳状态文本：WARNING: FATIGUE 或 STATUS: NORMAL

- macOS 下，当检测到疲劳条件时，会弹出系统通知并播放声音；在界面上同时会显示 “WARNING: FATIGUE”。

```bash
# Clone this repository
$ git clone https://github.com/Anukriti2512/Eye-Strain-Detection.git

# Go into the repository
$ cd Eye-Strain-Detection

# Create a virtual/conda environment and activate it: 
$ conda create --name myenv
$ conda activate myenv

# Install dependencies
$ pip install -r requirements.txt

# Run the app
$ python webstreaming.py

# Copy the IP Address on a web browser and use the application to see blink detection in real-time
```

## 未来改进与扩展

1. **更丰富的疲劳指标**

   除了眨眼次数和打哈欠次数，可结合头部姿态（nod）、瞳孔跳动等信息进行多模态融合。

2. **跨平台通知方案**

   当前仅支持 macOS 通知，后续会考虑添加 Windows 或 Linux 版本。

3. **优化嘴部 ROI 定位**

   目前示例使用 FaceMesh 的固定点，可根据实际数据集标注方式调整更准确的嘴部区域。

4. **前端可视化增强**

   增加绘制 60 秒滑动窗口内眨眼/打哈欠次数曲线，帮助用户直观了解疲劳趋势。

5. **模型轻量化与量化**

   将 YOLOv8 分类模型裁剪或量化部署到手机/嵌入式设备，实现移动端实时疲劳监测。


### References

1. [Eye-Strain-Detection](https://github.com/Anukriti2512/Eye-Strain-Detection)
2. [Research Paper: Real-Time Eye Blink Detection Using Facial Landmarks](http://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf)
3. [real-time-drowsy-driving-detection](https://github.com/tyrerodr/real-time-drowsy-driving-detection)
4. [Facial and Landmark Recognition Models](http://dlib.net/)
5. [Creating web-application using Flask](https://towardsdatascience.com/designing-a-machine-learning-model-and-deploying-it-using-flask-on-heroku-9558ce6bde7b)
6. [HTML+CSS](https://templatemo.com/tag/video)
7. [Eye Health knowledge](https://visionsource.com/blog/are-you-blinking-enough/)
