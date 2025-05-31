# Fatigue-Detection（基于 YOLOv8 的眨眼 + 打哈欠疲劳监测）

## 项目背景 

现如今学生和上班族每天面对电脑的时间越来越长，容易产生眼睛干涩、视疲劳，甚至长期伏案可能导致注意力下降、工作效率降低。在驾驶、操作机械等高风险场景下，疲劳更会带来严重的安全隐患。
为何选择眨眼和打哈欠作为疲劳监测信号？

·眨眼频率过低：人在高度集中用眼时眨眼次数会大幅减少，导致泪液分布不足，眼睛容易干涩发痒，久而久之出现视疲劳。

·打哈欠频率过高：频繁打哈欠往往与大脑缺氧、警觉性下降相关，也是在身体疲倦时的一种生理表现。

因此，通过我们电脑的摄像头实时检测用户的眨眼和打哈欠行为，在一定时间窗口内结合阈值进行判断，就能及时检测并提醒用户休息。


## 解决方案

本项目通过以下思路实现“疲劳监测与告警”：

1.改造自 Anukriti2512/Eye-Strain-Detection

  ·原项目在 macOS 下基于 OpenCV + dlib（68 点人脸关键点）进行眨眼检测，使用 EAR（Eye Aspect Ratio）阈值判断闭眼，给我们提供了非常好的基础。
  
  ·我们将其改造为 两个独立的 YOLOv8 分类模型：
  
    ·眨眼分类模型（Open Eye vs Close Eye）
    
    ·打哈欠分类模型（No Yawn vs Yawn）

2.为什么使用 YOLOv8 分类

  ·仅需为训练数据标注“开眼/闭眼”或“打哈欠/未打哈欠”分类标签，无需在每张图片上画检测框；
  
  ·基于 YOLOv8 预训练权重，只需微调分类头，训练速度快、效率高；
  
  ·对多样化人脸、光照、佩戴眼镜等情况具有更好泛化能力。

3.实时疲劳判断逻辑

  ·通过 MediaPipe FaceMesh 先定位左右眼和嘴部 ROI，分别送入两个已训练好的 YOLOv8 分类模型进行推理，得到“左右眼开/闭”和“嘴部是否在打哈欠”的信息；
  
  ·统计 每 60 秒（以眨眼判定时长为基准）窗口内的眨眼次数 window_blinks 和打哈欠次数 window_yawns；
  
  ·如果 window_blinks < BLINK_ALERT_THRESHOLD（默认 12 次） 或 window_yawns > YAWN_ALERT_THRESHOLD（默认 3 次），立即判定为疲劳：
  
    1).发送系统通知（弹窗 + 声音）
    
    2).在画面上显示 “WARNING: FATIGUE”
    
    3).立即将 window_blinks=0、window_yawns=0 并重置窗口计时，进入下一轮监测
    
  ·否则保持 “STATUS: NORMAL”，等待 60 秒后再次根据窗口值判定是否疲劳。

4.告警与提示

  ·我们在前端视频帧上实时叠加：
  
    ·累计眨眼次数 Blinks
    
    ·累计打哈欠次数 Yawns
    
    ·实时“Eye: Open/Closed” 和 “Mouth: Yawning/No Yawn”
    
    ·疲劳状态文本：WARNING: FATIGUE 或 STATUS: NORMAL
    
  ·使用 macOS 通知（pync 调用 Notification Center）在疲劳判定时弹窗告警，并发出声音提醒。

## 关键特性
·双模型分类：分别用 YOLOv8 模型训练“开/闭眼”和“打哈欠/未打哈欠”分类头

·高效训练：只需分类标签，无须标注检测框，基于 YOLOv8 预训练模型微调

·实时监测：MediaPipe FaceMesh 定位 ROI，Ultralytics YOLOv8 推理，Flask 推流

·疲劳综合判断：结合眨眼与打哈欠信号，60 秒内一旦触发条件 → 立即告警并重置窗口

·跨平台扩展：原生支持 macOS 通知，用户可根据需求替换为 Windows/Linux 通知方案

## 项目结构

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

## 核心文件
Blink_detection/blink_detection.py

·classify_eyes(frame, conf_thres)
  ·使用 MediaPipe FaceMesh 定位左右眼 6 个关键点，裁剪出左右眼 ROI
  ·以分类模式调用 blink_model.predict(roi, task='classify')，得到开/闭眼概率 probs
  ·根据闭眼类别置信度（probs[_closed_class_id]）和阈值判断 is_closed
  ·在原图画出左右眼边框及置信度标签，返回 (left_closed: bool, right_closed: bool)

·classify_yawn(frame, conf_thres)
  ·使用 MediaPipe FaceMesh 提取嘴部关键点 MOUTH_IDXS = [61,291,0,17]，裁剪出嘴部 ROI
  ·以分类模式调用 yawn_model.predict(roi, task='classify')，得到打哈欠/未打哈欠概率 probs
  ·根据打哈欠类别置信度（probs[_yawn_class_id]）和阈值判断 is_yawn
  ·在原图画出嘴部边框及置信度标签，返回 is_yawn: bool

·train_blink.py / train_yawn.py

  ·分别对眨眼与打哈欠数据集进行 YOLOv8 分类训练，将数据集配置 *.yaml 传给 model.train(task="classify")
  ·训练日志、最优权重保存在 runs/classify/train*/weights/best.pt 下，可将其拷贝到 Blink_detection/ 作为推理模型。

·webstreaming.py
  ·打开摄像头、启动 Flask 应用；后台线程不断读取帧，执行：
  1).classify_eyes(frame) → 更新 total_blinks, window_blinks，判断眨眼事件
  2).classify_yawn(frame) → 更新 total_yawns, window_yawns，判断打哈欠事件
  3).每帧叠加四组信息：累计眨眼次数、累计打哈欠次数、当前“Eye: …” / “Mouth: …”
  4).判断疲劳并实时更新“WARNING: FATIGUE” 或 “STATUS: NORMAL”
  5).将叠加完文本的 frame 存入全局变量 outputFrame


·Notifier.py（仅限 macOS）

  ·通过 pync.Notifier.notify(...) 向 macOS 发送系统通知，并附带声音和点击打开监测页面的链接。

  ·使用前需确保安装了 terminal-notifier：

```bash
brew install terminal-notifier
```


### Key Features 

- Instantly see your eye blinking rate
- Realtime eye strain alerts (notification & sound)
- The Computer Vision models used are robust and error rate is very low for different expressions, angles and even for people wearing spectacles[2]
- Click on the notifications to see more information!
- The app will work in the background and send reminders even if you don't open the browser
- If your eyes are closed for a prolonged period of time (5-6 secs), blinks are not detected


## How To Use

Note: Currently, the webapp has not been deployed but this can be cloned and used easily. Further, only supports MacOS as of now. 

To clone and run this application, you'll need [Git](https://git-scm.com), [Python](https://www.python.org/) and Anaconda 3 installed on your computer. From your command line:

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


## Approach

The blink detector computes a metric called the eye aspect ratio (EAR), introduced by Soukupová and Čech in their 2016 paper, Real-Time Eye Blink Detection Using Facial Landmarks[1]. The eye aspect ratio makes for an elegant algorithm that involves a very simple calculation based on the ratio of distances between facial landmarks of the eyes. 

Each eye is represented by 6 (x, y)-coordinates, starting at the left-corner of the eye, and then working clockwise around the remainder of the region:

![blink_detection_6_landmarks](https://user-images.githubusercontent.com/37685052/91079233-6ccfdf00-e661-11ea-8804-25269701d328.jpg) 

Based on this image, we can see find a relation between the width and the height of these coordinates. We can then derive an equation that reflects this relation called the eye aspect ratio (EAR): 

![blink_detection_equation](https://user-images.githubusercontent.com/37685052/91079328-8a04ad80-e661-11ea-90b7-01d89fad71d2.png)

The eye aspect ratio is approximately constant while the eye is open, but will rapidly fall to zero when a blink is taking place. Using this simple equation, we can avoid image processing techniques and simply rely on the ratio of eye landmark distances to determine if a person is blinking. A frame threshold range is used to ensure that the person actually blinked and that their eyes are not closed for a long time.

![blink_detection_plot](https://user-images.githubusercontent.com/37685052/91079315-87a25380-e661-11ea-9f03-9c32bee8f9cc.jpg)

In this project, I have used existing Deep Learning models that detect faces and facial landmarks from images/video streams. These return the coordinates of the facial features like left eye, right eye, nose, etc. which have been used to calculate EAR. Blinking rate is monitored per minute.


## Demo 

You can view a demo of this project here : https://youtu.be/Tt2DR8FvYDk

## Scope for improvement & future plans

1. Currenly, EAR is the only quantitative metric used to determine if a person has blinked. However, due to noise in a video stream, subpar facial landmark detections, or fast changes in viewing angle, it could produce a false-positive detection, reporting that a blink had taken place when in reality the person had not blinked. To improve the blink detector, Soukupová and Čech recommend constructing a 13-dim feature vector of eye aspect ratios (N-th frame, N – 6 frames, and N + 6 frames), followed by feeding this feature vector into a Linear SVM for classification.

2. The UI is still being improved, and the web-app will be deployed soon.

3. Currently, this only works for MacOS due to some library limilations. It will be made cross platform soon.

4. Visualizations will be added so users can see insights about their blinking habits.


### References

1. [Research Paper: Real-Time Eye Blink Detection Using Facial Landmarks](http://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf)
2. [Facial and Landmark Recognition Models](http://dlib.net/)
3. [Creating web-application using Flask](https://towardsdatascience.com/designing-a-machine-learning-model-and-deploying-it-using-flask-on-heroku-9558ce6bde7b)
4. [HTML+CSS](https://templatemo.com/tag/video)
5. [Eye Health knowledge](https://visionsource.com/blog/are-you-blinking-enough/)
