# Blink_detection/blink_detection.py

import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO

# 1) 初始化 MediaPipe FaceMesh，用于提取人脸关键点
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ———— 眨眼分类模型部分 ————

# 2) 左右眼的 6 个关键点索引（来自 MediaPipe FaceMesh）
LEFT_EYE_IDX  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 380, 373]

# 3) 加载「眨眼」分类模型权重，假设文件名为 best.pt，放在同级目录
BLINK_MODEL_PATH = __file__.replace('blink_detection.py', 'best.pt')
blink_model = YOLO(BLINK_MODEL_PATH)

# 4) 找到「open」「close」各自对应的 class_id
_open_class_id   = next((cid for cid,name in blink_model.model.names.items() if 'open'  in name.lower()), 0)
_closed_class_id = next((cid for cid,name in blink_model.model.names.items() if 'close' in name.lower()), 1)

# ———— 打哈欠分类模型部分 ————

# 5) 加载「打哈欠」分类模型权重，假设文件名为 best_yawn.pt
YAWN_MODEL_PATH = __file__.replace('blink_detection.py', 'best_yawn.pt')
yawn_model = YOLO(YAWN_MODEL_PATH)

# 6) 找到「yawn」「no_yawn」各自对应的 class_id
#    我们假设模型的 names = ['no_yawn', 'yawn']（可以通过打印 yawn_model.model.names 确认）
_no_yawn_class_id = next((cid for cid,name in yawn_model.model.names.items()
                          if 'no_yawn' in name.lower() or 'no yaw' in name.lower()), 0)
_yawn_class_id    = next((cid for cid,name in yawn_model.model.names.items()
                          if 'yawn'    in name.lower()), 1)

# ———— 辅助：把 out.probs 转为 numpy 数组的函数 ————

def _probs_to_numpy(raw):
    """
    将 out.probs（可能是 Tensor，也可能是 Results、list、etc）转为 numpy ndarray。
    """
    # 如果是 torch.Tensor
    if hasattr(raw, 'cpu') and hasattr(raw, 'detach'):
        return raw.detach().cpu().numpy()
    # 已经是 numpy
    if isinstance(raw, np.ndarray):
        return raw
    # 有 .data 属性
    if hasattr(raw, 'data'):
        d = raw.data
        if isinstance(d, np.ndarray):
            return d
        if hasattr(d, 'cpu'):
            return d.detach().cpu().numpy()
    # 如果 raw 本身是可迭代的 list/tuple
    try:
        arr = np.array(list(raw))
        if arr.dtype != object:
            return arr
    except Exception:
        pass
    raise RuntimeError(f"无法把 out.probs 转为 numpy，原始类型：{type(raw)}")

# ———— classify_eyes 函数：对左右眼做 open/close 分类 ————

def classify_eyes(frame: np.ndarray, conf_thres: float = 0.3):
    """
    对左右眼 ROI 用 blink_model 做 open/close 分类。
    在原图上画出左右眼的方框+标签，返回 (left_closed, right_closed)。
    """
    h, w = frame.shape[:2]
    rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if not results.multi_face_landmarks:
        # 没检测到人脸时，默认为都没闭眼
        return False, False

    lm = results.multi_face_landmarks[0].landmark
    eye_states = []

    for idxs in (LEFT_EYE_IDX, RIGHT_EYE_IDX):
        # 1) 计算眼部的 bounding‐box 坐标，并加若干 padding
        pts = [(int(lm[i].x * w), int(lm[i].y * h)) for i in idxs]
        xs, ys = zip(*pts)
        pad = 15
        x1 = max(min(xs) - pad, 0)
        x2 = min(max(xs) + pad, w)
        y1 = max(min(ys) - pad, 0)
        y2 = min(max(ys) + pad, h)
        roi = frame[y1:y2, x1:x2]

        # 默认：认为 open（未闭眼）
        is_closed = False
        label = blink_model.model.names[_open_class_id]
        color = (0, 255, 0)  # 绿色表示 open

        if roi.size > 0:
            # 2) 做一次分类推理
            out = blink_model.predict(roi, conf=conf_thres, verbose=False, task='classify')[0]
            raw = out.probs
            probs = _probs_to_numpy(raw)  # 一定得到 numpy ndarray

            # 3) 判断是否闭眼
            if probs[_closed_class_id] >= conf_thres:
                is_closed = True
                label = f"{blink_model.model.names[_closed_class_id]} {probs[_closed_class_id]:.2f}"
                color = (0, 0, 255)  # 红色表示 closed
            else:
                label = f"{blink_model.model.names[_open_class_id]} {probs[_open_class_id]:.2f}"
                color = (0, 255, 0)

        # 4) 把 ROI 框和文字画回原图
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        eye_states.append(is_closed)

    return eye_states[0], eye_states[1]


# ———— classify_yawn 函数：对「打哈欠」做分类 ————

# 这里我们简单地把「嘴部」当作 ROI，借助 FaceMesh 的部分嘴部关键点（例如 [13,14, 78,308] 只是示例），
# 然后把嘴部 ROI 提取出来，用 yawn_model 给出「打哈欠／不打哈欠」的概率。
# 这里给出一个基于 MediaPipe 裁口型区域的示例做法（可根据训练时的具体标注方式做微调）。

MOUTH_IDXS = [61, 291, 0, 17]  # 示例：左嘴角、右嘴角、下巴顶点、上嘴唇顶点

def classify_yawn(frame: np.ndarray, conf_thres: float = 0.3):
    """
    对嘴部 ROI 做「打哈欠／不打哈欠」分类，在原图上画 ROI + label，返回 is_yawning（True/False）。
    """
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if not results.multi_face_landmarks:
        return False

    lm = results.multi_face_landmarks[0].landmark

    # 1) 根据 MOUTH_IDXS 计算一个大致的嘴部矩形
    pts = [(int(lm[i].x * w), int(lm[i].y * h)) for i in MOUTH_IDXS]
    xs, ys = zip(*pts)
    pad_x = 20
    pad_y = 10
    x1 = max(min(xs) - pad_x, 0)
    x2 = min(max(xs) + pad_x, w)
    y1 = max(min(ys) - pad_y, 0)
    y2 = min(max(ys) + pad_y, h)
    mouth_roi = frame[y1:y2, x1:x2]

    # 默认：不打哈欠
    is_yawn = False
    label = yawn_model.model.names[_no_yawn_class_id]
    color = (0, 255, 0)  # 绿色表示“未打哈欠”

    if mouth_roi.size > 0:
        # 2) 做一次分类推理
        out = yawn_model.predict(mouth_roi, conf=conf_thres, verbose=False, task='classify')[0]
        raw = out.probs
        probs = _probs_to_numpy(raw)  # numpy ndarray

        # 3) 判断是否打哈欠
        if probs[_yawn_class_id] >= conf_thres:
            is_yawn = True
            label = f"{yawn_model.model.names[_yawn_class_id]} {probs[_yawn_class_id]:.2f}"
            color = (0, 0, 255)  # 红色表示“打哈欠”
        else:
            label = f"{yawn_model.model.names[_no_yawn_class_id]} {probs[_no_yawn_class_id]:.2f}"
            color = (0, 255, 0)

    # 4) 在主图上画 ROI 框和标签
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return is_yawn
