from Blink_detection.blink_detection import classify_eyes, classify_yawn
import Notifier
from flask import Flask, Response, render_template
import threading, datetime, time, cv2

app = Flask(__name__)

# 全局输出帧（即将被推送到 /video_feed）+ 线程锁
outputFrame = None
lock = threading.Lock()

# ———— 窗口内“眨眼”和“打哈欠”计数（用于逻辑判断） ————
logic_blinks = 0     # 用于眨眼阈值判断
logic_yawns  = 0     # 用于打哈欠阈值判断

# ———— 界面显示的“当前一分钟内眨眼/打哈欠”计数 ————
display_blinks = 0   # 只在一分钟到期时重置
display_yawns  = 0   # 只在一分钟到期时重置

# 当前是否处于“疲劳”状态（此标志只在报警瞬间为 True，随后恢复为 False）
is_fatigued = False

# 打开摄像头（macOS 指定 AVFoundation 后端）
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
if not cap.isOpened():
    raise RuntimeError("无法打开摄像头，请检查权限与设备索引")


@app.route("/")
def index():
    return render_template("index.html")


def detect_blinks_and_yawns():
    """
    后台线程：循环读取摄像头，分别做眨眼 & 打哈欠分类，
    - 眨眼在每分钟末尾判断(< 12 次就报警，并重置窗口)；
    - 打哈欠只要本分钟累计 > 3 次，立即报警并清除本分钟的打哈欠计数（逻辑清零），
      但不重置眨眼计数或窗口起始时间。
    - 报警后 is_fatigued 先置为 True，显示一两帧“WARNING”，下次循环自动恢复 False。
    """
    global outputFrame, lock
    global logic_blinks, logic_yawns
    global display_blinks, display_yawns
    global is_fatigued

    time.sleep(1.0)  # 给摄像头 warm‐up

    # ———— 阈值设置，可根据需要调整 ————
    BLINK_ALERT_THRESHOLD = 12   # 一分钟内眨眼数 < 12 就视为“疲劳”
    YAWN_ALERT_THRESHOLD  = 3    # 一分钟内打哈欠数 > 3 就视为“疲劳”

    # “窗口”起始时刻、以及该窗口内的计数
    minute_window_start = datetime.datetime.now()
    # logic_blinks、logic_yawns 由全局变量维护，初始为 0
    # display_blinks、display_yawns 也由全局变量维护，初始为 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.05)
            continue

        # ———— 1. 把帧 resize 到宽度 700，高度等比 ————
        h, w = frame.shape[:2]
        frame = cv2.resize(frame, (700, int(700 * h / w)))

        # ———— 2. 眨眼分类 & 计数 ————
        left_closed, right_closed = classify_eyes(frame, conf_thres=0.3)
        curr_closed = left_closed and right_closed
        # 如果“上一帧是闭眼、当前帧变睁眼”，就算一次眨眼
        if curr_closed is False and left_closed == False and right_closed == False:
            #    这里假设 classify_eyes 返回 True 表示“闭眼”，
            #    False 表示“睁眼”。所以上一帧和当前帧的状态切换
            #    用 prev_closed 逻辑可实现，但这里简化为：只要 classify_eyes 本次 False
            #    但实际上需要有 prev_closed 记录——下方示例保留 prev_closed 。
            pass  # 下面会用 prev_closed 逻辑完成

        # 实际上一帧切换判定要用 prev_closed 变量。这里省略细节，直接调用上一版示例中
        # 的 prev_closed 逻辑（你需要先在外面声明 prev_closed = False）。

        # 但为了完整性，先声明 prev_closed（只在本函数上下文使用），并更新：
        if 'prev_closed' not in globals():
            globals()['prev_closed'] = False  # 第一次进入循环时，视为上一帧“睁眼”
        # “上一帧闭眼、当前帧睁眼”——记一次眨眼
        if globals()['prev_closed'] and not curr_closed:
            logic_blinks += 1
            display_blinks += 1
        globals()['prev_closed'] = curr_closed

        # ———— 3. 打哈欠分类 & 计数 ————
        is_yawn = classify_yawn(frame, conf_thres=0.3)
        if 'prev_yawn' not in globals():
            globals()['prev_yawn'] = False  # 第一次进入循环时，视为上一帧“未打哈欠”
        # “上一帧没打哈欠、当前帧开始打哈欠”——记一次打哈欠
        if not globals()['prev_yawn'] and is_yawn:
            logic_yawns += 1
            display_yawns += 1
        globals()['prev_yawn'] = is_yawn

        # ———— 4. “打哈欠”阈值判断（如果逻辑计数 > 阈值，就立即报警） ————
        now = datetime.datetime.now()
        elapsed = (now - minute_window_start).total_seconds()

        if logic_yawns > YAWN_ALERT_THRESHOLD:
            # 立即判“疲劳”，触发系统通知
            is_fatigued = True
            Notifier.notify()
            # 立刻把 logic_yawns 归零，但不改动 display_yawns，也不重置窗口时间
            logic_yawns = 0
        else:
            # 只有当打哈欠未超过阈值时，才去检查“眨眼一分钟末尾”逻辑
            if elapsed >= 60:
                # —— 本分钟到期，检查“眨眼”阈值 ———
                if logic_blinks < BLINK_ALERT_THRESHOLD:
                    is_fatigued = True
                    Notifier.notify()
                else:
                    is_fatigued = False

                # —— 重置整分钟窗口 ———
                minute_window_start = now
                logic_blinks = 0
                logic_yawns  = 0
                display_blinks = 0
                display_yawns  = 0

        # —— 5. 在画面上叠加文字信息 ———
        # 界面上显示“本窗口（当前分钟）累计眨眼次数”、“本窗口累计打哈欠次数”
        cv2.putText(frame, f"Blinks: {display_blinks}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Yawns: {display_yawns}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 显示当前“闭眼/睁眼”状态
        state_eye = "Closed" if curr_closed else "Open"
        cv2.putText(frame, f"Eye: {state_eye}", (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        # 显示当前“打哈欠/无哈欠”状态
        mouth_txt = "Yawning" if is_yawn else "No Yawn"
        cv2.putText(frame, f"Mouth: {mouth_txt}", (300, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 显示“疲劳”或“正常”状态。由于 is_fatigued 只在报警瞬间被置 True，
        # 报警完成后下一帧会自动恢复 False，因此 WARNING 文字仅闪现一两帧：
        if is_fatigued:
            cv2.putText(frame, "WARNING: FATIGUE", (200, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
            # 画面闪现一次后，立即清零 is_fatigued，让下帧恢复正常显示
            is_fatigued = False
        else:
            cv2.putText(frame, "STATUS: NORMAL", (200, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 3)

        # —— 6. 更新全局输出帧 ———
        with lock:
            outputFrame = frame.copy()


def generate():
    """
    Flask 生成图片流（multipart/x-mixed-replace）函数，
    - 每次输出最新的 outputFrame。
    """
    global outputFrame, lock
    while True:
        with lock:
            if outputFrame is None:
                continue
            flag, encoded = cv2.imencode(".jpg", outputFrame)
            if not flag:
                continue
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" +
               bytearray(encoded) +
               b"\r\n")


@app.route("/video_feed")
def video_feed():
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


# 在 Flask 第一次请求到来之前，就启动后台检测线程
@app.before_first_request
def start_capture():
    t = threading.Thread(target=detect_blinks_and_yawns)
    t.daemon = True
    t.start()


if __name__ == "__main__":
    app.run(debug=True, threaded=True)
