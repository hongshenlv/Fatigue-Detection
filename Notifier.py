# Notifier.py

import os
from pync import Notifier as MacNotifier

def notify():
    """
    当系统判断用户可能疲劳时，调用此函数向 macOS 发送系统通知，
    点击通知会自动打开摄像头可视化页面（http://127.0.0.1:5000/）。
    """
    link = "http://127.0.0.1:5000/"

    title   = "疲 劳 提 醒"
    message = "检测到您可能疲劳了，请休息一下！点击查看实时监测状态。"

    MacNotifier.notify(
        message,
        title=title,
        open=link,
        sound="default"
    )

    # 发送完通知后，把这条通知移除（清理内存）
    MacNotifier.remove(os.getpid())
