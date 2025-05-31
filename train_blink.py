import os
from ultralytics import YOLO

def main():
    """
    用 Ultralytics YOLOv8 的 classification 模式
    训练开/闭眼二分类模型。
    """
    # 1. 确定项目根目录
    project_root = os.path.dirname(os.path.abspath(__file__))

    # 2. 分类预训练权重（请先下载到此目录）
    cls_model = os.path.join(project_root, "yolov8n-cls.pt")
    if not os.path.isfile(cls_model):
        print(f"Error: 未找到分类权重文件 '{cls_model}'，请下载后再运行。")
        return

    # 3. 绝对路径的 dataset.yaml
    data_cfg = os.path.join(project_root, "dataset.yaml")
    if not os.path.isfile(data_cfg):
        print(f"Error: 未找到数据集配置文件 '{data_cfg}'。")
        return

    # 4. 加载分类模型
    model = YOLO(cls_model)

    # 5. 启动训练
    model.train(
        data=os.path.join(project_root, "dataset"),  # 直接指向 dataset 文件夹
        task="classify",     # classification 模式
        epochs=100,
        batch=64,
        optimizer="Adam",
        lr0=0.01,
        device=0,            # GPU 0；若无 GPU 改成 "cpu"
        verbose=True
    )

if __name__ == "__main__":
    main()
