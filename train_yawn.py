import os
from ultralytics import YOLO

def main():
    """
    使用 Ultralytics YOLOv8 分类模式训练 ‘打哈欠’ 二分类模型
    """

    # 1. 确定项目根目录
    project_root = os.path.dirname(os.path.abspath(__file__))

    # 2. 分类预训练权重（请先从 https://github.com/ultralytics/assets/releases 下载到此目录）
    #    通常文件名为 yolov8n-cls.pt 或 yolov8s-cls.pt 等，根据显存、需求自行选择一个较小的分类预训练权重
    cls_weights = os.path.join(project_root, "yolov8n-cls.pt")
    if not os.path.isfile(cls_weights):
        print(f"Error: 未在 {project_root} 下找到分类预训练权重 'yolov8n-cls.pt'，请先下载后再运行。")
        return

    # 3. 绝对路径的 dataset.yaml （在上一步中已创建）
    data_cfg = os.path.join(project_root, "dataset.yaml")
    if not os.path.isfile(data_cfg):
        print(f"Error: 未在 {project_root} 下找到数据集配置文件 'dataset.yaml'。")
        return

    # 4. 加载 YOLOv8 分类模型
    #    这里把 task 指定为“分类模式”（classification）
    model = YOLO(cls_weights)

    # 5. 启动训练
    #    - data: dataset.yaml 的路径（路径里会自动读取 train/val 子目录结构）
    #    - task: "classify" 表示分类模式
    #    - epochs: 训练轮数，可按需修改
    #    - batch: 批大小，建议根据你的 GPU/显存大小来设置
    #    - optimizer: 优化器
    #    - lr0: 初始学习率
    #    - device: 如果有 GPU，写 “0” 或 “cpu”；如多 GPU 可写 “0,1” 等
    model.train(
        data=os.path.join(project_root, "dataset"),
        task="classify",
        epochs=100,
        batch=64,
        optimizer="Adam",
        lr0=0.001,
        device="0",       # 如果没有 GPU，改成 "cpu"
        verbose=True
    )

if __name__ == "__main__":
    main()
