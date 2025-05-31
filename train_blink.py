import os
from ultralytics import YOLO

def main():
    """
    �� Ultralytics YOLOv8 �� classification ģʽ
    ѵ����/���۶�����ģ�͡�
    """
    # 1. ȷ����Ŀ��Ŀ¼
    project_root = os.path.dirname(os.path.abspath(__file__))

    # 2. ����Ԥѵ��Ȩ�أ��������ص���Ŀ¼��
    cls_model = os.path.join(project_root, "yolov8n-cls.pt")
    if not os.path.isfile(cls_model):
        print(f"Error: δ�ҵ�����Ȩ���ļ� '{cls_model}'�������غ������С�")
        return

    # 3. ����·���� dataset.yaml
    data_cfg = os.path.join(project_root, "dataset.yaml")
    if not os.path.isfile(data_cfg):
        print(f"Error: δ�ҵ����ݼ������ļ� '{data_cfg}'��")
        return

    # 4. ���ط���ģ��
    model = YOLO(cls_model)

    # 5. ����ѵ��
    model.train(
        data=os.path.join(project_root, "dataset"),  # ֱ��ָ�� dataset �ļ���
        task="classify",     # classification ģʽ
        epochs=100,
        batch=64,
        optimizer="Adam",
        lr0=0.01,
        device=0,            # GPU 0������ GPU �ĳ� "cpu"
        verbose=True
    )

if __name__ == "__main__":
    main()
