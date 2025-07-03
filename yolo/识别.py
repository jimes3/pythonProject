import cv2
from ultralytics import YOLO

# 加载模型（可以使用官方模型或者你自己训练的模型）
model = YOLO('yolov8n.pt')  # 更换成你训练的手势识别模型路径，比如 'best.pt'

# 打开摄像头（0 是默认摄像头）
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("无法打开摄像头")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 推理
    results = model(frame)

    # 可视化检测结果
    annotated_frame = results[0].plot()

    # 显示图像
    cv2.imshow("YOLOv8 手势识别", annotated_frame)

    # 按 q 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
