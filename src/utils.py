import cv2
import torch
from facenet_pytorch import MTCNN
import matplotlib.pyplot as plt
import os



def process(file_path,model ,output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(file_path)
    output_path = f"{output_folder}/{os.path.basename(file_path)}"
    if not cap.isOpened():
        print(f"Lỗi: Không thể mở file {file_path}")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (width, height)

    print(f"fps: {fps}, frame_size: {frame_size}")

    # Thiết lập codec và VideoWriter trước vòng lặp
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    while True:
        ret, frame = cap.read()
        if not ret:
            break



        video_writer.write(frame)  # Ghi frame vào video đầu ra

    cap.release()
    video_writer.release()
    print(f"Xử lý hoàn tất {os.path.basename(file_path)}")


if __name__ == "__main__":
    process("../videos/809286291090666.mp4","../videos/output")
