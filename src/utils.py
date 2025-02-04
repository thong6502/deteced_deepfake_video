import cv2
import torch
from facenet_pytorch import MTCNN
import matplotlib.pyplot as plt
import os
from .model import MyConvNeXt
from .dataset import MyDataset
from .config import TRANSFORM
from PIL import Image
import torch.nn.functional as F

def get_classes():
    # Lấy đường dẫn tuyệt đối của file hiện tại
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Xây dựng đường dẫn tuyệt đối tới thư mục dataset
    dataset_path = os.path.join(current_dir, '..', 'dataset224')

    # Khởi tạo MyDataset với đường dẫn tuyệt đối
    classes = MyDataset(dataset_path).classes
    return classes

def detect_face(frame):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    mtcnn = MTCNN(
        min_face_size=50,
        margin=14,
        factor=0.6,
        thresholds=[0.9, 0.9, 0.9],
        keep_all=True,
        device=device
    )

    boxes, _ = mtcnn.detect(frame)
    return boxes

def load_model(check_point_path):
    check_point = torch.load(check_point_path)
    model = MyConvNeXt(2)
    model.load_state_dict(check_point["model"])
    return model

def process(file_path, model, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(file_path)
    output_path = os.path.join(output_folder, os.path.basename(file_path))
    if not cap.isOpened():
        print(f"Lỗi: Không thể mở file {file_path}")
        return
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    classes = get_classes()
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (width, height)

    print(f"fps: {fps}, frame_size: {frame_size}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes = detect_face(frame_rgb)

        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                face_crop = frame_rgb[y1:y2, x1:x2]
                if face_crop.shape[0] == 0 or face_crop.shape[1] == 0:
                    print(f"x1={x1}, y1={y1}, x2={x2}, y2={y2}")
                    print("Cảnh báo: Ảnh crop không hợp lệ với kích thước 0.")
                    continue  # hoặc xử lý theo cách thích hợp

                face_crop = Image.fromarray(face_crop)  # Chuyển đổi numpy.ndarray -> PIL.Image
                face_crop = TRANSFORM(face_crop).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    outputs = model(face_crop)
                    probs = F.softmax(outputs, dim=1)
                    preds = torch.argmax(probs, dim=1)

                confidence = probs[0, preds.item()].item() * 100  # Lấy xác suất và chuyển thành %
                label = f"{classes[preds.item()]}: {confidence:.2f}%"  # Định dạng nhãn hiển thị

                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)


        video_writer.write(frame)  # Ghi frame trước khi resize

        # scale_percent = 50
        # width = int(frame.shape[1] * scale_percent / 100)
        # height = int(frame.shape[0] * scale_percent / 100)
        # frame_resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        # cv2.imshow("Frame", frame_resized)
        # if cv2.waitKey(1) & 0xFF == ord("q"):
        #     break

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()
    print(f"Xử lý hoàn tất {os.path.basename(file_path)}")
    return output_path