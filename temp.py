import cv2
import torch
from facenet_pytorch import MTCNN
import matplotlib.pyplot as plt

class FastMTCNN:
    """Fast MTCNN implementation without multithreading."""
    
    def __init__(self, stride, filename, resize=224, *args, **kwargs):
        self.stride = stride
        self.resize = resize
        self.filename = filename
        self.faces = []
        self.boxs = []
        self.mtcnn = MTCNN(*args, **kwargs)

    def __call__(self):
        """Detect faces in frames using strided MTCNN."""
        # Mở video và xử lý từng khung hình
        v_cap = cv2.VideoCapture(self.filename)
        frame_idx = 0

        while True:
            ret, frame = v_cap.read()
            if not ret:
                break

            if frame_idx % self.stride == 0:
                # Chuyển frame sang RGB để phát hiện khuôn mặt
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Phát hiện khuôn mặt
                boxes, _ = self.mtcnn.detect(frame_rgb)

                if boxes is not None:
                    for box in boxes:
                        # Ép kiểu tọa độ thành số nguyên và đảm bảo không âm
                        box = [max(0, int(b)) for b in box]
                        face = frame[box[1]:box[3], box[0]:box[2]]

                        if face.size == 0:
                            continue

                        # Resize khuôn mặt về kích thước mong muốn
                        face = cv2.resize(face, (self.resize, self.resize))
                        # Chuyển từ BGR sang RGB để hiển thị bằng matplotlib
                        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

                        self.faces.append(face)
                        self.boxs.append(box)

            frame_idx += 1

        v_cap.release()
        return self.boxs, self.faces

def detect_and_display_faces():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Đường dẫn đến video
    video_path = "../videos/809286291090666.mp4"
    
    # Khởi tạo đối tượng FastMTCNN với các tham số cần thiết
    fast_mtcnn = FastMTCNN(
        stride=10,
        resize=224,
        filename=video_path,
        min_face_size=50,
        margin=14,
        factor=0.6,
        thresholds=[0.9, 0.9, 0.9],
        keep_all=True,
        device=device
    )

    # Phát hiện khuôn mặt
    boxs, faces = fast_mtcnn()
    print(f"Detected {len(faces)} faces.")


    len_face = 10  # Số khuôn mặt muốn hiển thị

    for i, face in enumerate(faces[:len_face]):
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        cv2.imshow(f"face{i}: ", face)
        cv2.waitKey(0)
if __name__ == '__main__':
    detect_and_display_faces()

