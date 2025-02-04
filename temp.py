from src.utils import load_model, detect_face, process, get_classes
import cv2 

def show_face_detect():
  frame = cv2.imread("images/123.jpg")
  frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  boxes = detect_face(frame)

  if boxes is not None:
    for box in boxes:
        x, y, w, h = box
        x, y, w, h = int(x), int(y), int(w), int(h)
        cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)

  # Hiển thị ảnh với các bounding boxes đã vẽ
  frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  cv2.imshow("Detected Faces", frame)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

if __name__ == "__main__":
  # show_face_detect()
  model = load_model("trained_model/best.pt")
  process("videos/fake1.mp4",model,"videos/output")
  # print(get_classes())
  pass