import cv2
from ultralytics import YOLO
from src.ultis.get_resource_path import ResourcePath

# Load YOLO model
model = YOLO('../../assets/best.pt')


def draw_boxes(frame, results):
    """
    Vẽ bounding box và label lên frame
    """
    if results and len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes

        for box in boxes:
            # Lấy tọa độ box
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Lấy confidence và class
            conf = box.conf[0].item()
            class_id = int(box.cls[0].item())

            # Lấy tên class
            class_name = results[0].names[class_id]

            # Chọn màu box dựa trên confidence
            color = (0, 255, 0)  # Xanh lá - confidence cao

            # Vẽ bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Tính toán kích thước text
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(class_name, font, font_scale, thickness)

            # Vẽ background cho text
            cv2.rectangle(frame,
                          (x1, y1 - text_height - baseline - 5),
                          (x1 + text_width, y1),
                          color, -1)

            # Vẽ text
            cv2.putText(frame, class_name, (x1, y1 - baseline - 5),
                        font, font_scale, (255, 255, 255), thickness)

    return frame


def main():
    """
    Hàm main để test YOLO với webcam
    """
    print("=== YOLO Plant Disease Detection - Webcam Test ===")
    print("Hướng dẫn:")
    print("- Nhấn 'q' để thoát")
    print("- Nhấn 's' để lưu ảnh hiện tại")
    print("- Nhấn 'p' để pause/resume")
    print("- Màu box: Xanh lá (conf > 0.8), Vàng (conf > 0.6), Đỏ (conf <= 0.6)")
    print("Đang khởi động webcam...")

    # Khởi tạo webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Lỗi: Không thể mở webcam!")
        return

    # Thiết lập độ phân giải webcam
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    paused = False
    frame_count = 0
    detection_count = 0

    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("Không thể đọc frame từ webcam")
                    break

                frame_count += 1

                # Predict với YOLO
                results = model.predict(frame)

                # Vẽ bounding boxes
                frame = draw_boxes(frame, results)

                # Đếm số detection
                if results and len(results) > 0 and results[0].boxes is not None:
                    num_detections = len(results[0].boxes)
                    if num_detections > 0:
                        detection_count += 1

                        # Hiển thị thông tin detection
                        info_text = f"Detections: {num_detections}"
                        cv2.putText(frame, info_text, (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Hiển thị thông tin frame
                status_text = f"Frame: {frame_count} | Detections: {detection_count}"
                cv2.putText(frame, status_text, (10, frame.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Hiển thị frame
            window_title = "YOLO Plant Disease Detection"
            if paused:
                window_title += " - PAUSED"

            cv2.imshow(window_title, frame)

            # Xử lý phím nhấn
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("Thoát chương trình...")
                break
            elif key == ord('s'):
                # Lưu ảnh hiện tại
                filename = f"detection_frame_{frame_count}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Đã lưu ảnh: {filename}")
            elif key == ord('p'):
                paused = not paused
                status = "PAUSED" if paused else "RESUMED"
                print(f"Trạng thái: {status}")

    except KeyboardInterrupt:
        print("\nChương trình bị dừng bởi người dùng")

    except Exception as e:
        print(f"Lỗi: {e}")

    finally:
        # Giải phóng tài nguyên
        cap.release()
        cv2.destroyAllWindows()
        print(f"Đã xử lý {frame_count} frames với {detection_count} detections")
        print("Kết thúc chương trình")


if __name__ == "__main__":
    main()