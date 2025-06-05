import cv2
from ultralytics import YOLO
from src.ultis.get_resource_path import ResourcePath

model = YOLO(ResourcePath.MODEL_PATH)

disease_names = {
    "Mold": "Bệnh nấm mốc",
    "SpiderMite": "Nhện đỏ cắn lá",
    "Septoria": "Bệnh đốm Septoria",
    "CurlVirus": "Bệnh virus xoăn lá",
    "EarlyBlight": "Bệnh thối rễ sớm",
    "LateBlight": "Bệnh thối rễ muộn",
    "TargetSpot": "Bệnh đốm mục tiêu",
    "BacterialSpot": "Bệnh đốm vi khuẩn",
}

disease_descriptions = {
    "Bệnh nấm mốc": "Bệnh nấm mốc là một bệnh do nấm gây ra, thường xuất hiện trong điều kiện ẩm ướt. Lá cây sẽ xuất hiện các mảng mốc trắng hoặc xám, làm giảm khả năng quang hợp và có thể gây chết cây nếu không xử lý kịp thời.",
    "Nhện đỏ cắn lá": "Nhện đỏ là loài côn trùng gây hại bằng cách hút nhựa lá, khiến lá cây bị chuyển màu vàng, khô héo và rụng sớm. Chúng sinh sản rất nhanh trong thời tiết nóng và khô.",
    "Bệnh đốm Septoria": "Bệnh đốm Septoria là do nấm Septoria lycopersici gây ra, tạo ra các đốm nâu tròn với viền sẫm trên lá già. Bệnh làm giảm diện tích lá quang hợp và lan nhanh nếu không được kiểm soát.",
    "Bệnh virus xoăn lá": "Bệnh virus xoăn lá gây ra hiện tượng lá bị xoăn, biến dạng và còi cọc. Virus thường lây qua côn trùng như bọ phấn trắng. Bệnh ảnh hưởng nghiêm trọng đến sự sinh trưởng và năng suất cây trồng.",
    "Bệnh thối rễ sớm": "Bệnh thối rễ sớm do nấm Alternaria solani gây ra, thường bắt đầu bằng những đốm nâu ở phần gốc thân và rễ, khiến cây dễ bị đổ ngã và chết sớm. Bệnh phát triển mạnh trong điều kiện ẩm và nóng.",
    "Bệnh thối rễ muộn": "Bệnh thối rễ muộn do nấm Phytophthora infestans gây ra, làm cho thân và rễ cây bị úng nước, chuyển màu đen và mềm nhũn. Bệnh thường xuất hiện sau khi có mưa lớn hoặc tưới quá nhiều.",
    "Bệnh đốm mục tiêu": "Bệnh đốm mục tiêu tạo ra các vết đốm tròn giống như đích ngắm trên lá, gây ra bởi nấm Corynespora cassiicola. Lá bệnh sẽ vàng và rụng sớm, làm giảm năng suất cây trồng.",
    "Bệnh đốm vi khuẩn": "Bệnh đốm vi khuẩn gây ra bởi vi khuẩn *Xanthomonas campestris, tạo ra các vết đốm nhỏ màu nâu sẫm, có viền vàng trên lá và quả. Bệnh có thể lan rộng nhanh chóng qua nước mưa hoặc tưới tiêu.",
}


def yolo_prediction(image):
    """
    Dự đoán bệnh cây từ ảnh đầu vào.

    Args:
        image (numpy.ndarray): Ảnh đầu vào đã được tiền xử lý (BGR format).

    Returns:
        str: Kết quả dự đoán.
        str: Mô tả bệnh.
        numpy.ndarray: Hình ảnh đã được vẽ hộp giới hạn và nhãn dự đoán.
    """
    print("Đang dự đoán bệnh cây...")

    name = "Cây khỏe mạnh"
    description = "Cây của bạn đang trong tình trạng khỏe mạnh, không có dấu hiệu bệnh tật."

    # Tạo bản sao của image để vẽ
    result_image = image.copy()

    try:
        results = model.predict(image)

        if results and len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
            # Lấy box có confidence cao nhất
            best_box = None
            best_conf = 0

            for box in results[0].boxes:
                conf = box.conf[0].item()
                if conf > best_conf:
                    best_conf = conf
                    best_box = box

            if best_box is not None:
                # Vẽ bounding box
                x1, y1, x2, y2 = map(int, best_box.xyxy[0])
                cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Lấy tên class
                class_id = int(best_box.cls[0].item())
                label = results[0].names[class_id]
                name = disease_names.get(label, "Cây khỏe mạnh")
                description = disease_descriptions.get(name, "Không có mô tả cho bệnh này.")

                # Vẽ label
                label_text = f"{name} ({best_conf:.2f})"
                cv2.putText(result_image, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                print(f"Phát hiện bệnh: {name} với độ tin cậy: {best_conf:.2f}")
            else:
                print("Không phát hiện bệnh nào")
        else:
            print("Không phát hiện đối tượng nào trong ảnh")

    except Exception as e:
        print(f"Lỗi trong quá trình dự đoán: {e}")
        name = "Lỗi dự đoán"
        description = "Có lỗi xảy ra trong quá trình dự đoán."

    print(f"Kết quả dự đoán: {name}")
    return name, description, result_image

