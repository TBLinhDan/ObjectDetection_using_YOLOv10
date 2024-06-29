# I. **Object Detection with YOLOv10 for a Image:**

**Sử dụng YOLOv10 với pre-trained models _ yolov10n.pt, tự động xác định vị trí của các đối tượng trong một tấm ảnh làm đầu vào, đầu ra trả về Tọa độ (bounding box) của các các đối tượng.**

**1. Tải mã nguồn YOLOv10 từ GitHub:**  
Trên Google Colab, khởi tạo một code cell sử dụng lệnh:  
```
!git clone https://github.com/THU-MIG/yolov10.git
```
(refresh lại phần Files của Google Colab để xem thư mục YOLOv10 đã xuất hiện hay chưa).

**2. Cài đặt và import các thư viện cần thiết để sử dụng được YOLOv10:**
```
%cd yolov10
!pip install -q -r requirements.txt
!pip install -e .
```
**3. Tải trọng số của pre-trained models, yolov10n.pt, đã được huấn luyện trên bộ dữ liệu COCO.**
```
!wget https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10n.pt
```
(refresh phần Files của Google Colab và tìm kiếm file có tên YOLOv10n.pt)  

**Khởi tạo mô hình YOLOv10 với phiên bản nano (n) từ trọng số đã tải về**

```
from ultralytics import YOLOv10
MODEL_PATH = 'yolov10n.pt'
model = YOLOv10(MODEL_PATH)
```
**4.Test mô hình trên một ảnh bất kì. Tải ảnh cần dự đoán:**  
```
! gdown '1tr9PSRRdlC2pNir7jsYugpSMG-7v32VJ' -O './images/' # HCMC_Street.jpg
```

**5. Để chạy dự đoán cho ảnh đã tải về, truyền đường dẫn ảnh vào mô hình**

```
IMG_PATH = './images/HCMC_Street.jpg'
result = model(source = IMG_PATH)[0]
```
**6. Để lưu lại ảnh đã được dự đoán**
```
result.save('./images/HCMC_Street_predict.png')
```

**7. Hiển thị kết quả ảnh dự đoán**
```
import cv2
import matplotlib.pyplot as plt

image_predict = cv2.imread('./images/HCMC_Street_predict.png')
plt.imshow(cv2.cvtColor(image_predict, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
```

# **II. Helmet Safety Detection with YOLOv10:**
**Huấn luyện YOLOv10 (fine-tuning) trên tập dữ liệu Helmet Safety Detection, sử dụng pre-trained models _ yolov10n.pt  
Thực hiện Kiểm tra việc tuân thủ đội mũ bảo vệ theo phương thức Object Detection: tự động xác định vị trí của các đối tượng trong một tấm ảnh làm Input, Output trả về Tọa độ (bounding box) của các nhân viên và phần mũ bảo hiểm.**

**1. Tải mã nguồn YOLOv10 từ GitHub:**  
Trên Google Colab, khởi tạo một code cell sử dụng lệnh:  
```
!git clone https://github.com/THU-MIG/yolov10.git
```
(refresh lại phần Files của Google Colab để xem thư mục YOLOv10 đã xuất hiện hay chưa).

**2. Cài đặt và import các thư viện cần thiết để sử dụng được YOLOv10:**
```
%cd yolov10
!pip install -q -r requirements.txt
!pip install -e .
```
**3. Tải trọng số của pre-trained models, yolov10n.pt, đã được huấn luyện trên bộ dữ liệu COCO.**
```
!wget https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10n.pt
```
(refresh phần Files của Google Colab và tìm kiếm file có tên YOLOv10n.pt)  

**Khởi tạo mô hình YOLOv10 với phiên bản nano (n) từ trọng số đã tải về**

```
from ultralytics import YOLOv10
MODEL_PATH = 'yolov10n.pt'
model = YOLOv10(MODEL_PATH)
```
**4. Tải bộ dữ liệu về Helmet Safety Detection. Giải nén bộ dữ liệu vào folder datasets**  
```
!gdown '1twdtZEfcw4ghSZIiPDypJurZnNXzMO7R'
!mkdir safety_helmet_dataset
!unzip -q './Safety_Helmet_Dataset.zip' -d './safety_helmet_dataset'
```
(Bộ dữ liệu này đã được gán nhãn và đưa vào format cấu trúc dữ liệu training theo yêu cầu của YOLO (Test-Train-Value: images & labels). Vì vậy, sẽ không cần thực hiện bước chuẩn bị dữ liệu (như việc gán nhãn thủ công mẫu dữ liệu sử dụng labelImg).  

**5. Tiến hành huấn luyện YOLOv10 trên bộ dữ liệu Helmet Safety Detection:**  

```
YAML_PATH = '../safety_helmet_dataset/data.yaml'
EPOCHS = 50 	         # Số lần lặp qua bộ dữ liệu trong quá trình huấn luyện.
IMG_SIZE = 640	       # Kích thước ảnh training, mặc định là 640
BATCH_SIZE = 256/128/64  # bộ dữ liệu train sẽ được chia ra thành các batch có 256/128/64 mẫu dữ liệu

model.train(data = YAML_PATH,
	        epochs = EPOCHS,		
	        batch = BATCH_SIZE,
	        imgsz = IMG_SIZE)
```

**6. Thực hiện đánh giá mô hình trên tập test**
```
TRAINED_MODEL_PATH = 'runs/detect/train/weights/best.pt'
model = YOLOv10(TRAINED_MODEL_PATH)

model.val(data = YAML_PATH,
	      imgsz = IMG_SIZE,
	      split ='test')
```

**7. Hiển thị kết quả dự đoán:**  
```
from google.colab.patches import cv2_imshow

TRAINED_MODEL_PATH = 'runs/detect/train/weights/best.pt'
model = YOLOv10(TRAINED_MODEL_PATH)

IMAGE_URL = 'https://ips-dc.org/wp-content/uploads/2022/05/Black-Workers-Need-a-Bill-of-Rights.jpeg'
CONF_THRESHOLD = 0.3
results = model.predict(source=IMAGE_URL,
                       imgsz=IMG_SIZE,
                       conf=CONF_THRESHOLD)
annotated_img = results[0].plot()

cv2_imshow(annotated_img)
```

