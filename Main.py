import cv2
import numpy as np
import os
import time
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn import svm, neighbors, tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Hàm tiền xử lý ảnh và trích xuất đặc trưng bằng HOG
def preprocess_image(image_path, img_size=(32, 32)):
    # Đọc ảnh và chuyển thành ảnh xám
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, img_size)

    # Trích xuất đặc trưng HOG
    features, hog_image = hog(
        img,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        visualize=True,
        feature_vector=True
    )

    return features  # Trả về vector đặc trưng HOG

# Hàm tải ảnh và gán nhãn
def load_data(dataset_path):
    X, y = [], []
    if not os.path.exists(dataset_path):
        raise ValueError(f"Path '{dataset_path}' does not exist.")
    class_names = os.listdir(dataset_path)
    
    # In ra tên các thư mục con
    print(f"Found classes: {class_names}")
    
    for label, class_name in enumerate(class_names):
        class_folder = os.path.join(dataset_path, class_name)
        
        # Kiểm tra nếu là thư mục
        if os.path.isdir(class_folder):
            image_count = 0
            for filename in os.listdir(class_folder):
                img_path = os.path.join(class_folder, filename)
                
                # Kiểm tra nếu là tệp hình ảnh
                if os.path.isfile(img_path) and filename.endswith(('.jpg', '.png')):
                    img_features = preprocess_image(img_path)  # Trích xuất đặc trưng
                    X.append(img_features)
                    y.append(class_name)
                    image_count += 1
            print(f"Loaded {image_count} images from class '{class_name}'.")
        else:
            print(f"Skipping {class_folder} as it is not a directory.")
    
    # Kiểm tra số lượng ảnh đã tải
    if len(X) == 0 or len(y) == 0:
        raise ValueError("No images found. Please check the dataset path or ensure the directory contains images.")

    return np.array(X), np.array(y)

# Đọc tập dữ liệu ảnh
dataset_path = 'Dataset\\archive\\animal_dataset_intermediate\\train'  # Thay bằng đường dẫn của bạn
X, y = load_data(dataset_path)

# In số lượng ảnh đã tải để kiểm tra
print(f"Total images loaded: {len(X)}")
print(f"Total labels loaded: {len(y)}")

# Mã hóa nhãn (gán nhãn cho chuỗi thành số)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Chia tập dữ liệu thành tập huấn luyện và tập kiểm thử
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)


# Danh sách các mô hình
models = {
    "SVM": svm.SVC(kernel='linear'),
    "KNN": neighbors.KNeighborsClassifier(n_neighbors=3),
    "Decision Tree": tree.DecisionTreeClassifier()
}

# Đánh giá mô hình và lưu kết quả
results = []

for name, model in models.items():
    # Đo thời gian huấn luyện
    start_time = time.time()
    model.fit(X_train, y_train)
    end_time = time.time()

    # Dự đoán và tính toán độ chính xác, precision, recall
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')

    # Lưu kết quả
    results.append({
        "Model": name,
        "Training Time (s)": end_time - start_time,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
    })

    # In báo cáo phân loại cho từng mô hình
    print(f"\nClassification Report for {name}:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

# Hiển thị bảng kết quả so sánh các mô hình
results_df = pd.DataFrame(results)
print("\nComparison of Models:")
print(results_df)
