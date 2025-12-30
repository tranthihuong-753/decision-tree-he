# Sử dụng resnes50 để chuyẻn đổi img sang vector và giảm chiều bằng GAP global average pooling 
# Giảm từ (1, 7, 7, 2048) về (1, 2048)
# PCA Giảm thành (1, 100)
# Nhãn vẫn để dạng string 

from PIL import Image
import numpy as np
import pandas as pd
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from sklearn.decomposition import PCA
import joblib

# Đường dẫn lưu file .csv sau khi tiền xử lý
path_preprocessed_csv = 'C:/hu/decision-tree-he/Build_Project/Release/cats_vs_dogs_dataset_resnet_100.csv' #file.csv lưu đầu ra 
path_datatrain = 'Datatrain/cats_vs_dogs_dataset' #Foulder đọc img 
path_pcs= 'pca_cats_vs_dogs_dataset_resnet_100.pkl' #file.pkl lưu pca để dùng cho img test 
n=100 # Số chiều sử dụng PCA 

# Tải mô hình ResNet50 đã được huấn luyện trên ImageNet
# include_top=False: loại bỏ lớp phân loại cuối cùng, giữ lại các lớp trích xuất đặc trưng
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Định nghĩa mô hình trích xuất đặc trưng (chỉ bao gồm các lớp tích chập của ResNet50)
feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)

def extract_features_resnet50(image_path):
    """
    Trích xuất đặc trưng ảnh sử dụng ResNet50
    Input path_img, Output vector 1 chiều (2048 features)
    Tải ảnh và thay đổi kích thước về 224x224 (yêu cầu của ResNet50)
    """
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize((224, 224))
    except Exception as e:
        print(f"Lỗi tải ảnh {image_path}: {e}")
        return None

    # 2. Chuyển ảnh thành mảng numpy
    img_array = image.img_to_array(img) # Kích thước (224, 224, 3)

    # 3. Thêm chiều batch (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0) 

    # 4. Chuẩn hóa giá trị pixel theo yêu cầu của ResNet50 (từ Keras)
    # Hàm preprocess_input sẽ chuẩn hóa các kênh màu dựa trên ImageNet.
    img_array = preprocess_input(img_array)
    
    # 5. Trích xuất đặc trưng
    features = feature_extractor.predict(img_array)

    # 6. THAY ĐỔI: Áp dụng Global Average Pooling (GAP) để giảm về 2048 features
    # Kích thước 'features' là (1, 7, 7, 2048). np.mean tính trung bình trên trục 1 và 2 (chiều cao và chiều rộng)
    # Giảm từ (1, 7, 7, 2048) về (1, 2048)
    features_pooled = np.mean(features, axis=(1, 2))

    # 7. Làm phẳng vector (từ (1, 2048) thành (2048,))
    img_one = features_pooled.flatten()
    print(f"Kích thước vector đặc trưng ResNet50 (sử dụng GAP): {img_one.shape} (2048 features)")

    return img_one


def reduce_features_pca(features, n_components=n):
    """
    Giảm chiều 
    features: numpy array (N, 2048)
    return: reduced (N, 100), pca_model
    """
    print("reduce_features_pca()")
    print("Đang chạy PCA giảm chiều...")

    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(features)

    print("completed reduce_features_pca()")
    print("Reduced shape:", reduced.shape)

    return reduced, pca

# Lưu pca để predict 
def save_pca_model(pca, output_path=path_pcs):
    joblib.dump(pca, output_path)
    print("Đã lưu PCA model tại:", output_path)

#Áp dụng PCA cho ảnh mới (predict)
def extract_and_reduce_single_image_pca(image_path, pca):
    feat2048 = extract_features_resnet50(image_path)
    reduced100 = pca.transform([feat2048])[0]
    return reduced100

# Hàm lưu mảng đã tiền xử lý vào file .csv với nhiều ảnh 1 lúc, input path + label 
def save_preprocessed_data_to_csv(image_paths, labels, csv_path):
    print("save_preprocessed_data_to_csv()")
    features_2048 = []
    for idx, path in enumerate(image_paths):
        feat = extract_features_resnet50(path)
        features_2048.append(feat)
        print(f"- {idx+1}/{len(image_paths)} done")
    
    features_2048 = np.array(features_2048)  # (N,2048)

    # ====== PCA duy nhất 1 lần ======
    features_100, pca = reduce_features_pca(features_2048)
    save_pca_model(pca)

    # ====== Ghép nhãn ======
    data_with_label = np.hstack([features_100, np.array(labels).reshape(-1, 1)])

    # ====== Lưu CSV ======
    df = pd.DataFrame(data_with_label)
    df.to_csv(csv_path, index=False, header=False)

    print(f"Lưu CSV thành công: {csv_path}")
    print(f"Tổng mẫu: {len(data_with_label)}")

# Hàm đọc lấy path tự động từ folder 
def get_image_paths_and_labels(folder_path):
    image_paths = []
    labels = []
    for label in os.listdir(folder_path):
        label_folder = os.path.join(folder_path, label)
        if os.path.isdir(label_folder):
            for img_file in os.listdir(label_folder):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(os.path.join(label_folder, img_file))
                    labels.append(label)
    return image_paths, labels

image_paths, labels = get_image_paths_and_labels(path_datatrain)
save_preprocessed_data_to_csv(image_paths, labels, path_preprocessed_csv)

