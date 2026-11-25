#code for image pre-processing
#augmenting and labeling images using yolo model
import os
label_map = {0: 1, 1: 0, 2: 2}
old_label_dir = r'C:/Users/STAR/Desktop/BCCD/labels'
new_label_dir = r'C:\Users\STAR\Desktop\TXL_PBC\labels'
for split in ['train', 'val', 'test']:
    input_dir = os.path.join(old_label_dir, split)
    output_dir = os.path.join(new_label_dir, split)
    os.makedirs(output_dir, exist_ok=True)
    for file in os.listdir(input_dir):
        if file.endswith('.txt'):
            with open(os.path.join(input_dir, file), 'r') as f:
                lines = f.readlines(
            new_lines = []
            for line in lines:
                parts = line.strip().split()
                cls = int(parts[0])
                new_cls = label_map.get(cls, cls)
                new_line = ' '.join([str(new_cls)] + parts[1:])
                new_lines.append(new_line)
            with open(os.path.join(output_dir, file), 'w') as f:
                f.write('\n'.join(new_lines))
import os
import xml.etree.ElementTree as ET
import cv2
def voc_to_yolo(voc_path, yolo_path, image_path):
    tree = ET.parse(voc_path)
    root = tree.getroot()
    img = cv2.imread(image_path)
    img_height, img_width, _ = img.shape
    yolo_annotations = []
    for obj in root.findall('object'):
        class_name = obj.find('name').text
        if class_name == 'RBC':
            class_id = 0
        elif class_name == 'WBC':
            class_id = 1
        elif class_name == 'Platelets':
            class_id = 2
        else:
            continue  
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        center_x = (xmin + xmax) / 2.0 / img_width
        center_y = (ymin + ymax) / 2.0 / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height
        yolo_annotations.append(f"{class_id} {center_x} {center_y} {width} {height}")
    yolo_file = os.path.join(yolo_path, os.path.basename(voc_path).replace('.xml', '.txt'))
    with open(yolo_file, 'w') as f:
        for line in yolo_annotations:
            f.write(f"{line}\n")
voc_annotations_folder = r'C:\Users\STAR\Desktop\BCCD\Annotations'  
images_folder = r'C:\Users\STAR\Desktop\BCCD\JPEGImages'  
yolo_annotations_folder = r'C:\Users\STAR\Desktop\BCCD\YOLO_annotations_folder' 
if not os.path.exists(yolo_annotations_folder):
    os.makedirs(yolo_annotations_folder)
for xml_file in os.listdir(voc_annotations_folder):
    if xml_file.endswith('.xml'):
        voc_file = os.path.join(voc_annotations_folder, xml_file)
        image_file = os.path.join(images_folder, xml_file.replace('.xml', '.jpg'))  
        voc_to_yolo(voc_file, yolo_annotations_folder, image_file)

print("Conversion to YOLO format complete.")
import os
import shutil
import random
from pathlib import Path
base_dir = Path(r"C:\Users\STAR\Desktop\BCCD")
img_dir = base_dir / "JPEGImages"
label_dir = base_dir / "YOLO_annotations_folder"
output_img_dir = base_dir / "images"
output_label_dir = base_dir / "labels"
for split in ['train', 'val', 'test']:
    (output_img_dir / split).mkdir(parents=True, exist_ok=True)
    (output_label_dir / split).mkdir(parents=True, exist_ok=True)
images = list(img_dir.glob("*.jpg"))
random.shuffle(images)
n_total = len(images)
train_split = int(n_total * 0.7)
val_split = int(n_total * 0.9)
splits = {
    'train': images[:train_split],
    'val': images[train_split:val_split],
    'test': images[val_split:]
}
for split, img_list in splits.items():
    for img_path in img_list:
        label_path = label_dir / (img_path.stem + ".txt")

        if not label_path.exists():
            print(f" Label missing for: {img_path.name}")
            continue
        shutil.copy(img_path, output_img_dir / split / img_path.name)
        shutil.copy(label_path, output_label_dir / split / label_path.name)
print("Dataset has been split and organized in YOLOv8 format.")
import cv2
import os
import random
import albumentations as A
from pathlib import Path
original_image_path = r"C:\Users\STAR\Desktop\TXL_PBC\images"
original_label_path = r"C:\Users\STAR\Desktop\TXL_PBC\labels"
augmented_image_path = r"C:\Users\STAR\Desktop\TXL_PBC\augmented_images\images"
augmented_label_path = r"C:\Users\STAR\Desktop\TXL_PBC\augmented_images\labels"
Path(augmented_image_path).mkdir(parents=True, exist_ok=True)
Path(augmented_label_path).mkdir(parents=True, exist_ok=True)
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=30, p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.RandomScale(scale_limit=0.2, p=0.3),
    A.augmentations.geometric.resize.Resize(width=640, height=480)
])
def load_image_and_labels(image_path, label_path):
    image = cv2.imread(image_path)
    with open(label_path, 'r') as file:
        labels = file.readlines()
    
    return image, labels
def save_augmented_image_and_labels(image, labels, image_save_path, label_save_path):
    cv2.imwrite(image_save_path, image)
    with open(label_save_path, 'w') as file:
        file.writelines(labels)
def augment_and_save_images(image_folder, label_folder, augmented_image_folder, augmented_label_folder, split='train', augment_count=10):
    image_files = os.listdir(image_folder)
    for img_file in image_files:
        if not img_file.endswith(('.jpg', '.png', '.jpeg')):
            continue
        img_name = img_file.split('.')[0]
        image_path = os.path.join(image_folder, img_file)
        label_path = os.path.join(label_folder, img_name + '.txt')
        image, labels = load_image_and_labels(image_path, label_path)
        for i in range(augment_count): 
            augmented = transform(image=image)
            augmented_image = augmented['image']
            augmented_image_save_path = os.path.join(augmented_image_folder, split, f"{img_name}_aug_{i+1}.jpg")
            augmented_label_save_path = os.path.join(augmented_label_folder, split, f"{img_name}_aug_{i+1}.txt")
save_augmented_image_and_labels(augmented_image, labels, augmented_image_save_path, augmented_label_save_path)
def main():
    splits = ['train', 'val', 'test']
    for split in splits:
        image_folder = os.path.join(original_image_path, split)
        label_folder = os.path.join(original_label_path, split)
        augmented_image_split_folder = os.path.join(augmented_image_path, split)
        augmented_label_split_folder = os.path.join(augmented_label_path, split)
 Path(augmented_image_split_folder).mkdir(parents=True, exist_ok=True)
        Path(augmented_label_split_folder).mkdir(parents=True, exist_ok=True)
        augment_and_save_images(image_folder, label_folder, augmented_image_path, augmented_label_path, split, augment_count=10)
if __name__ == "__main__":
    main()
    from ultralytics import YOLO
import yaml
import os
class_names = ['class_0', 'class_1']  
dataset_path = r'C:/Users/STAR/Desktop/TXL_PBC/augmented_images'
data_yaml = {
    'path': dataset_path.replace('\\', '/'),  
    'train': 'images/train',
    'val': 'images/val',
    'test': 'images/test', 
    'names': {i: name for i, name in enumerate(class_names)}
}
yaml_path = r'C:/Users/STAR/Desktop/TXL_PBC/data.yaml'
with open(yaml_path, 'w') as f:
    yaml.dump(data_yaml, f)
print(f" Created data.yaml at: {yaml_path}")
model = YOLO('yolov8n.pt')  
model.train(
    data=yaml_path,
    epochs=10,
    imgsz=320,
    batch=16,
    name='yolov8n_augmented',
    project=r'C:/Users/STAR/Desktop/TXL_PBC/yolo_runs'
)
print(" Training completed.")
from ultralytics import YOLO
model = YOLO(r"runs\detect\train7\weights\best.pt")
print(model.info)
from ultralytics import YOLO
import os
model = YOLO("runs/detect/train7/weights/best.pt")
dataset_path = r"C:\Users\STAR\Desktop\TXL_PBC\images\test"
results = model.predict(source=dataset_path, conf=0.25)
total_counts = {0: 0, 1: 0, 2: 0}  
for r in results:
    class_ids = r.boxes.cls.tolist()  
    for class_id in class_ids:
        total_counts[int(class_id)] += 1
class_names = model.names
print("\n Total Blood Cell Counts Across Dataset:")
for class_id, count in total_counts.items():
    print(f"{class_names[class_id]}: {count}")
from ultralytics import YOLO
import os
model = YOLO("runs/detect/train7/weights/best.pt")  
dataset_path = r"C:\Users\STAR\Desktop\TXL_PBC\images\test"
total_counts = {0: 0, 1: 0, 2: 0} 
for image_name in os.listdir(dataset_path):
    image_path = os.path.join(dataset_path, image_name)
    if image_path.endswith(('.jpg', '.png', '.jpeg')): 
        results = model.predict(source=image_path, conf=0.25)
          for r in results:
            class_ids = r.boxes.cls.tolist() 
            for class_id in class_ids:
                total_counts[int(class_id)] += 1  
class_names = model.names
print("\nTotal Blood Cell Counts Across the Entire Dataset:")
for class_id, count in total_counts.items():
    print(f"{class_names[class_id]}: {count}")
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
model = YOLO("runs/detect/train7/weights/best.pt")  
image_path = r"C:\Users\STAR\Desktop\TXL_PBC\images\test\0ee104a7-bfd9-529c-a525-85363b0ca3c3.jpg"
results = model.predict(source=image_path, conf=0.25)
for result in results:
    result.show()  
    image = result.orig_img  
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  
    plt.axis('off')  
    plt.show()
    print(f"Detected {len(result.boxes)} blood cells in image {image_path}:")
    for i, box in enumerate(result.boxes):
        class_id = int(box.cls.item()) 
        confidence = box.conf.item()  
        coords = box.xywh[0].numpy()  
        print(f"  {i+1}. Class: {model.names[class_id]}, Confidence: {confidence:.2f}, Coordinates: {coords}")
from ultralytics import YOLO
model = YOLO("runs/detect/train7/weights/best.pt")  
metrics = model.val()  
print(metrics)
import os
dataset_root = r'C:\Users\STAR\Desktop\TXL_PBC\images'
image_extensions = ['.jpg', '.jpeg', '.png']
image_counts = {}
for split in ['train', 'val', 'test']:
    split_path = os.path.join(dataset_root, split)
    count = 0
    if os.path.exists(split_path):
        for file in os.listdir(split_path):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                count += 1
        image_counts[split] = count
    else:
        image_counts[split] = 'Folder not found'
for split, count in image_counts.items():
    print(f"{split.capitalize()} images: {count}")
import cv2
import os
import random
import albumentations as A
from pathlib import Path
original_image_path = r"C:\Users\STAR\Desktop\TXL_PBC\images"
original_label_path = r"C:\Users\STAR\Desktop\TXL_PBC\labels"
augmented_image_path = r"C:\Users\STAR\Desktop\TXL_PBC\augmented_images\images"
augmented_label_path = r"C:\Users\STAR\Desktop\TXL_PBC\augmented_images\labels"
Path(augmented_image_path).mkdir(parents=True, exist_ok=True)
Path(augmented_label_path).mkdir(parents=True, exist_ok=True)
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=30, p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.RandomScale(scale_limit=0.2, p=0.3),
    A.augmentations.geometric.resize.Resize(width=640, height=480)
])
def load_image_and_labels(image_path, label_path):
    image = cv2.imread(image_path)
    with open(label_path, 'r') as file:
        labels = file.readlines()
    return image, labels
def save_augmented_image_and_labels(image, labels, image_save_path, label_save_path):
    cv2.imwrite(image_save_path, image)
    with open(label_save_path, 'w') as file:
        file.writelines(labels)
def augment_and_save_images(image_folder, label_folder, augmented_image_folder, augmented_label_folder, split='train', augment_count=10):
    image_files = os.listdir(image_folder)
    for img_file in image_files:
        if not img_file.endswith(('.jpg', '.png', '.jpeg')):
            continue
        img_name = img_file.split('.')[0]
        image_path = os.path.join(image_folder, img_file)
        label_path = os.path.join(label_folder, img_name + '.txt')
        image, labels = load_image_and_labels(image_path, label_path)       
        for i in range(augment_count): 
            augmented = transform(image=image)
            augmented_image = augmented['image']
            augmented_image_save_path = os.path.join(augmented_image_folder, split, f"{img_name}_aug_{i+1}.jpg")
            augmented_label_save_path = os.path.join(augmented_label_folder, split, f"{img_name}_aug_{i+1}.txt")
 save_augmented_image_and_labels(augmented_image, labels, augmented_image_save_path, augmented_label_save_path)
def main():
    splits = ['train', 'val', 'test']
    for split in splits:
        image_folder = os.path.join(original_image_path, split)
        label_folder = os.path.join(original_label_path, split)
        augmented_image_split_folder = os.path.join(augmented_image_path, split)
        augmented_label_split_folder = os.path.join(augmented_label_path, split)
 Path(augmented_image_split_folder).mkdir(parents=True, exist_ok=True)        Path(augmented_label_split_folder).mkdir(parents=True, exist_ok=True)
        augment_and_save_images(image_folder, label_folder, augmented_image_path, augmented_label_path, split, augment_count=10)
if __name__ == "__main__":
    main()
import os
def count_images_in_dataset(base_path):
    splits = ['train', 'val', 'test']
    image_counts = {
        'train': 0,
        'val': 0,
        'test': 0
    }
    for split in splits:
        image_folder = os.path.join(base_path, 'images', split)
        if os.path.exists(image_folder):
            image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
            image_counts[split] = len(image_files)
    
    else:
            print(f"Warning: {image_folder} does not exist!")
    return image_counts
def main():
    augmented_image_path = 

r"C:\Users\STAR\Desktop\TXL_PBC\augmented_images"
    counts = count_images_in_dataset(augmented_image_path)
    print("Image counts in the augmented dataset:")
    for split, count in counts.items():
        print(f"{split.capitalize()}: {count} images")
if __name__ == "__main__":
    main()
#code for blood cell counter
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QPushButton, QFileDialog
from PyQt5.QtGui import QPixmap, QImage
import sys

class BloodCellClassifier:
    def __init__(self):
        pass

    def classify_and_annotate(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            label = "RBC" if w * h < 1000 else "WBC"
            color = (0, 255, 0) if label == "RBC" else (0, 0, 255)

            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return img, {"RBC": 5, "WBC": 3, "Platelet": 1}  
class ImageBrowser(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Blood Cell Detection')
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.image_label = QLabel()
        self.layout.addWidget(self.image_label)

        self.info_label = QLabel("No image loaded")
        self.layout.addWidget(self.info_label)
        self.load_button = QPushButton('Load Folder')
        self.load_button.clicked.connect(self.load_folder)
        self.layout.addWidget(self.load_button)
        self.next_button = QPushButton('Next Image')
        self.next_button.clicked.connect(self.next_image)
        self.layout.addWidget(self.next_button)
        self.classifier = BloodCellClassifier()
        self.image_paths = []
        self.current_index = 0
    def load_folder(self):
        folder = QFileDialog.getExistingDirectory(self, 'Select Image Folder')

        if folder:
            self.image_paths = [os.path.join(folder, f)
                                for f in os.listdir(folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
            self.current_index = 0
            self.show_image()
    def show_image(self):
        if not self.image_paths:
            return
        path = self.image_paths[self.current_index]
        img = cv2.imread(path)
        annotated_img, counts = 

self.classifier.classify_and_annotate(img)
        height, width, channels = annotated_img.shape
        bytes_per_line = channels * width
        qimg = QImage(annotated_img.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped() self.image_label.setPixmap(QPixmap.fromImage(qimg).scaledToWidth(600))
        self.info_label.setText(f"Image {self.current_index+1}/{len(self.image_paths)} — Counts: {counts}")

    def next_image(self):
        if self.image_paths and self.current_index + 1 < len(self.image_paths):
            self.current_index += 1
            self.show_image()

app = QApplication(sys.argv)
window = ImageBrowser()
window.show()
sys.exit(app.exec_())

import os
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QPushButton, QFileDialog, QHBoxLayout
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt
import sys
class BloodCellClassifier:
    def __init__(self):
        pass
    def classify_and_annotate(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, 

cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        counts = {"RBC": 0, "WBC": 0, "Platelet": 0}
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h

            if area < 500:
                label = "Platelet"
                color = (255, 255, 0)  
                counts["Platelet"] += 1

            elif area < 2000:
                label = "RBC"
                color = (0, 255, 0)  
                counts["RBC"] += 1

            else:
                label = "WBC"
                color = (0, 0, 255)
                counts["WBC"] += 1
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y - 10), 

cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return img, counts

class ImageBrowser(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Blood Cell Detection')
        self.setFixedSize(800, 600) 
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.image_label)
        self.info_label = QLabel("No image loaded", self)
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setFont(QFont("Arial", 12))
        self.layout.addWidget(self.info_label)
        button_layout = QHBoxLayout()
        self.layout.addLayout(button_layout)
        self.load_button = QPushButton('Load Folder', self)
        self.load_button.setStyleSheet("background-color: 

#4CAF50; color: white; font-size: 16px; padding: 10px;")
        self.load_button.clicked.connect(self.load_folder)
        button_layout.addWidget(self.load_button)
        self.next_button = QPushButton('Next Image', self)
        self.next_button.setStyleSheet("background-color: 

#2196F3; color: white; font-size: 16px; padding: 10px;")
        self.next_button.clicked.connect(self.next_image)
        button_layout.addWidget(self.next_button)
        self.classifier = BloodCellClassifier()
        self.image_paths = []
        self.current_index = 0
    def load_folder(self):
        """Open a folder dialog and load all image files."""
        folder = QFileDialog.getExistingDirectory(self, 'Select 
Image Folder', os.getcwd())
        if folder:
            self.image_paths = [os.path.join(folder, f)
                                for f in os.listdir(folder)
                                if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            self.current_index = 0
            self.show_image()
    def show_image(self):
        """Display the image with cell count information."""
        if not self.image_paths:
            return
        path = self.image_paths[self.current_index]
        img = cv2.imread(path)
        annotated_img, counts = 
self.classifier.classify_and_annotate(img)
        height, width, channels = annotated_img.shape
        bytes_per_line = channels * width
        qimg = QImage(annotated_img.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()    self.image_label.setPixmap(QPixmap.fromImage(qimg).scaledToWidth(700))
        self.info_label.setText(f"Image {self.current_index + 1}/{len(self.image_paths)} — RBC: {counts['RBC']}, WBC: {counts['WBC']}, Platelets: {counts['Platelet']}")
    def next_image(self):
        """Load the next image in the folder."""
        if self.image_paths and self.current_index + 1 < len(self.image_paths):
            self.current_index += 1
            self.show_image()
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ImageBrowser()
    window.show()
    sys.exit(app.exec_())
