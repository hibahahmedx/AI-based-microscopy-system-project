#code for the graphical user interface of Smartcope project
import numpy as np
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QPushButton, QFileDialog, QHBoxLayout, QRadioButton, QGroupBox
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt
import sys

class BloodCellClassifier:
    def __init__(self, test_type="CBC"):
        self.test_type = test_type

    def classify_and_annotate(self, img):
        if self.test_type == "Malarial":
            return self.malarial_detection(img)
        elif self.test_type == "Leukemia":
            return self.leukemia_detection(img)
        else:
            return self.cbc_detection(img)

    def malarial_detection(self, img):
circular objects in the RBC
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        counts = {"Malaria Parasite": 0}
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            if area < 200:  # Classifying small objects (could 

represent parasites)
                label = "Malaria Parasite"
                color = (255, 0, 0)  # Red for parasites
                counts["Malaria Parasite"] += 1
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return img, counts
    def leukemia_detection(self, img):
        # Dummy logic for leukemia detection: detecting large abnormal cells (blasts)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        counts = {"Leukemia Blasts": 0}
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            if area > 3000:  # Classifying large cells (could 

represent blasts)
                label = "Leukemia Blasts"
                color = (0, 255, 255)  # Yellow for blasts
                counts["Leukemia Blasts"] += 1
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return img, counts

    def cbc_detection(self, img):
WBCs, and Platelets
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        counts = {"RBC": 0, "WBC": 0, "Platelet": 0}
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h

            if area < 500:
                label = "Platelet"
                color = (255, 255, 0)  # Yellow for platelets
                counts["Platelet"] += 1

            elif area < 2000:
                label = "RBC"
                color = (0, 255, 0)  # Green for RBCs
                counts["RBC"] += 1

            else:
                label = "WBC"
                color = (0, 0, 255)  # Red for WBCs
                counts["WBC"] += 1
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y - 10), 

cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return img, count

class ImageBrowser(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Blood Cell Detection: Malarial, 

Leukemia, CBC')
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
        self.test_group = QGroupBox("Select Test")
        self.layout.addWidget(self.test_group)
        test_layout = QVBoxLayout()
        self.malarial_radio = QRadioButton("Malarial 

Detection")
        self.malarial_radio.setChecked(True)
        test_layout.addWidget(self.malarial_radio)
        self.leukemia_radio = QRadioButton("Leukemia Detection")
        test_layout.addWidget(self.leukemia_radio)
        self.cbc_radio = QRadioButton("CBC (Complete 

Blood Count)")
        test_layout.addWidget(self.cbc_radio)
        self.test_group.setLayout(test_layout)
        # Buttons layout
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

        self.classifier = None
        self.image_paths = []
        self.current_index = 0

    def load_folder(self):
        folder = QFileDialog.getExistingDirectory(self, 'Select Image Folder', os.getcwd())
        if folder:
            self.image_paths = [os.path.join(folder, f)
                                for f in os.listdir(folder)
                                if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

            self.current_index = 0
            self.show_image()

    def show_image(self):
        if not self.image_paths:
            return
        path = self.image_paths[self.current_index]
        img = cv2.imread(path)

        if self.malarial_radio.isChecked():
            self.classifier = 

BloodCellClassifier(test_type="Malarial")
        elif self.leukemia_radio.isChecked():
            self.classifier = 

BloodCellClassifier(test_type="Leukemia")
        else:
            self.classifier = BloodCellClassifier(test_type="CBC")

        annotated_img, counts = self.classifier.classify_and_annotate(img)
        height, width, channels = annotated_img.shape
        bytes_per_line = channels * width

        qimg = QImage(annotated_img.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
self.image_label.setPixmap(QPixmap.fromImage(qimg).scaledToWidth(700))

        self.info_label.setText(f"Image {self.current_index + 1}/{len(self.image_paths)} — Counts: {counts}")
    def next_image(self):

        if self.image_paths and self.current_index + 1 < len(self.image_paths):
            self.current_index += 1
            self.show_image()
if __name__ == '__main__':

    app = QApplication(sys.argv)    
window = ImageBrowser()
    window.show()
    sys.exit(app.exec_())
