import sys
import cv2
import face_recognition
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QFileDialog, QHBoxLayout
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt

class FaceRecognitionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        
        self.img1 = None
        self.img2 = None

    def initUI(self):
        self.setWindowTitle('Face Recognition')
        self.setGeometry(100, 100, 600, 400)
        self.setStyleSheet("background-color: #1c1c1c; color: white; font-family: Arial;")

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)

        self.label = QLabel("Upload two images for face recognition", self)
        self.label.setFont(QFont("Arial", 14))
        layout.addWidget(self.label)

        button_layout = QHBoxLayout()

        self.btn1 = QPushButton('Upload Image 1', self)
        self.btn1.setStyleSheet("background-color: #007acc; color: white; border-radius: 10px; padding: 10px;")
        self.btn1.clicked.connect(self.upload_image1)
        button_layout.addWidget(self.btn1)

        self.btn2 = QPushButton('Upload Image 2', self)
        self.btn2.setStyleSheet("background-color: #007acc; color: white; border-radius: 10px; padding: 10px;")
        self.btn2.clicked.connect(self.upload_image2)
        button_layout.addWidget(self.btn2)

        layout.addLayout(button_layout)

        self.compare_btn = QPushButton('Compare Faces', self)
        self.compare_btn.setStyleSheet("background-color: #28a745; color: white; border-radius: 10px; padding: 10px;")
        self.compare_btn.clicked.connect(self.compare_faces)
        layout.addWidget(self.compare_btn)

        self.result_label = QLabel("", self)
        self.result_label.setFont(QFont("Arial", 12))
        layout.addWidget(self.result_label)

        self.setLayout(layout)

    def upload_image1(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Image 1", "", "Images (*.png *.jpg *.jpeg);;All Files (*)", options=options)
        if file_name:
            self.img1 = face_recognition.load_image_file(file_name)
            self.label.setText(f"Image 1 uploaded: {file_name.split('/')[-1]}")

    def upload_image2(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Image 2", "", "Images (*.png *.jpg *.jpeg);;All Files (*)", options=options)
        if file_name:
            self.img2 = face_recognition.load_image_file(file_name)
            self.label.setText(f"Image 2 uploaded: {file_name.split('/')[-1]}")

    def compare_faces(self):
        if self.img1 is None or self.img2 is None:
            self.result_label.setText("Please upload both images.")
            return

        # Convert images to RGB
        img1_rgb = cv2.cvtColor(self.img1, cv2.COLOR_BGR2RGB)
        img2_rgb = cv2.cvtColor(self.img2, cv2.COLOR_BGR2RGB)

        # Find face encodings
        face_enc1 = face_recognition.face_encodings(img1_rgb)
        face_enc2 = face_recognition.face_encodings(img2_rgb)

        if not face_enc1 or not face_enc2:
            self.result_label.setText("No faces found in one of the images.")
            return

        # Compare faces
        face_comp = face_recognition.compare_faces([face_enc1[0]], face_enc2[0])
        face_dis = face_recognition.face_distance([face_enc1[0]], face_enc2[0])
        percentage = (1 - face_dis) * 100

        self.result_label.setText(f"Faces are similar: {face_comp[0]} \nFace match percentage: {percentage[0]:.2f}%")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = FaceRecognitionApp()
    window.show()
    sys.exit(app.exec_())