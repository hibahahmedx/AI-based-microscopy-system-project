#code for the early detection of leukemia
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import plot_model
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
normal_dataset_path = r'F:\bloodcells_dataset'
all_original_path = r'F:\ALL Dataset/Original'
all_segmented_path = r'F:\ALL Dataset/Segmented'
batch_size = 32
img_size = (150, 150)

data_gen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
all_train_gen = data_gen.flow_from_directory(
    'F:\ALL Dataset/Original',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',  # Change this for multi-class classification
    subset='training'
)
all_val_gen = data_gen.flow_from_directory(
    'F:\ALL Dataset/Original',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)
print("Class indices (training):", all_train_gen.class_indices)
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(5, activation='softmax')  # Change to 5 output classes
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
epochs = 10
history = model.fit(
    all_train_gen,
    epochs=epochs,
    validation_data=all_val_gen
)
batch_size = 32
img_size = (150, 150)
data_gen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
all_train_gen = data_gen.flow_from_directory(
    'F:\ALL Dataset/Original',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',  # Change this for multi-class classification
    subset='training'
)
all_val_gen = data_gen.flow_from_directory(
    'F:\ALL Dataset/Original',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)
print("Class indices (training):", all_train_gen.class_indices)
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
class_names = ['Benign', 'Early', 'Preleukemic', 'Progressive', 'Proliferative']
def predict_leukemia(img_path):
    img = image.load_img(img_path, target_size=img_size)  
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    predicted_class_idx = np.argmax(prediction)
    predicted_class = class_names[predicted_class_idx] 
    plt.imshow(img)
    plt.axis('off')  # Hide axes
    plt.title(f"Predicted: {predicted_class}", fontsize=14)
    plt.show()
    if predicted_class in class_names:
        return f"Test is Positive, Patient is diagnosed with Leukemia ({predicted_class})"
    else:
        return "Test is Negative, Patient is healthy"
test_img_path = r"F:\ALL Dataset\Original\Benign\WBC-Benign-002.jpg"  # Replace with the actual path of the image
print(predict_leukemia(test_img_path))
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy')
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss')
plt.show()
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

class_names = ['Benign', 'Early', 'Preleukemic', 
'Progressive', 'Proliferative']

def predict_leukemia(img_path):
    img = image.load_img(img_path, target_size=img_size)  
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)
    predicted_class_idx = np.argmax(prediction)
    predicted_class = class_names[predicted_class_idx]
    
    if predicted_class in class_names:
        return f"Positive for Leukemia ({predicted_class})"
   
 else:
        return "Negative for Leukemia"

def upload_and_predict():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if not file_path:
        return 
    
    img = Image.open(file_path)
    img_resized = img.resize((250, 250))  # Resize for display
    img_tk = ImageTk.PhotoImage(img_resized)
    img_label.config(image=img_tk)
    img_label.image = img_tk
    
    prediction = predict_leukemia(file_path)
    result_label.config(text=prediction)

root = tk.Tk()
root.title("Leukemia Detection")
root.geometry("400x500")

title_label = tk.Label(root, text="Leukemia Detection System", font=("Arial", 16))
title_label.pack(pady=10)

img_label = tk.Label(root)
img_label.pack(pady=10)

upload_button = tk.Button(root, text="Upload Image", command=upload_and_predict, font=("Arial", 12))
upload_button.pack(pady=10)

result_label = tk.Label(root, text="", font=("Arial", 14), fg="blue")
result_label.pack(pady=20)
root.mainloop()
