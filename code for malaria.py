#code for MALARIA detection test
import os, glob
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import Callback, EarlyStopping
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.metrics import classification_report
file_path = (r"C:\Users\STAR\malaria detection\cell_images\cell_images")
name_class = os.listdir(file_path)
name_class
filepaths = list(glob.glob(file_path+'/**/*.*'))
filepaths

labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1],filepaths))
labels

filepath = pd.Series(filepaths, name = 'Filepath').astype(str)
labels = pd.Series(labels, name = 'Label')
data = pd.concat([filepath,labels], axis = 1)
data = data.sample(frac=1).reset_index(drop = True)
data.head(5)

counts = data.Label.value_counts()
sns.barplot(x=counts.index, y=counts)
plt.xlabel('Type')
plt.xticks(rotation = 90);
train,test = train_test_split(data, test_size = 0.25, random_state = 42)
print(len(train))
print(len(test))

fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(10,8), subplot_kw={'xticks':[], 'yticks':[]})
for i, ax in enumerate(axes.flat):
    ax.imshow(plt.imread(data.Filepath[i]))
    ax.set_title(data.Label[i])
plt.tight_layout()
plt.show()
train_datagen = ImageDataGenerator(preprocessing_function = preprocess_input,
                                   validation_split=0.2)
test_datagen = ImageDataGenerator(preprocessing_function = preprocess_input)
print(len(train))
print(len(test))
train_gen = train_datagen.flow_from_dataframe(
    dataframe=train,
    x_col='Filepath',
    y_col='Label',
    target_size=(100,100),
    class_mode='categorical',
    batch_size=32,
    shuffle=True,
    seed = 42
)
valid_gen = train_datagen.flow_from_dataframe(
    dataframe=test,
    x_col='Filepath',
    y_col='Label',
    target_size=(100,100),
    class_mode='categorical',
    batch_size=32,
    shuffle=False,
    seed = 42
)
test_gen = train_datagen.flow_from_dataframe(
    dataframe=test,
    x_col='Filepath',
    y_col='Label',
    target_size=(100,100),
    class_mode='categorical',
    batch_size=32,
    shuffle=False,
    seed = 42
)
    import tensorflow as tf
pretrained_model = tf.keras.applications.mobilenet.MobileNet(
    input_shape=(100,100,3), alpha=1.0,depth_multiplier=1,dropout=0.001,
    include_top=False, weights='imagenet', input_tensor=None, pooling='max',
    classes=1000, classifier_activation='softmax'
)
pretrained_model.trainable= False
inputs = pretrained_model.input
x = Dense(128, activation='relu')(pretrained_model.output)
x = Dense(128, activation='relu')(x)
outputs = Dense(2, activation='softmax')(x)
model = Model(inputs=inputs, outputs=outputs)
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
my_callbacks = [EarlyStopping(monitor='val_accuracy',
                              min_delta=0,
                              patience=2,
                              mode='auto')]
history = model.fit(
    train_gen,
    validation_data = valid_gen,
    epochs = 10,
    callbacks=my_callbacks
)
model.save('model_mobilenet.h5')
pd.DataFrame(history.history)[['accuracy','val_accuracy']].plot()
plt.title('Accuracy')
plt.show()
pd.DataFrame(history.history)[['loss','val_loss']].plot()
plt.title('Loss')
plt.show()
results = model.evaluate(test_gen, verbose=0)
print('    Test Loss: {:5f}'.format(results[0]))
print('Test Accuracy: {:2f}%'.format(results[1]*100))
pred = model.predict(test_gen)
pred = np.argmax(pred,axis=1)
labels = (train_gen.class_indices)
labels = dict((v,k) for k,v in labels.items())
pred = [labels[k] for k in pred]
y_test = list(test.Label)
y_test = y_test[:len(pred)]
print(classification_report(y_test, pred))
fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(12,8), subplot_kw={'xticks':[], 'yticks':[]})
for i, ax in enumerate(axes.flat):
    ax.imshow(plt.imread(test.Filepath.iloc[i]))
    ax.set_title(f"True: {test.Label.iloc[i]}\nPredicted: {pred[i]}")
plt.tight_layout()
plt.show()
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=test.Label.unique(), yticklabels=test.Label.unique())
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

import customtkinter as ctk
from tkinter import filedialog, messagebox
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image, ImageTk

model = load_model('model_mobilenet.h5')
root = ctk.CTk()
root.title("Malaria Detection Application")
root.geometry("600x500")

frame = ctk.CTkFrame(root)
frame.pack(pady=20, padx=40, fill="both", expand=True)

title_label = ctk.CTkLabel(frame, text="Final Year Project", font=("Arial", 30, "bold"), text_color="darkblue")
title_label.pack(pady=10)
title_label = ctk.CTkLabel(frame, text="Malaria Detection 
Application using CNN", font=("Arial", 24, "bold"), text_color="darkblue")
title_label.pack(pady=10)
image_label = ctk.CTkLabel(frame, text="Upload an 

Image", width=300, height=200, corner_radius=10, bg_color="lightgray")
image_label.pack(pady=20)
uploaded_image = None
def upload_image():
    """Open file dialog to upload an image"""
    global uploaded_image
    file_path = 

filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        img = Image.open(file_path)
        img = img.resize((300, 200))  # Resize the image to fit the label

        uploaded_image = img  # Store the uploaded image
        img = ImageTk.PhotoImage(img)
        image_label.configure(image=img, text="")  # Update the label with the image
        image_label.image = img

upload_button = ctk.CTkButton(frame, text="Upload Image", command=upload_image)
upload_button.pack(pady=10)

def predict():
    """Predict if the image is infected with malaria"""
    global uploaded_image
    if uploaded_image:

        img = uploaded_image.convert("RGB")
        img = img.resize((100, 100))  # Resize to (100, 100) 

as expected by the model
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

preds = model.predict(img_array)
        class_idx = np.argmax(preds, axis=1)[0]
        if class_idx == 0:
            result = "Plasmodium Detected: Infected"
            messagebox.showinfo("Prediction", result)

        else:
            result = "No Infection Detected"
            messagebox.showinfo("Prediction", result)
   
 else:
        messagebox.showerror("Error", "Please upload an image first!")

predict_button = ctk.CTkButton(frame, text="Predict", command=predict)

predict_button.pack(pady=10)
root.mainloop()
