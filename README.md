# Face Landmarks Detection 🎭

* This project demonstrates how to detect **facial landmarks** (such as eyes, nose, and mouth positions) using **Deep Learning with PyTorch**. 

* Inspired by applications like Snapchat filters, this project trains a deep neural network to identify **68 key facial landmarks** from images.

---

## 📌 Project Overview
- Uses the **iBUG 300-W Large Face Landmark Dataset** containing 6,666+ annotated face images.  
- Implements **data preprocessing and augmentation** (rotation, cropping, resizing, jitter).  
- Defines a custom **PyTorch Dataset class** for handling face landmark annotations.  
- Trains a **ResNet-18 model** (modified for grayscale input and 68×2 landmark outputs).  
- Evaluates the model on unseen validation data and visualizes predictions.  

---

## 📂 Dataset
The project uses the official **DLIB iBUG 300-W dataset**:

```bash
wget http://dlib.net/files/data/ibug_300W_large_face_landmark_dataset.tar.gz
tar -xvzf ibug_300W_large_face_landmark_dataset.tar.gz
rm ibug_300W_large_face_landmark_dataset.tar.gz
```
### Dataset includes:

1. Annotated .pts and .xml files containing landmark coordinates.

2. Images with bounding boxes for cropping faces.

## 🚀 Training Pipeline

1. Preprocessing & Augmentation

2. Crop face using bounding box

3. Resize to (224 × 224)

4. Apply random rotation, color jitter, normalization

5. Dataset Splitting

6. 90% Training (~6000 images)

7. 10% Validation (~666 images)

## Model

- Backbone: ResNet18

- Modified conv1 to accept 1-channel (grayscale) images

- Modified fc layer → output size = 68 × 2 = 136 landmarks

## Training

- Loss: MSELoss

- Optimizer: Adam (lr=1e-4)

- Epochs: 2000

- Saves best model (face_landmarks.pth) when validation loss improves

## 📊 Results & Visualization

- During training, both train loss and validation loss are logged.
Sample prediction visualization:

- Green Dots → Ground Truth Landmarks

- Red Dots → Predicted Landmarks

## 🖼️ Output

- Model predicts accurate facial keypoints for unseen images.

- Useful for AR applications (Snapchat filters, face swaps, emotion detection).