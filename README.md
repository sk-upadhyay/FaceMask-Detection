# Face Mask Detection

## Overview
This project implements a real-time face mask detection system using deep learning. It consists of a Convolutional Neural Network (CNN) trained to classify faces as wearing a mask or not, and a live video processing script that detects faces and applies the model in real-time.

## Features
- Real-time face detection using MTCNN
- CNN-based mask classification
- Live video feed from webcam
- Visual feedback with bounding boxes and labels

## Requirements
- Python 3.x
- TensorFlow/Keras
- OpenCV
- MTCNN
- NumPy
- Pillow (for notebook)
- Matplotlib (for notebook)
- Scikit-learn (for notebook)

Install dependencies:
```
pip install tensorflow opencv-python mtcnn numpy pillow matplotlib scikit-learn
```

## Dataset
The model is trained on the Face Mask Dataset from Kaggle (https://www.kaggle.com/omkargurav/face-mask-dataset).
- 3725 images with mask
- 3828 images without mask

## Training the Model
Run the `FaceMask_Detection.ipynb` notebook to train the model:
1. Download the dataset from Kaggle (requires kaggle.json API key)
2. Preprocess images (resize to 128x128, normalize)
3. Build and train the CNN model
4. Evaluate performance
5. Save the model as `mask_detector.keras`

Note: The notebook is designed for Google Colab environment.

## Running Real-time Detection
Execute the `main.py` script:
```
python main.py
```
- Opens webcam feed
- Detects faces using MTCNN
- Classifies each face as "Mask" or "No Mask"
- Displays results with bounding boxes
- Press 'q' to quit

## Model Architecture
- Input: 128x128 RGB images
- Conv2D (32 filters, 3x3) -> MaxPool (2x2)
- Conv2D (64 filters, 3x3) -> MaxPool (2x2)
- Flatten -> Dense (128) -> Dropout (0.5)
- Dense (64) -> Dropout (0.5)
- Dense (2) with sigmoid activation
- Optimizer: Adam
- Loss: Sparse Categorical Crossentropy

## Files
- `FaceMask_Detection.ipynb`: Training notebook
- `main.py`: Real-time detection script
- `mask_detector.keras`: Trained model
- `haarcascade_frontalface_default.xml`: Unused Haar cascade (MTCNN used instead)
- `README.md`: This file

## How It Works
1. **Face Detection**: MTCNN identifies faces in each video frame with high confidence (>0.9)
2. **Preprocessing**: Extract face ROI, resize to 128x128, normalize pixel values
3. **Classification**: Feed preprocessed image to CNN model
4. **Output**: Display prediction ("Mask" or "No Mask") above each face

## Performance
- Trained for 5 epochs on ~6800 images
- Training Accuracy: Starts at approximately 83% and reaches about 94% after 5 epochs
- Validation Accuracy: Starts at around 90% and stabilizes at approximately 92% after 5 epochs
- Test accuracy: Likely around 90-92% based on validation performance
- Real-time processing suitable for standard webcams

## Future Improvements
- Train on larger/more diverse dataset
- Implement additional face attributes detection
- Optimize for mobile deployment
- Add audio alerts or logging

