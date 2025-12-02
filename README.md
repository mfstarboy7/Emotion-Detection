# Emotion Detection System

A Deep Learning project that detects human emotions in real-time using a webcam feed. The system utilizes a Convolutional Neural Network (CNN) built with TensorFlow/Keras to classify faces into one of seven emotions.

## 📋 Features
- **Real-time Detection:** Captures video from the default webcam and predicts emotions on the fly.
- **7 Emotion Classes:** Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise.
- **Face Detection:** Uses OpenCV's Haar Cascade Classifiers to isolate faces before processing.
- **Customizable Training:** Includes Jupyter Notebooks for training the model on custom datasets (e.g., FER-2013).

## 🛠️ Tech Stack
- **Python 3.x**
- **TensorFlow & Keras** (Deep Learning Model)
- **OpenCV** (Image processing and Video Capture)
- **NumPy** (Numerical operations)
- **Matplotlib** (Visualization during training)

## ⚙️ Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd Emotion-Detection

2. **Create a virtual environment (Recommended):**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate

3. **Install dependencies: The required libraries are listed in requirements.txt.**
   ```bash
   pip install -r requirements.txt

# 🚀 Usage

 **1. Real-time Emotion Detection**
  
    To start the webcam feed and detect emotions:
    
    1. Ensure you have the trained model file (e.g., emotion_detection_model.h5) in the project directory.

    2.Run the main script:
    python main.py

    3. Press 'q' to quit the video window.

       Note: The main.py script looks for a model file. Ensure the filename in the script (line 6) matches your actual model filename (e.g., emotion_detection_model.h5).

  **2. Training the Model**

  If you wish to train the model yourself, you can use the provided notebooks.

- Dataset Structure: Ensure your dataset is organized as follows:
   ```bash
   dataset/
   ├── train/
   │   ├── Angry/
   │   ├── Happy/
   │   └── ...
   └── test/
       ├── Angry/
       ├── Happy/
       └── ...   
**- Notebooks:**

 - Emotion_Detection_System.ipynb: Designed for Google Colab. Includes steps to upload datasets and mount Google Drive.

 - main.ipynb: Designed for Local Training. Uses data augmentation and trains a Sequential CNN model

# 📂 Project Structure
    ```bash
    Emotion-Detection/
    ├── Emotion_Detection_System.ipynb  # Colab notebook for training (200 epochs)
    ├── main.ipynb                      # Local notebook for training (50 epochs)
    ├── main.py                         # Real-time inference script using Webcam
    ├── requirements.txt                # List of python dependencies
    ├── test.py                         # Simple environment test script
    └── README.md                       # Project documentation

# 🧠 Model Architecture

The model processes grayscale images resized to 48x48 pixels. It consists of a Sequential CNN architecture:

 1. **Convolutional Layers:** Extract features using filters (32, 64, 128 filters).

 2. **Max Pooling:** Reduces spatial dimensions.

 3. **Dropout:** Prevents overfitting.

 4. **Dense Layers:** Fully connected layers for classification.

 5. **Output Layer:** Softmax activation with 7 units (one for each emotion).

# 📊 Results

- **Training Accuracy:** ~88% (varies by epoch count)

- **Validation/Test Accuracy:** ~55% - 60% on the test set.




