
# 🧠 Parkinson's Disease Prediction from Audio

🚀 **A machine learning-powered application for detecting Parkinson's Disease using audio recordings and deep learning models.**

---

## 📂 Project Structure

```
📂 Parkinson_Audio_Detection
│── main.py             # Main application script
│── model.py            # Neural network model definition
│── preprocess.py       # Feature extraction and data preprocessing
│── dataset(name as singam)/            # Audio dataset (healthy & Parkinson samples)
│── requirements.txt    # Dependencies list
│── README.md           # Project documentation

```

---

## 🎯 Project Overview

This project utilizes **machine learning** and **deep learning** techniques to detect Parkinson’s Disease from voice recordings. It processes **biomedical voice features** like MFCCs, Chroma, Mel Spectrogram, and Spectral Contrast to classify audio samples into *Healthy* or *Parkinson's Disease*. The model is deployed with **Streamlit** for user-friendly predictions.

### 🚀 Key Features
✔️ **Biomedical voice analysis for accurate Parkinson’s detection**  
✔️ **Neural network-based classification model** for high precision  
✔️ **Feature extraction using MFCC, Chroma, Mel Spectrogram, and Spectral Contrast**  
✔️ **User-friendly web interface for uploading audio or recording speech**  
✔️ **Real-time prediction of Parkinson’s Disease presence**  

---

## 🛠️ Installation & Setup

### 1️⃣ Clone the Repository
```sh
git clone https://github.com/yourusername/Parkinson_Audio_Detection.git  
cd Parkinson_Audio_Detection  
```

### 2️⃣ Create a Virtual Environment (Optional but Recommended)
```sh
python -m venv env  
# Activate on Mac/Linux  
source env/bin/activate  
# Activate on Windows  
env\Scripts\activate  
```

### 3️⃣ Install Dependencies
```sh
pip install -r requirements.txt  
```

---

## 📊 Parkinson's Detection Workflow

### 🔍 1. Data Collection
- Voice samples from **healthy individuals** and **Parkinson’s patients** are stored in separate directories.
- The dataset is sourced from research databases or clinical recordings.

### 🔬 2. Feature Extraction
- Extract **MFCCs (Mel-Frequency Cepstral Coefficients)**, **Chroma**, **Mel Spectrogram**, and **Spectral Contrast**.
- Features are **flattened and padded** to a fixed size.
- Standardized using **scikit-learn’s StandardScaler**.

### 🤖 3. Model Training
- A **Deep Neural Network (DNN)** with 4 hidden layers is trained.
- Uses **Batch Normalization, Dropout, and ReLU activation**.
- Optimized using **Adam optimizer** with **binary cross-entropy loss**.

### ✅ 4. Model Testing & Evaluation
- Evaluated using **accuracy, precision, recall, and F1-score**.
- Trained model is stored and used for predictions.

### 📌 5. Real-Time Prediction (Streamlit Interface)
- **Upload an audio file** (wav, mp3, flac) to get a prediction.
- **Record live speech** using a microphone for diagnosis.
- **Outputs “Healthy” or “Parkinson Present”** based on model inference.

---

## 🚀 How to Run the Application

### 1️⃣ Run the Streamlit App
```sh
streamlit run main.py  
```

### 2️⃣ Upload an Audio File or Record Speech
- Click the **“Choose an audio file”** button to upload a recording.
- Click **“Click to Record Speech”** to record live voice.
- The model will predict whether Parkinson’s Disease is present.

---

## 📦 Dependencies (`requirements.txt`)
```txt
numpy  
pandas  
librosa  
tensorflow  
scikit-learn  
streamlit  
speechrecognition  
pyaudio   
```
```sh
pip install -r requirements.txt
 
```
## Output:
## Image 1

![image](https://github.com/user-attachments/assets/035b4d69-2aad-4e22-a4d2-cee4926090e7)

## Image 2
![image](https://github.com/user-attachments/assets/5a7e4fd9-5154-44a6-935d-26f5f8afa4b1)


## Image 3 (using Real time Speech)
![image](https://github.com/user-attachments/assets/88597379-d0e3-4ca2-8a52-09647591ed33)

---



