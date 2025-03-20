import os
import librosa
import numpy as np
import tensorflow as tf
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import collections
import speech_recognition as sr  

# Function to extract features (MFCC, Chroma, Mel Spectrogram, Spectral Contrast)

def extract_features(file_path, max_pad_length=100):
    y, sr = librosa.load(file_path, sr=22050)
    
    
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    
    # Flatten and concatenate all features
    features = np.concatenate([mfcc.flatten(), chroma.flatten(), mel.flatten(), contrast.flatten()])
    
    
    pad_width = max_pad_length - features.shape[0]
    if pad_width > 0:
        features = np.pad(features, (0, pad_width), mode='constant')
    else:
        features = features[:max_pad_length]  # Trim to fixed length
    
    return features


data_dirs = {
    "healthy": r"E:\dverse\singam\healthy\healthy_in",  # Update with correct paths
    "parkinson": r"E:\dverse\singam\parkinson\parkinson_in"
}

# Load the dataset
data = []
labels = []

for label, directory in data_dirs.items():
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        try:
            feature = extract_features(file_path)
            data.append(feature)
            labels.append(label)
        except Exception as e:
            print(f"Error processing {file}: {e}")

# Convert to numpy arrays
data = np.array(data)
labels = np.array(labels)

# Check dataset balance
print("Dataset Distribution:", collections.Counter(labels))

# Encode labels
encoder = LabelEncoder()
labels = encoder.fit_transform(labels)  # 0 = Healthy, 1 = Parkinson

# Normalize features
scaler = StandardScaler()
data = scaler.fit_transform(data)

# Split dataset (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Define the neural network model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation='relu', input_shape=(data.shape[1],)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy', metrics=['accuracy'])

# Train the model with improved hyperparameters
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# Streamlit interface
st.title("Parkinson's Disease Prediction from Audio")

# File uploader
uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "flac"])

# Function to predict new audio samples
def predict_audio(file_path):
    
    feature = extract_features(file_path).reshape(1, -1)
    feature = scaler.transform(feature)  
    prediction = model.predict(feature)
    return "Parkinson Present" if prediction > 0.5 else "Healthy"


def record_and_predict():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    st.info("Please speak now...")

    with mic as source:
        recognizer.adjust_for_ambient_noise(source)  
        audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)  

    # Save as temp file
    with open("temp_audio.wav", "wb") as f:
        f.write(audio.get_wav_data())

    # Predict based on the recorded audio
    result = predict_audio("temp_audio.wav")
    return result


if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    
    # Save the uploaded file 
    temp_file_path = 'temp_uploaded_audio.wav'
    with open(temp_file_path, 'wb') as f:
        f.write(uploaded_file.read())
    
    # Prediction
    result = predict_audio(temp_file_path)
    if result == "Parkinson Present":
        st.markdown(f"<h3 style='color: red; text-align: center;'>🚨 {result} 🚨</h3>", unsafe_allow_html=True)
    else:
        st.markdown(f"<h3 style='color: green; text-align: center;'>✅ {result} ✅</h3>", unsafe_allow_html=True)

# Speech input button for user to record
if st.button("Click to Record Speech"):
    result = record_and_predict()
    if result == "Parkinson Present":
        st.markdown(f"<h3 style='color: red; text-align: center;'>🚨 {result} 🚨</h3>", unsafe_allow_html=True)
    else:
        st.markdown(f"<h3 style='color: green; text-align: center;'>✅ {result} ✅</h3>", unsafe_allow_html=True)
