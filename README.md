Face Detection, Age & Gender Prediction with Speech Recognition
This project implements real-time face detection, age & gender prediction, and speech recognition using OpenCV, deep learning models, and Google Speech Recognition.

Features
✅ Face Detection using OpenCV DNN and Haar Cascades

✅ Age & Gender Prediction using pre-trained models

✅ Real-time Speech Recognition (converts speech to text)

✅ Live Webcam Processing


Installation
1️⃣ Install Required Packages
Run the following command:

sh
Copy
Edit
pip install opencv-python numpy speechrecognition pyaudio
⚠ If pyaudio installation fails, install it manually:

sh
Copy
Edit
pip install pipwin
pipwin install pyaudio
2️⃣ Clone the Repository
sh
Copy
Edit
git clone https://github.com/Dhananjay66/face_detection.git
cd face_detection
3️⃣ Download Pre-trained Models
Place the following files in the project folder:

deploy.prototxt

res10_300x300_ssd_iter_140000.caffemodel

age_deploy.prototxt

age_net.caffemodel

gender_deploy.prototxt

gender_net.caffemodel

Usage
Run the script to start the application:

sh
Copy
Edit
python facedetection.py
Controls
🔹 Press "q" to exit the program

How It Works

1️⃣ Face Detection:

Uses OpenCV DNN model to detect faces in real-time from a webcam feed.

2️⃣ Age & Gender Prediction:

Extracts the detected face and predicts age & gender using deep learning models.

3️⃣ Speech Recognition:

Continuously listens for voice input and converts speech to text, saving it in speech_output.txt.

Demo

Troubleshooting
🚨 Error: No module named 'pyaudio'
👉 Run the following:

sh
Copy
Edit
pip install pipwin
pipwin install pyaudio
🚨 Camera Not Opening?
👉 Ensure your webcam is working and available.

Contributing
Feel free to fork this repository and submit a pull request with improvements!
