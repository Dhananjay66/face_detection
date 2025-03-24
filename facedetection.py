import cv2
import numpy as np
import speech_recognition as sr
import threading

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Load pre-trained deep learning models
dnn_model = r"D:\face detection\deploy.prototxt"
dnn_weights = r"D:\face detection\res10_300x300_ssd_iter_140000 (1).caffemodel"
age_model = r"D:\face detection\age_deploy.prototxt"
age_weights = r"D:\face detection\age_net.caffemodel"
gender_model = r"D:\face detection\gender_deploy.prototxt"
gender_weights = r"D:\face detection\gender_net.caffemodel"

# Load DNN models
face_net = cv2.dnn.readNetFromCaffe(dnn_model, dnn_weights)
age_net = cv2.dnn.readNetFromCaffe(age_model, age_weights)
gender_net = cv2.dnn.readNetFromCaffe(gender_model, gender_weights)

# Age and gender labels
AGE_LABELS = ["0-2", "4-6", "8-12", "15-20", "25-32", "38-43", "48-53", "60-100"]
GENDER_LABELS = ["Male", "Female"]

# Speech recognition setup
recognizer = sr.Recognizer()
speech_text = ""  # Store speech text


def detect_faces_dnn(image, net):
    """Detects faces in an image using OpenCV DNN."""
    height, width = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            (x, y, x_max, y_max) = box.astype("int")
            faces.append((x, y, x_max, y_max))

    return faces


def predict_age_gender(face_image):
    """Predicts age and gender for a given face image."""
    blob = cv2.dnn.blobFromImage(face_image, 1.0, (227, 227), (78.426337, 87.768914, 114.895847), swapRB=False)

    # Predict gender
    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    gender = GENDER_LABELS[gender_preds[0].argmax()]

    # Predict age
    age_net.setInput(blob)
    age_preds = age_net.forward()
    age = AGE_LABELS[age_preds[0].argmax()]

    return gender, age


def recognize_speech():
    """Continuously records speech and saves it to a text file."""
    global speech_text
    with sr.Microphone() as source:
        print("🎤 Listening... Speak now!")
        recognizer.adjust_for_ambient_noise(source)

        while True:
            try:
                audio = recognizer.listen(source, timeout=5)
                text = recognizer.recognize_google(audio)
                speech_text += text + " "

                # Save speech to a file
                with open("speech_output.txt", "w") as file:
                    file.write(speech_text)

                print(f"📝 Recognized: {text}")

            except sr.WaitTimeoutError:
                print("⏳ No speech detected... continuing.")
            except sr.UnknownValueError:
                print("❌ Could not understand audio.")
            except sr.RequestError:
                print("⚠ Speech recognition service error.")


def main():
    cap = cv2.VideoCapture(0)

    # Start speech recognition in a separate thread
    speech_thread = threading.Thread(target=recognize_speech, daemon=True)
    speech_thread.start()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = detect_faces_dnn(frame.copy(), face_net)

        for (x, y, x_max, y_max) in faces:
            face = frame[y:y_max, x:x_max]  # Extract face ROI
            if face.size == 0:
                continue

            # Predict age and gender
            gender, age = predict_age_gender(face)

            # Draw face rectangle
            cv2.rectangle(frame, (x, y), (x_max, y_max), (0, 255, 0), 2)

            # Display age and gender
            label = f"{gender}, {age}"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('Age & Gender Prediction', frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
