import cv2
import numpy as np
from PIL import Image
from keras.models import load_model
import tensorflow as tf

# Load the pre-trained Keras model for emotion detection
model = load_model('model.h5')

# Load the Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start video capture from the webcam
video = cv2.VideoCapture(0)

# Define a dictionary to map predicted labels to emotions
emotion_mapper = {
    0: 'anger',
    1: 'disgust',
    2: 'fear',
    3: 'happiness',
    4: 'sadness',
    5: 'surprise',
    6: 'neutral'
}

while True:
    # Capture a frame from the webcam
    ret, frame = video.read()
    if not ret:
        break

    # Convert the frame to grayscale (required for face detection)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Draw a bounding box around the detected face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Extract the region of interest (ROI) of the face
        roi_gray = gray_frame[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Convert the ROI to RGB
        rgb_roi = cv2.cvtColor(roi_color, cv2.COLOR_BGR2RGB)

        # Convert the RGB ROI to a PIL Image
        pil_image = Image.fromarray(rgb_roi)

        # Resize the image to the dimensions used in training (e.g., 48x48)
        resized_image = pil_image.resize((48, 48))

        # Convert the PIL Image to a NumPy array and normalize pixel values
        img_array = np.asarray(resized_image) / 255.0

        # Expand dimensions to match the 4D tensor shape (batch_size, height, width, channels)
        img_array = np.expand_dims(img_array, axis=0)

        # Predict the emotion using the model
        prediction = np.argmax(model.predict(img_array)[0])

        # Get the emotion label from the prediction
        emotion_label = emotion_mapper[prediction]

        # Display the emotion label above the bounding box
        cv2.putText(frame, emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2, cv2.LINE_AA)

    # Show the frame with the bounding box and prediction
    cv2.imshow("Prediction", frame)

    # Break the loop if the 'q' key is pressed
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# Release the video capture and close windows
video.release()
cv2.destroyAllWindows()