from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import csv
import time
from datetime import datetime
import os

# Load the background image
imgBackground = cv2.imread("background_image.jpeg")

COL_NAMES = ['NAME', 'TIME']

# Check if the image was loaded successfully
if imgBackground is None:
    print("Error: Unable to load the background image. Check the file path or file integrity.")
    exit()

# Print the shape of the background image for debugging
print("Background image shape:", imgBackground.shape)

# Initialize video capture and face detector
video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('data\\haarcascade_frontalface_default.xml')

# Load the saved face data and labels
with open('data/names.pkl', 'rb') as f:
    LABELS = pickle.load(f)
with open('data/faces_data.pkl', 'rb') as f:
    FACES = pickle.load(f)

# Initialize and train the KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

# Dimensions of the square in the background image where the frame will be placed
square_x, square_y = 50, 50  # Example top-left corner coordinates
square_w, square_h = 301, 301  # Adjust these values to match the actual square dimensions

# Print the intended section for debugging
print(f"Placing resized frame at [{square_y}:{square_y + square_h}, {square_x}:{square_x + square_w}]")

# Ensure the Attendance directory exists
if not os.path.exists("Attendance"):
    os.makedirs("Attendance")

# Start video capturing
while True:
    ret, frame = video.read()  # Read frame from the video
    if not ret:
        break  # Break the loop if the camera is not working

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:  # Draw rectangles around detected faces
        crop_img = frame[y:y+h, x:x+w]
        resized_image = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
        output = knn.predict(resized_image)
        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timeStamp = datetime.fromtimestamp(ts).strftime("%H-%M-%S")
        attendance_file = "Attendance/Attendance_" + date + ".csv"
        attendance = [str(output[0]), str(timeStamp)]

        cv2.putText(frame, str(output[0]), (x, y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)

    # Resize the frame to fit the square in the background
    resized_frame = cv2.resize(frame, (square_w, square_h))
    
    # Overlay the resized frame in the center of the square
    imgBackground[square_y:square_y + square_h, square_x:square_x + square_w] = resized_frame

    cv2.imshow("Frame", imgBackground)  # Display the frame with detected faces

    k = cv2.waitKey(1)
    if k == ord('o'):
        with open(attendance_file, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if not os.path.isfile(attendance_file) or os.stat(attendance_file).st_size == 0:
                writer.writerow(COL_NAMES)  # Write header if file doesn't exist or is empty
            writer.writerow(attendance)
    if k == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
