import numpy as np
from keras_facenet import FaceNet
import cv2
import os

# Initialize FaceNet model
facenet_model = FaceNet()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def face_recognition(cropped_image, embedding_files, threshold=0.65):
    loaded_face_embeddings = [np.load(file) for file in embedding_files]
    # Find embeddings of the face in the test image
    test_face_embeddings = facenet_model.embeddings([np.array(cropped_image)])

    # Compare the test face embeddings with the loaded embeddings
    similarity_scores = [np.dot(loaded_face_embeddings[i], test_face_embeddings.T) for i in range(len(loaded_face_embeddings))]

    # Check if the test face belongs to any person in the database
    max_score = np.max(similarity_scores)
    if max_score > threshold:
        label = np.argmax(similarity_scores)
        base_name = os.path.basename(embedding_files[label])
        name_without_extension = os.path.splitext(base_name)[0].split('_')[0]
        recognized_person = name_without_extension.capitalize()  # Name of the recognized person
    else:
        recognized_person = "Unknown"

    return recognized_person, max_score

def simple_video_recognise_faces(test_image, embedding_files, threshold=0.65, show_frame=True):
    global facenet_model

    # Convert the image to grayscale
    gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

    # Perform face detection
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    recognized_faces = {}  # Dictionary to store recognized faces and their similarity scores
    image_np_original = np.array(test_image)

    for (x, y, w, h) in faces:
        xmin, ymin, xmax, ymax = int(x), int(y), int(x+w), int(y+h)
        if show_frame:
            cv2.rectangle(image_np_original, (xmin, ymin), (xmax, ymax), (0, 255, 0), 5)

        face_roi = cv2.resize(test_image[ymin:ymax, xmin:xmax], (160, 160))

        # Perform face recognition using the new function
        recognized_person, max_score = face_recognition(face_roi, embedding_files, threshold=threshold)

        if recognized_person == "Unknown":
            continue
        else:
            recognized_faces[recognized_person] = max_score  # Add to recognized_faces dictionary

        if show_frame:
            cv2.putText(image_np_original, f"{recognized_person} ({max_score:.2f})", (xmin, ymin - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    # Display the result
    if show_frame:
        cv2.imshow('Detected and Recognized Faces', image_np_original)

    return recognized_faces

