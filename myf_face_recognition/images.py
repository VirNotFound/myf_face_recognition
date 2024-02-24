import numpy as np
from keras_facenet import FaceNet
import cv2
import os
from mtcnn import MTCNN

# Initialize FaceNet model
facenet_model = FaceNet()

# Initialize MTCNN detector
detector = MTCNN()

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

def image_recognise_faces(test_image, embedding_files, threshold=0.65, show_frame=True):
    global facenet_model

    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
    image_np_original = np.array(test_image)

    results = detector.detect_faces(image_np_original)
    recognized_faces = {}  # Dictionary to store recognized faces and their similarity scores

    if results:
        # loaded_face_embeddings = [np.load(file) for file in embedding_files]

        for r in results:
            x, y, width, height = r['box']
            xmin, ymin, xmax, ymax = int(x), int(y), int(x + width), int(y + height)
            if show_frame:
                cv2.rectangle(image_np_original, (xmin, ymin), (xmax, ymax), (0, 255, 0), 5)

            # Crop the face region
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

        image_np_original = cv2.cvtColor(image_np_original, cv2.COLOR_BGR2RGB)
        if show_frame:
            cv2.imshow("Detected and Recognized Faces", image_np_original)
            print("Press any key to continue.")
            cv2.waitKey(0)
    else:
        print("No faces found")

    return recognized_faces
