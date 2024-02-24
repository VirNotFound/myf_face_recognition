import numpy as np
from keras_facenet import FaceNet
import cv2
import os
from mtcnn import MTCNN
from ultralytics import YOLO

# Initialize FaceNet model
print("Loading FaceNet Model...")
facenet_model = FaceNet()
print("FaceNet model loaded successfully.")

# Initialize MTCNN detector
detector = None

# Initialize YOLO model
yolo_model = None

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize MTCNN detector
def load_mtcnn_model():
    global detector
    if detector is None:
        print("Loading MTCNN model...")
        detector = MTCNN()
        print("MTCNN model loaded successfully.")

# Initialize YOLO detector
def load_yolo_model():
    global yolo_model
    if yolo_model is None:
        print("Loading YOLO model...")
        yolo_model = YOLO('myf_face_recognition\\myf_face_recognition\\yolov8n-face.pt')
        print("YOLO model loaded successfully.")

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

def generate_face_embeddings(name, folder_path, output_folder_path):
 
    # Function to extract face embeddings from an image
    def extract_face_embeddings(image_path, count):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect face in the image
        load_mtcnn_model()
        results = detector.detect_faces(image)

        if results:
            x, y, w, h = results[0]['box']
            x, y, w, h = int(x), int(y), int(w), int(h)
            cropped_image = image[y:y + h, x:x + w]

            # training_images = cv2.resize(cropped_image, (480,480))
            # cv2.imshow("Training", training_images)

            resized_image = cv2.resize(cropped_image, (160, 160))


            # Find embeddings of the face in the image
            face_embeddings = facenet_model.embeddings([np.array(resized_image)])
            print(str(count) + " Completed")

            return face_embeddings.flatten()  # Flatten to make it a 1D array
        else:
            return None

    # List to store face embeddings
    all_embeddings = []

    # Iterate through each image in the folder
    count = 0
    for image_file in os.listdir(folder_path):
        if image_file.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(folder_path, image_file)
            count += 1

            # Extract face embeddings from the image
            face_embeddings = extract_face_embeddings(image_path, count)

            if face_embeddings is not None:
                # Add face embeddings to the list
                all_embeddings.append(face_embeddings)

    # Combine the embeddings into a single embedding vector
    combined_embedding = np.mean(all_embeddings, axis=0)

    # Save the numpy file
    file_name = os.path.join(output_folder_path, f'{name}_embeddings.npy')
    np.save(file_name, combined_embedding)
    print(f"Embeddings generated and saved successfully as {file_name}")

def image_recognise_faces(test_image, embedding_files, threshold=0.65, show_frame=True):
    global facenet_model

    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
    image_np_original = np.array(test_image)

    load_mtcnn_model()
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

def video_recognise_faces(test_image, embedding_files, threshold=0.65, show_frame=True):
    global facenet_model

    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
    image_np_original = np.array(test_image)

    load_yolo_model()
    results = yolo_model.predict(test_image, show=False, stream=True)
    recognized_faces = {}  # Dictionary to store recognized faces and their similarity scores

    if results:
        for r in results:
            for box, score, class_id in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls):
                xmin, ymin, xmax, ymax = map(int, box.tolist())
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
            cv2.imshow("Frame", image_np_original)
    else:
        print("No faces found")

    return recognized_faces  # Return dictionary containing recognized faces and their similarity scores

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

