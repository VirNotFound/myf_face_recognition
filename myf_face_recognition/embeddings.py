import numpy as np
from keras_facenet import FaceNet
import cv2
import os
from mtcnn import MTCNN

# Initialize FaceNet model
facenet_model = FaceNet()

# Initialize MTCNN detector
detector = MTCNN()

def generate_face_embeddings(name, folder_path, output_folder_path):
 
    # Function to extract face embeddings from an image
    def extract_face_embeddings(image_path, count):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect face in the image
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
