# MyF Face Recognition Library

MyF Face Recognition Library is a Python library for face recognition using a combination of MTCNN (Multi-task Cascaded Convolutional Networks), YOLO (You Only Look Once), and FaceNet models. It provides scripts for face embeddings generation, image-based face recognition, and video-based face recognition.

## Installation

```bash
pip install myf_face_recognition

Usage
Face Embeddings Generation
Generate face embeddings from a folder of images:
myf_face_recognition_embeddings --name <NAME> --folder <IMAGES_FOLDER> --output <OUTPUT_FOLDER>

Image-based Face Recognition
Perform face recognition on a single image:
myf_face_recognition_images --test_image <TEST_IMAGE_PATH> --embedding_files <EMBEDDING_FILES_FOLDER>

Simple Video-based Face Recognition
Perform face recognition on a video using a simple face detection approach:
myf_face_recognition_simple_video --test_video <TEST_VIDEO_PATH> --embedding_files <EMBEDDING_FILES_FOLDER>

Video-based Face Recognition with YOLO
Perform face recognition on a video using YOLO for object detection:
myf_face_recognition_video --test_video <TEST_VIDEO_PATH> --embedding_files <EMBEDDING_FILES_FOLDER>

Dependencies:
numpy
opencv-python
keras-facenet
mtcnn
ultralytics

License
This project is licensed under the MIT License - see the LICENSE file for details.


Make sure to replace placeholders like `<NAME>`, `<IMAGES_FOLDER>`, `<OUTPUT_FOLDER>`, `<TEST_IMAGE_PATH>`, `<TEST_VIDEO_PATH>`, and `<EMBEDDING_FILES_FOLDER>` with the actual values or paths.

Feel free to add more sections, details, or examples based on your library's features and usage.
