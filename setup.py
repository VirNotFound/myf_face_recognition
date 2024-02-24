from setuptools import setup, find_packages

setup(
    name='myf_face_recognition',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'opencv-python',
        'keras-facenet',
        'mtcnn',
        'ultralytics',
    ],
    entry_points={
    'console_scripts': [
        'myf_face_recognition_all=myf_face_recognition.all:main',
        'myf_face_recognition_embeddings=myf_face_recognition.embeddings:main',
        'myf_face_recognition_images=myf_face_recognition.images:main',
        'myf_face_recognition_simple_video=myf_face_recognition.simple_video:main',
        'myf_face_recognition_video=myf_face_recognition.video:main',
    ],
},

    package_data={'myf_face_recognition': ['yolov8n-face.pt']},
    include_package_data=True,
    zip_safe=False,
)
