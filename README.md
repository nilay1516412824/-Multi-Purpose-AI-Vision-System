# -Multi-Purpose-AI-Vision-System
Build an intelligent image analysis platform that can do multiple vision tasks

Live-Demo Link: https://replit.com/@nilaygupta1602/-Multi-Purpose-AI-Vision-System

Demo Video: https://www.loom.com/share/b0a9ce4e0c5f4ea3b3edd9009c7d2a91?sid=34a7b56e-6135-4669-a573-58c43b8f36ad


Setup Guide

1.Install Deep Learning Libraries

• TensorFlow

• PyTorch

• OpenCV

2.Cloud Computing Setup

• Google Colab Pro(recommended for GPU access)

• Kaggle Kernels(free GPU time)

• Hugging Face Spaces(free AI app hosting)

Features

📸 Smart Photo Classifier (PyTorch ResNet-50)

Upload a single photo for classification into 1000 ImageNet categories.
Displays top-5 predictions with confidence scores.

📁 Batch Processing

Upload multiple images simultaneously for fast classification.
View aggregated results in a data table.

🩺 Specialized Analyzers (TensorFlow/Keras Placeholder)

Medical Image Analyzer (X-ray detection demo).
Plant Disease Detector (mock output with recommendations).
Note: These are placeholder outputs and can be extended with actual TensorFlow models.

👤 Face Analysis

Upload a photo with faces to analyze age, emotion, and gender.
Currently simulated with placeholder TensorFlow outputs.

🎥 Real-time Camera Processing

Live webcam classification using streamlit-webrtc + PyTorch + OpenCV.
Predicts the top object in each frame and overlays results on video.
