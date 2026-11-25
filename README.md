
# EcoSort AI: Computer Vision for Waste

EcoSort AI is a computer vision system that identifies waste items from a single photo and tells users whether they can be recycled.

## Overview
Poor waste sorting increases recycling costs and environmental pollution. EcoSort AI addresses this by classifying images into six categories (glass, paper, cardboard, plastic, metal, trash) and mapping each to a recyclable or non-recyclable decision.

## Features
- Six-class waste classification using transfer learning
- Comparison of two models: MobileNetV2 and ResNet50
- Recyclable / non-recyclable prediction
- Grad-CAM visual explanations of model decisions
- Class imbalance handling (class weights / weighted sampling)
- Evaluation on a locally collected image set
- Interactive Streamlit web app (image upload and camera)

## Tech Stack
Python, PyTorch, torchvision, scikit-learn, Streamlit, Google Colab

## Dataset
Trained on [TrashNet](https://github.com/garythung/trashnet) (Stanford), supplemented with locally collected photos for testing generalization. Please cite the original repository when using the dataset.

## Results
| Model | Accuracy | Macro-F1 |
|-------|----------|----------|
| MobileNetV2 | TBD | TBD |
| ResNet50 | TBD | TBD |

## Quick Start
    pip install -r requirements.txt
    streamlit run app/app.py

## Team
Built as a Capstone project for Samsung Innovation Campus (AI / ML / DI).
