# 🌿 Crop Disease Classification 🧪  
> Deep Learning Models Trained on PlantVillage for Classifying Crop Leaf Diseases  

This project applies a range of computer vision techniques to identify crop diseases from leaf images. Below is a breakdown of the key CV components:

📸 1. Image Acquisition & Dataset
The dataset used is the PlantVillage dataset, which consists of high-resolution leaf images from different crops (e.g., tomato, potato, pepper), labeled as either healthy or affected by specific diseases (e.g., bacterial spot, early blight).

Multiclass image classification task with 15 distinct classes

Images are captured in controlled conditions with clear background

🧼 2. Preprocessing
To make the dataset suitable for deep learning:

Resizing: All images are resized to 64x64x3 to standardize input dimensions for CNNs

Normalization: Pixel values are scaled to the range [0, 1] to accelerate training convergence

Data Augmentation:

Random rotation

Horizontal/vertical flipping

Zooming

These techniques help prevent overfitting and simulate real-world image variations.

🧠 3. Feature Extraction via CNNs
At the heart of the project is automatic feature extraction, powered by deep convolutional neural networks.

Key Models Used:
✅ Custom CNN: Learns spatial patterns like leaf texture, color spots, and disease boundaries from scratch

✅ VGG16: Pretrained on ImageNet, this model transfers learned filters for detecting textures, shapes, and edges

✅ EfficientNet: Scales up width/depth efficiently while preserving spatial hierarchies

✅ DenseNet121 (mentioned in your paper): Promotes feature reuse and learns dense, hierarchical visual patterns

These models are trained to detect fine-grained differences between diseases that often look very similar to the human eye.

🧪 4. Evaluation of Vision Models
Performance is assessed using:

Accuracy

Precision, Recall, F1-Score

Loss and accuracy plots across epochs

Confusion matrices to observe class-specific performance
---

🚀 Getting Started

🛠️ Installation

1.Clone this repository
```bash
git clone https://github.com/YOUR_USERNAME/crop-disease-classification.git
cd crop-disease-classification
```

2.Create and activate virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate
```
3.Install required dependencies
```bash
pip install -r requirements.txt
```
## 📚 Dataset
[PlantVillage Dataset on Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease)

📥 You don’t need to download it manually — the dataset is downloaded automatically in the notebook using the Kaggle API:
```python
!kaggle datasets download emmarex/plantdisease
```

🧠 Trained Models

| Filename                                        | Description                               |
| ----------------------------------------------- | ----------------------------------------- |
| `model_CNN_GPU_.keras`                          | Lightweight custom CNN                    |
| `model_DN.keras`                                | Deep custom CNN with multiple conv blocks |
| `plant_disease_model_EfficientNet1.keras`       | EfficientNet base model                   |
| `plant_disease_model_EfficientNet1_final.keras` | Fine-tuned EfficientNet                   |
| `vgg16_plantvillage_2.keras`                    | Transfer learning using VGG16             |


All models are located in the models/ directory. These models were trained on GPU using TensorFlow/Keras.

📓 Using the Notebook
To run and explore the model:
```bash
jupyter notebook notebooks/plantvillage_Final_deeplearning.ipynb
```
This notebook contains:
Dataset preprocessing
Model building (CNN, EfficientNet, VGG16)
Evaluation metrics
Accuracy and confusion matrix plots

