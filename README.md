# 🌿 Crop Disease Classification 🧪  
> Deep Learning Models Trained on PlantVillage for Classifying Crop Leaf Diseases  

This repository contains well-trained deep learning models and tools to detect crop diseases from leaf images using the PlantVillage dataset.

---

#🚀 Getting Started
#🛠️ Installation
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

#📥 Download Dataset
You don’t need to download it manually — the dataset is downloaded automatically in the notebook using the Kaggle API:
```python
!kaggle datasets download emmarex/plantdisease```

#🧠 Trained Models
| Filename                                        | Description                               |
| ----------------------------------------------- | ----------------------------------------- |
| `model_CNN_GPU_.keras`                          | Lightweight custom CNN                    |
| `model_DN.keras`                                | Deep custom CNN with multiple conv blocks |
| `plant_disease_model_EfficientNet1.keras`       | EfficientNet base model                   |
| `plant_disease_model_EfficientNet1_final.keras` | Fine-tuned EfficientNet                   |
| `VGG16_model.keras`                             | Transfer learning using VGG16             |

All models are located in the models/ directory. These models were trained on GPU using TensorFlow/Keras.

#📓 Using the Notebook
To run and explore the model:
```bash
jupyter notebook notebooks/plantvillage_Final_deeplearning.ipynb
```
This notebook contains:
Dataset preprocessing
Model building (CNN, EfficientNet, VGG16)
Evaluation metrics
Accuracy and confusion matrix plots

