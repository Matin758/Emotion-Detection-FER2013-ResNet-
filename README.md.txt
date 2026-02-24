# 😊 Facial Emotion Recognition using ResNet18 (FER2013)

A deep learning model for classifying human facial expressions into 7 emotion categories using the FER-2013 dataset and a fine-tuned ResNet18 architecture.

## 🎯 Emotions Detected
| Label | Emotion |
|-------|---------|
| 0 | Angry (خشم) |
| 1 | Disgust (تنفر) |
| 2 | Fear (ترس) |
| 3 | Happy (خوشحالی) |
| 4 | Sad (ناراحتی) |
| 5 | Surprise (تعجب) |
| 6 | Neutral (خنثی) |

## ✨ Features
- **Architecture:** Pre-trained `ResNet18` fine-tuned on FER-2013
- **Data Augmentation:** Random crop, horizontal flip, rotation
- **Optimizer:** Adam with layer-wise learning rates
- **Regularization:** Dropout, Label Smoothing, Weight Decay
- **Scheduler:** `ReduceLROnPlateau` for adaptive learning rate
- **Early Stopping:** Prevents overfitting automatically
- **Evaluation:** Confusion Matrix + Loss/Accuracy plots

## 📋 Requirements
```bash
pip install -r requirements.txt


Dataset
This project uses the FER-2013 dataset containing 35,887 grayscale face images (48×48 pixels).

👉 Download from Kaggle: FER-2013 Dataset

After downloading, organize the dataset in this structure:

text
dataset/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── sad/
│   ├── surprise/
│   └── neutral/
├── val/
│   └── ...
└── test/
    └── ...
🚀 How to Run
bash
python phase1.py --data_dir /path/to/your/dataset
Example:

bash
python phase1.py --data_dir ./dataset
📊 Outputs
After training completes, the following files will be generated:

File	Description
best_model.pth	Best model weights (saved automatically)
training_results.png	Train/Val Loss and Accuracy curves
confusion_matrix.png	Confusion Matrix on the test set
🛠️ Tech Stack
Python 3.8+

PyTorch & TorchVision

Scikit-learn

Matplotlib & Seaborn

📜 License
This project is licensed under the AUT License.