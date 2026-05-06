# Heart Disease Risk Predictor

ML-powered desktop app for heart disease classification using UCI Heart Disease Dataset.

## Models Used
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree

## Results
| Model | Accuracy | F1-Score |
|-------|----------|----------|
| Logistic Regression | 80.98% | 83.09% |
| KNN | 82.61% | 83.67% |
| Decision Tree | 78.80% | 80.00% |

**Best Model: KNN (F1: 83.67%)**

## Project Structure
- `eda.py` — Exploratory Data Analysis
- `preprocessing.py` — Data Cleaning & Scaling
- `models_lr_knn.py` — LR & KNN Training
- `evaluation.py` — Decision Tree + Model Comparison
- `gui.py` — Full Desktop Dashboard

## How to Run
```bash
pip install -r requirements.txt
python preprocessing.py
python models_lr_knn.py
python evaluation.py
python gui.py
```

## Dataset
UCI Heart Disease Dataset — [Kaggle](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data)
