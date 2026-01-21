# Network Traffic Classification Model – Assignment 2

## 📊 Project Overview

This project implements an improved machine learning model for classifying network traffic types using the Darknet dataset. The model accurately distinguishes between four traffic categories:

- Non-Tor  
- NonVPN  
- Tor  
- VPN  

The final model achieves **99.89% accuracy**, representing a major improvement over the baseline model.

---

## 🎯 Performance Results

### Model Performance

| Metric | Value |
|------|------|
| Test Accuracy | **99.89%** |
| Baseline Accuracy | ~70% |
| Improvement | **+29.89%** |

### Model Comparison

| Model | Accuracy | Precision | Recall | F1-Score |
|------|---------|-----------|--------|---------|
| Random Forest | 99.89% | 1.00 | 1.00 | 1.00 |
| Neural Network | 92.95% | - | - | - |
| Baseline | ~70% | - | - | - |

---

## 📈 Classification Performance by Class

| Class | Precision | Recall | F1-Score | Support |
|------|-----------|--------|---------|---------|
| Non-Tor | 1.00 | 1.00 | 1.00 | 22,079 |
| NonVPN | 0.99 | 1.00 | 1.00 | 4,772 |
| Tor | 1.00 | 0.90 | 0.95 | 279 |
| VPN | 1.00 | 1.00 | 1.00 | 4,584 |

---

## 📁 Dataset Information

| Property | Value |
|--------|------|
| Source | Darknet Dataset |
| Original Samples | 158,616 |
| Clean Samples | 158,566 |
| Features | 82 (81 + label) |
| Missing Values | 50 rows removed |

Missing rows were removed due to NaN values in:

- `Flow Bytes/s`
- `Flow Packets/s`

---

## 🏗️ Model Architecture

**Best Model:** Random Forest Classifier

| Parameter | Value |
|---------|------|
| Trees | 100 |
| Max Depth | 15 |
| Min Samples Split | 10 |
| Min Samples Leaf | 4 |
| Random State | 42 |

---

## 📈 Key Feature Importance (Top 10)

| Feature | Importance |
|-------|-----------|
| Src IP | 26.06% |
| Dst IP | 15.55% |
| Idle Max | 4.51% |
| Fwd Seg Size Min | 3.69% |
| Bwd Packet Length Min | 3.68% |
| FWD Init Win Bytes | 2.70% |
| Idle Mean | 2.32% |
| Flow IAT Min | 2.19% |
| Bwd Packets/s | 2.11% |
| Packet Length Min | 1.91% |

**Insight:** Src/Dst IP addresses contribute over **41%** of total predictive power.

---

## 🔧 Implementation Details

### Data Preprocessing

- Removed 50 NaN rows  
- Factorized IP addresses  
- RobustScaler for outlier handling  
- Infinite/NaN validation checks  

### Key Improvements

- Feature scaling with RobustScaler  
- 3-fold cross-validation  
- Hyperparameter tuning  
- Data validation & error handling  
- Model persistence  
- Feature importance analysis  
- Multi-model comparison  

---

## 📈 Model Evaluation

### Confusion Matrix

| True\Predicted | Non-Tor | NonVPN | Tor | VPN |
|-----------|--------|-------|-------|--------|
| Non-Tor |22075 | 4 | 0 | 0 |
| NonVPN | 0 | 4772 | 0 | 0 |
| Tor | 0 | 28 | 251 | 0 |
| VPN | 0 | 2 | 0 | 4582 |


### Key Insights

- Non-Tor: 22075 / 22079 correct  
- NonVPN: Perfect classification  
- Tor: Hardest class (90% recall)  
- VPN: Near perfect classification  

---

## 🚀 How to Use

### Requirements

bash
pip install pandas numpy scikit-learn matplotlib joblib

Run
improved_model.py

--------------

Generated Files: 

- best_model.pkl
-scaler.pkl
-label_encoder.pkl
-Diagnostic console outputs


---

Load and Use the Model
import joblib

model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')
le = joblib.load('label_encoder.pkl')

new_data_scaled = scaler.transform(new_data)
predictions = model.predict(new_data_scaled)
original_labels = le.inverse_transform(predictions)


---

## Data Characteristics

### Dataset Statistics

| Metric | Range |
|------|------|
| Flow Duration | 1ms – 119,999,999ms |
| Packet Size | 0 – 66,247 bytes |
| Flow Rate | Up to 346M bytes/s |
| Idle Time | Up to 1.45 quadrillion μs |

### Class Distribution

| Class | Percentage |
|------|------------|
| Non-Tor | 69.6% |
| NonVPN | 15.0% |
| Tor | 0.9% |
| VPN | 14.5% |

---

## Technical Details

### Cross-Validation Results

| Fold | Accuracy |
|-----|---------|
| Fold 1 | 99.90% |
| Fold 2 | 99.87% |
| Fold 3 | 99.92% |

**Mean Accuracy:** **99.90% ± 0.02%**

### Training Information

| Metric | Value |
|------|------|
| Training Time | 2–3 minutes |
| Memory Usage | 1–2GB RAM |
| CPU Utilization | Multi-threaded |

---

## Assignment 2 Requirements Met

### Required

- Increased accuracy  
- Improved preprocessing  
- Optimized Random Forest  
- Cross-validation  
- Confusion matrix  
- Error analysis  
- Feature analysis  

### Additional

- Model persistence  
- Multi-model comparison  
- Data diagnostics  
- Production readiness  

---

## Business / Research Implications

### Key Insights

- IP features dominate prediction  
- Timing features are critical  
- Tor remains hardest to classify  
- Overall production-ready accuracy  

### Potential Applications

- Network security monitoring  
- QoS optimization  
- VPN/Tor detection  
- Anomaly detection  

---

## Conclusion

The improved Random Forest model successfully meets all Assignment 2 objectives with **99.89% accuracy**. It significantly outperforms both the baseline and Neural Network models, proving the importance of robust preprocessing and feature engineering.

### Key Success Factors

- RobustScaler usage  
- Clean data handling  
- Optimized hyperparameters  
- Comprehensive evaluation  
- Production-ready implementation  


