# Assignment-2


Network Traffic Classification Model - Assignment 2
Project Overview
This project implements an improved machine learning model for classifying network traffic types from the Darknet dataset. The model successfully distinguishes between four traffic categories: Non-Tor, NonVPN, Tor, and VPN with exceptional accuracy.

Performance Results
Model Performance
Test Accuracy: 99.89% (0.9989)

Baseline Accuracy: ~70%

Improvement: +29.89% (29.9 percentage points)

Model Comparison

Model	           Accuracy	    Precision	   Recall  	F1-Score
Random Forest	    99.89%      	1.00	      1.00	   1.00
Neural Network	  92.95%	       -	         -	      -
Baseline	         ~70%	         -	         -	      -


Classification Performance by Class
Class	         Precision	      Recall	     F1-Score	    Support
Non-Tor	         1.00           	1.00	      1.00	      22,079
NonVPN	         0.99           	1.00      	1.00	      4,772
Tor	             1.00            	0.90	      0.95	       279
VPN	             1.00	            1.00	      1.00	      4,584

---------------------------------------------------------------------------------------------------------------

Dataset Information
Source: Darknet Dataset

Total Samples: 158,566 (after cleaning)

Original Samples: 158,616

Features: 82 columns (81 features + 1 label)

Missing Values: 50 rows removed due to NaN in 'Flow Bytes/s' and 'Flow Packets/s'

-------------------------------------------------------------------------------------------------------------------------------------------------

Model Architecture
Best Model: Random Forest Classifier

Number of Trees: 100

Max Depth: 15

Min Samples Split: 10

Min Samples Leaf: 4

Random State: 42

------------------------------------------------------------------------
Key Feature Importance
The top 10 most important features identified by the model:

Src IP (26.06%) - Source IP address

Dst IP (15.55%) - Destination IP address

Idle Max (4.51%) - Maximum idle time

Fwd Seg Size Min (3.69%) - Minimum forward segment size

Bwd Packet Length Min (3.68%) - Minimum backward packet length

FWD Init Win Bytes (2.70%) - Forward initial window bytes

Idle Mean (2.32%) - Mean idle time

Flow IAT Min (2.19%) - Minimum inter-arrival time

Bwd Packets/s (2.11%) - Backward packets per second

Packet Length Min (1.91%) - Minimum packet length

--------------------------------------------------------------------------------------------------

Implementation Details
Data Preprocessing
Handled Missing Values: Removed 50 rows with NaN values

IP Address Encoding: Factorized categorical IP addresses

Outlier Handling: Used RobustScaler for feature scaling

Data Validation: Multiple checks for infinite/NaN values


-------------------------------------------------------------------------------
Key Improvements Implemented

Feature Scaling: RobustScaler for better outlier handling

Cross-Validation: 3-fold CV with 99.9% mean accuracy

Hyperparameter Tuning: Optimized Random Forest parameters

Error Handling: Comprehensive data validation

Model Persistence: Save/load capability

Feature Analysis: Top feature importance identification

Multi-model Comparison: Random Forest vs Neural Network


-------------------------------------------------------------------------------------

Model Evaluation Metrics
Confusion Matrix
text
True\Predicted  Non-Tor  NonVPN  Tor   VPN
Non-Tor          22075     4      0     0
NonVPN              0    4772     0     0
Tor                 0     28    251     0
VPN                 0      2      0  4582


Key Insights

Non-Tor: Nearly perfect classification (22075/22079 correct)

NonVPN: Perfect classification (4772/4772 correct)

Tor: Good but has 28 misclassifications (251/279 correct = 90%)

VPN: Excellent classification (4582/4584 correct)


------------------------------------------------------------------------------

# How to Use
Requirements -> pip install pandas numpy scikit-learn matplotlib joblib

Run the Model
python improved_model.py
Generated Files:

1.best_model.pkl - Trained Random Forest model

2.scaler.pkl - Feature scaler

3.label_encoder.pkl - Label encoder

4.Various diagnostic outputs in console

------------------------------------------------------------

Load and Use the Model
python
import joblib
import numpy as np

Load saved components
model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')
le = joblib.load('label_encoder.pkl')

Prepare new data (must match training features)
new_data = ...  # Your data here

Scale and predict
new_data_scaled = scaler.transform(new_data)
predictions = model.predict(new_data_scaled)
original_labels = le.inverse_transform(predictions)


------------------------------------------------------------

Data Characteristics
Dataset Statistics

Flow Duration: 1ms to 119,999,999ms (2 minutes)

Packet Sizes: 0 to 66,247 bytes

Flow Rates: Up to 346,000,000 bytes/second

Idle Times: Up to 1.45 quadrillion microseconds

-------------------------------------------------------------

Class Distribution
Non-Tor: 69.6% (110,395 samples)

NonVPN: 15.0% (23,860 samples)

Tor: 0.9% (1,395 samples)

VPN: 14.5% (22,916 samples)


---------------------------------------------------------------

Technical Details
Cross-Validation Results

Fold 1: 99.90% accuracy

Fold 2: 99.87% accuracy

Fold 3: 99.92% accuracy

Mean: 99.90% ± 0.02%


---------------------------------------------------

Model Training
Training Time: ~2-3 minutes (depending on hardware)

Memory Usage: ~1-2GB RAM

CPU Utilization: Multi-threaded (n_jobs=-1)

------------------------------------------------

Assignment 2 Requirements Met
Required Improvements Implemented:
Increased Accuracy: From ~70% to 99.89%

Better Data Preprocessing: Robust scaling, outlier handling

Improved Model Architecture: Optimized Random Forest

Comprehensive Evaluation: CV, confusion matrix, classification report

Error Analysis: Identified Tor traffic as hardest to classify

Feature Analysis: Top 10 important features identified


# Additional Enhancements:
Model Persistence: Save/load capability

Multi-model Comparison: RF vs Neural Network

Data Diagnostics: Extensive data validation

Production Readiness: Error handling and logging

--------------------------------------------

Business/Research Implications
High-Value Insights

IP Addresses are Critical: Src/Dst IPs contribute ~41% of predictive power

Timing Features Matter: Idle times and IAT are significant predictors

Tor Detection Challenge: 90% recall for Tor (hardest class to detect)

Near-Perfect Classification: Overall 99.89% accuracy is production-ready

------------------------------------------------------

Potential Applications
Network security monitoring

Traffic shaping and QoS

Anomaly detection systems

VPN/Tor usage analysis


# Conclusion
The improved model successfully addresses all Assignment 2 requirements with exceptional performance (99.89% accuracy). The Random Forest classifier outperforms both the baseline (~70%) and alternative Neural Network (92.95%), demonstrating the effectiveness of proper data preprocessing and feature engineering.
---------------------------------------------------------------------------------
Key Success Factors:

Proper handling of outliers and missing data

Effective feature scaling with RobustScaler

Optimized Random Forest hyperparameters

Comprehensive model evaluation

Production-ready implementation
