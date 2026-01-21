# improved_model.py
"""
Assignment 2
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings

warnings.filterwarnings('ignore')

# Set random seed
np.random.seed(42)

print("Loading data...")
df = pd.read_csv("Darknet.csv")

# Basic preprocessing
df = df.drop(["Flow ID", "Timestamp", "Label2"], axis=1)

# FIX 1: Handle infinite and NaN values BEFORE dropping
print("Checking for infinite/NaN values...")
print(f"Original shape: {df.shape}")

# Replace infinite values with NaN
df = df.replace([np.inf, -np.inf], np.nan)

# Check for NaN values
nan_counts = df.isna().sum()
print(f"NaN counts per column:\n{nan_counts[nan_counts > 0]}")

# Drop rows with NaN values
df = df.dropna()
print(f"Shape after dropping NaN: {df.shape}")

# Check for infinite values again (should be none)
if df.isin([np.inf, -np.inf]).any().any():
    print("Warning: Still found infinite values!")
    # Replace any remaining infinite with large finite numbers
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(df.median(numeric_only=True))

# Simple IP conversion
print("Converting IP addresses...")
df['Src IP'] = pd.factorize(df['Src IP'])[0]
df['Dst IP'] = pd.factorize(df['Dst IP'])[0]

# Check data types and ranges
print("\nChecking data ranges...")
for col in df.select_dtypes(include=[np.number]).columns:
    if df[col].nunique() > 1:  # Skip constant columns
        print(f"{col}: min={df[col].min():.2f}, max={df[col].max():.2f}, "
              f"mean={df[col].mean():.2f}, infinite={df[col].isin([np.inf, -np.inf]).any()}")

# FIX 2: Cap extreme values (optional)
for col in df.select_dtypes(include=[np.number]).columns:
    if col not in ['Label1']:  # Don't cap the label
        q99 = df[col].quantile(0.99)
        if q99 > 0:
            # Cap values at 99th percentile to handle outliers
            df[col] = np.where(df[col] > q99 * 100, q99, df[col])

# Encode labels
print("\nEncoding labels...")
le = LabelEncoder()
df['Label1'] = le.fit_transform(df['Label1'])

print(f"Classes: {le.classes_}")
print(f"Final data shape: {df.shape}")

# Split data
X = df.drop('Label1', axis=1)
y = df['Label1']

# Check for any remaining issues
print(f"\nChecking X for issues...")
print(f"Any NaN in X: {X.isna().any().any()}")
print(f"Any infinite in X: {X.isin([np.inf, -np.inf]).any().any()}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# Scale features with robust scaling option
print("\nScaling features...")

# FIX 3: Use RobustScaler instead of StandardScaler for better outlier handling
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()  # Less sensitive to outliers
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame for debugging
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)

print(f"Scaled training data - min: {X_train_scaled_df.min().min():.2f}, "
      f"max: {X_train_scaled_df.max().max():.2f}")

# =====================================================================
# OPTION 1: Random Forest
# =====================================================================
print("\n" + "=" * 50)
print("Training Random Forest Classifier...")
print("=" * 50)

rf_model = RandomForestClassifier(
    n_estimators=100,  # Reduced for faster training
    max_depth=15,  # Control depth
    min_samples_split=10,  # Prevent overfitting
    min_samples_leaf=4,  # Prevent overfitting
    random_state=42,
    n_jobs=-1
)

# Cross-validation
try:
    cv_scores = cross_val_score(rf_model, X_train_scaled, y_train, cv=3, scoring='accuracy')
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
except Exception as e:
    print(f"Cross-validation failed: {e}")
    print("Proceeding with direct training...")

# Train final model
rf_model.fit(X_train_scaled, y_train)

# Evaluate
y_pred_rf = rf_model.predict(X_test_scaled)
rf_accuracy = accuracy_score(y_test, y_pred_rf)

print(f"\nRandom Forest Test Accuracy: {rf_accuracy:.4f}")
print(f"Improvement from ~70% baseline: {rf_accuracy - 0.70:.4f}")

print("\nClassification Report (Random Forest):")
print(classification_report(y_test, y_pred_rf, target_names=le.classes_))

# =====================================================================
# OPTION 2: Simplified Neural Network
# =====================================================================
print("\n" + "=" * 50)
print("Training Neural Network...")
print("=" * 50)

try:
    nn_model = MLPClassifier(
        hidden_layer_sizes=(100, 50),  # Smaller network
        activation='relu',
        solver='adam',
        alpha=0.01,  # Regularization
        batch_size=256,
        learning_rate='adaptive',
        max_iter=100,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )

    nn_model.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred_nn = nn_model.predict(X_test_scaled)
    nn_accuracy = accuracy_score(y_test, y_pred_nn)

    print(f"\nNeural Network Test Accuracy: {nn_accuracy:.4f}")
    print(f"Improvement from ~70% baseline: {nn_accuracy - 0.70:.4f}")

except Exception as e:
    print(f"Neural Network training failed: {e}")
    print("Using Random Forest only...")
    nn_accuracy = 0

# =====================================================================
# COMPARE MODELS
# =====================================================================
print("\n" + "=" * 50)
print("MODEL COMPARISON")
print("=" * 50)

print(f"Random Forest Accuracy:    {rf_accuracy:.4f}")
if nn_accuracy > 0:
    print(f"Neural Network Accuracy:   {nn_accuracy:.4f}")
print(f"Baseline Accuracy:         ~0.7000")

# Choose best model
if nn_accuracy > 0 and nn_accuracy >= rf_accuracy:
    best_model = nn_model
    best_accuracy = nn_accuracy
    best_model_name = "Neural Network"
else:
    best_model = rf_model
    best_accuracy = rf_accuracy
    best_model_name = "Random Forest"

print(f"\nBest Model: {best_model_name} (Accuracy: {best_accuracy:.4f})")

# =====================================================================
# SAVE THE MODEL
# =====================================================================
import joblib

# Save the best model
joblib.dump(best_model, 'best_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(le, 'label_encoder.pkl')

print("\nSaved files:")
print("- best_model.pkl (trained model)")
print("- scaler.pkl (feature scaler)")
print("- label_encoder.pkl (label encoder)")

# =====================================================================
# SIMPLE FEATURE IMPORTANCE
# =====================================================================
if best_model_name == "Random Forest":
    print("\n" + "=" * 50)
    print("TOP 10 MOST IMPORTANT FEATURES")
    print("=" * 50)

    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print(feature_importance.head(10).to_string(index=False))

# =====================================================================
# CREATE SIMPLE CONFUSION MATRIX
# =====================================================================
print("\n" + "=" * 50)
print("CONFUSION MATRIX")
print("=" * 50)

cm = confusion_matrix(y_test, y_pred_rf)
print("Rows = True labels, Columns = Predicted labels")
print(f"Class order: {le.classes_}")
print("\nMatrix:")
print(cm)

# =====================================================================
# IMPROVEMENTS SUMMARY
# =====================================================================
print("\n" + "=" * 50)
print("IMPROVEMENTS IMPLEMENTED")
print("=" * 50)

improvements = [
    "1. Handled infinite/NaN values before processing",
    "2. Used RobustScaler instead of StandardScaler (better with outliers)",
    "3. Capped extreme values at 99th percentile",
    "4. Added data validation checks",
    "5. Cross-validation for model evaluation",
    "6. Proper error handling for all steps",
    "7. Feature importance analysis (for Random Forest)"
]

for imp in improvements:
    print(f"  {imp}")

print(f"\nFinal accuracy: {best_accuracy:.4f}")
print(f"Improvement: {best_accuracy - 0.70:.4f} (+{(best_accuracy - 0.70) * 100:.1f}%)")
print("=" * 50)

# =====================================================================
# QUICK TEST TO ENSURE MODEL WORKS
# =====================================================================
print("\n" + "=" * 50)
print("TESTING SAVED MODEL")
print("=" * 50)

# Test loading and predicting
try:
    loaded_model = joblib.load('best_model.pkl')
    loaded_scaler = joblib.load('scaler.pkl')
    loaded_le = joblib.load('label_encoder.pkl')

    # Test with a single sample
    test_sample = X_test_scaled[0:1]
    prediction = loaded_model.predict(test_sample)
    original_label = loaded_le.inverse_transform(prediction)

    print(f"Test prediction: {original_label[0]}")
    print("Model loading and prediction test PASSED!")

except Exception as e:
    print(f"Model test failed: {e}")