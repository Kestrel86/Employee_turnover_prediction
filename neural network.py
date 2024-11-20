import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('employee_attrition_data.csv')

# Encode categorical variables
data = pd.get_dummies(data, drop_first=True)

# Split data into features and target
X = data.drop('Attrition', axis=1)
y = data['Attrition']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Ensure data is numeric for TensorFlow
X_train = np.array(X_train, dtype=np.float32)
X_test = np.array(X_test, dtype=np.float32)
y_train = np.array(y_train, dtype=np.float32)
y_test = np.array(y_test, dtype=np.float32)

# ========== Baseline Model: Random Forest ==========
def baseline_model(X_train, y_train, X_test):
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    y_prob = rf_model.predict_proba(X_test)[:, 1]
    return rf_model, y_pred, y_prob

# Train baseline model
baseline_model, y_baseline_pred, y_baseline_prob = baseline_model(X_train, y_train, X_test)

# ========== Neural Network Model ==========
def build_nn_model(input_dim):
    model = Sequential([
        Dense(128, input_dim=input_dim, activation='relu'),
        Dropout(0.5),  # Dropout layer to prevent overfitting
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')  # Binary classification output
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Scale features (Normalization or Standardization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build and train neural network model
nn_model = build_nn_model(X_train_scaled.shape[1])

# Early stopping to avoid overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = nn_model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_data=(X_test_scaled, y_test), callbacks=[early_stopping])

# ========== Evaluate Models: Random Forest vs Neural Network ==========
# Predict with the neural network
y_nn_prob = nn_model.predict(X_test_scaled).flatten()
y_nn_pred = (y_nn_prob > 0.5).astype(int)

# ROC Curve and AUC score
fpr, tpr, thresholds = roc_curve(y_test, y_nn_prob)
roc_auc = auc(fpr, tpr)

# Evaluate Test Accuracy
nn_test_loss, nn_test_accuracy = nn_model.evaluate(X_test_scaled, y_test, verbose=0)

print(f"Neural Network Test Accuracy: {nn_test_accuracy * 100:.2f}%")

# Classification report for Baseline Model
print("Classification Report for Baseline Model:")
print(classification_report(y_test, y_baseline_pred))

# Classification report for Neural Network Model
print("Classification Report for Neural Network Model:")
print(classification_report(y_test, y_nn_pred))

# ========== Plotting and Evaluation ==========
# Plot ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# Plot training and validation accuracy
plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Neural Network Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

# Plot training and validation loss
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Neural Network Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()

# ========== Department Morale ==========
def calculate_department_morale(data):
    # Identify department column for each employee
    data['Department'] = data.filter(like='Department_').idxmax(axis=1)
    department_morale = data.groupby('Department')['Attrition'].mean().apply(lambda x: 1 - x)  # Inverse of attrition rate
    return department_morale

# Calculate and print department morale
department_morale = calculate_department_morale(data)
print("Department Morale (Higher is Better):")
print(department_morale)

# Add morale to the output data
data['Morale'] = data['Department'].map(department_morale)

# Plot department morale
plt.figure(figsize=(10, 6))
department_morale.sort_values().plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Department Morale (Higher is Better)')
plt.xlabel('Department')
plt.ylabel('Morale')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
