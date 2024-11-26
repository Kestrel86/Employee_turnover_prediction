import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# -------------------- Data Loading and Preprocessing --------------------
# Load the dataset
data = pd.read_csv('employee_attrition_data.csv')

# Ensure the dataset includes an Employee ID column
if 'EmployeeNumber' not in data.columns:
    raise ValueError("The dataset must include an 'EmployeeID' column.")

# Encode categorical variables using one-hot encoding
data = pd.get_dummies(data, drop_first=True)

# Split the data into features (X) and target (y)
X = data.drop(['Attrition', 'EmployeeNumber'], axis=1)  # Remove Attrition and EmployeeID from features
y = data['Attrition']

# Store Employee IDs for final output
employee_ids = data['EmployeeNumber']

# Split data into training and test sets (70% train, 30% test)
X_train, X_test, y_train, y_test, employee_ids_train, employee_ids_test = train_test_split(
    X, y, employee_ids, test_size=0.3, random_state=42
)

# Ensure data is in numeric format for TensorFlow
X_train = np.array(X_train, dtype=np.float32)
X_test = np.array(X_test, dtype=np.float32)
y_train = np.array(y_train, dtype=np.float32)
y_test = np.array(y_test, dtype=np.float32)

# -------------------- Baseline Model: Random Forest --------------------
# Initialize and train the Random Forest model
baseline_model = RandomForestClassifier(random_state=42)
baseline_model.fit(X_train, y_train)

# Predict using the baseline model
y_baseline_pred = baseline_model.predict(X_test)
y_baseline_prob = baseline_model.predict_proba(X_test)[:, 1]

# -------------------- Neural Network Model --------------------
# Scale features for better model convergence (Normalization/Standardization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build the Neural Network model
model = Sequential([
    Dense(128, input_dim=X_train_scaled.shape[1], activation='relu'),
    Dropout(0.5),  # Dropout to prevent overfitting
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')  # Sigmoid for binary classification
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping to avoid overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the Neural Network
history = model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_data=(X_test_scaled, y_test), callbacks=[early_stopping])

# -------------------- Model Predictions --------------------
# Predict using the neural network
y_nn_prob = model.predict(X_test_scaled).flatten()
y_nn_pred = (y_nn_prob > 0.5).astype(int)

# -------------------- Model Evaluation --------------------
# Evaluate the neural network model
nn_test_loss, nn_test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Neural Network Test Accuracy: {nn_test_accuracy * 100:.2f}%")

# Classification report for the baseline model
print("Classification Report for Baseline Model:")
print(classification_report(y_test, y_baseline_pred))

# Classification report for the neural network model
print("Classification Report for Neural Network Model:")
print(classification_report(y_test, y_nn_pred))
# -------------------- Save Employee Predictions to CSV --------------------
# Create a DataFrame with Employee IDs, predictions, and probabilities
output_df = pd.DataFrame({
    'EmployeeNumber': employee_ids_test,      # Employee IDs
    'PredictedAttrition': y_nn_pred,          # Predictions (1 = Leave, 0 = Stay)
    'AttritionProbability': y_nn_prob        # Probability of Attrition
})

# Add a human-readable label for the prediction
output_df['PredictionLabel'] = output_df['PredictedAttrition'].map({1: 'Leave', 0: 'Stay'})

# Save the predictions to a CSV file for later review
output_filename = 'employee_attrition_predictions.csv'
output_df.to_csv(output_filename, index=False)

print(f"Predictions saved to {output_filename}.")
print("\nSample Output of Employee Predictions:")
print(output_df.head())


# -------------------- Department Morale Calculation --------------------
# Identify the department with the highest value for each employee
data['Department'] = data.filter(like='Department_').idxmax(axis=1)

# Calculate department morale (inverse of attrition rate)
department_morale = data.groupby('Department')['Attrition'].mean().apply(lambda x: 1 - x)

# -------------------- ROC Curve --------------------
# Compute ROC Curve and AUC for the Neural Network
fpr, tpr, thresholds = roc_curve(y_test, y_nn_prob)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
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

# -------------------- Department Morale Output --------------------
# Print the morale for each department
print("Department Morale (higher is better):")
print(department_morale)

# Visualize department morale
plt.figure(figsize=(10, 6))
department_morale.sort_values().plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Department Morale (Higher is Better)')
plt.xlabel('Department')
plt.ylabel('Morale')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -------------------- Model Evaluation Outputs --------------------

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_nn_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['Stay', 'Leave'])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix: Neural Network")
plt.show()

# Feature Importance Plot
importances = baseline_model.feature_importances_
feature_names = X.columns
sorted_indices = np.argsort(importances)[::-1]
sorted_features = feature_names[sorted_indices]
sorted_importances = importances[sorted_indices]

plt.figure(figsize=(10, 6))
plt.barh(sorted_features, sorted_importances, color='skyblue', edgecolor='black')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
