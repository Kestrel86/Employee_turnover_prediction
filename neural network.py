import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.dummy import DummyClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

# Load dataset and split data
data = pd.read_csv("employee_attrition_data.csv")
X = data.drop('Attrition', axis=1)  
y = data['Attrition']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Baseline model
baseline_model = DummyClassifier(strategy="most_frequent")
baseline_model.fit(X_train, y_train)
y_baseline_pred = baseline_model.predict(X_test)
baseline_auc = roc_auc_score(y_test, y_baseline_pred)

print("\nClassification Report for Baseline Model:")
print(classification_report(y_test, y_baseline_pred))
print("\nConfusion Matrix for Baseline Model:")
print(confusion_matrix(y_test, y_baseline_pred))
print(f"\nBaseline Model AUC: {baseline_auc}")

# Neural Network model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Predict on test data
y_nn_pred_prob = model.predict(X_test).flatten()
y_nn_pred = np.round(y_nn_pred_prob)

# Neural Network Model evaluation
nn_auc = roc_auc_score(y_test, y_nn_pred_prob)

print("\nClassification Report for Neural Network Model:")
print(classification_report(y_test, y_nn_pred))
print("\nConfusion Matrix for Neural Network Model:")
print(confusion_matrix(y_test, y_nn_pred))
print(f"\nNeural Network Model AUC: {nn_auc}")

# Calculate ROC curve for both models
fpr_baseline, tpr_baseline, _ = roc_curve(y_test, y_baseline_pred)  # Baseline binary predictions
fpr_nn, tpr_nn, _ = roc_curve(y_test, y_nn_pred_prob)               # Neural network probabilities

# Plot ROC curves
plt.figure(figsize=(8, 6))
plt.plot(fpr_baseline, tpr_baseline, linestyle='--', color='blue', label='Baseline Model (AUC = %0.2f)' % baseline_auc)
plt.plot(fpr_nn, tpr_nn, color='red', label='Neural Network Model (AUC = %0.2f)' % nn_auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison of Baseline vs Neural Network')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()
