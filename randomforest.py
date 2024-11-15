'''
4210 Group Project
Medium Complexity Code - Random Forest
Initial Code - Michael Chon
'''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
def load_data():
    data = pd.read_csv("employee_attrition_data.csv")
    return data

# Preprocessing function (modify as needed based on dataset)
def preprocess_data(data):
    # Assuming the last column is the target and the rest are features
    X = data.iloc[:, :-1]  # Features
    y = data.iloc[:, -1]   # Target variable
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

# Function to train and evaluate the Random Forest model
def random_forest_model(X_train, X_test, y_train, y_test):
    # Initialize the Random Forest Classifier
    model = RandomForestClassifier(random_state=42)
    
    # Fit the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate model performance
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
    
    # Feature importance plot
    feature_importances = model.feature_importances_
    feature_names = X_train.columns
    feature_imp_df = pd.DataFrame({"Feature": feature_names, "Importance": feature_importances})
    feature_imp_df = feature_imp_df.sort_values(by="Importance", ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=feature_imp_df)
    plt.title("Feature Importances")
    plt.show()

def main():
    data = load_data()
    if data.empty:
        print("Please replace the load_data() function to load your actual dataset.")
        return
    
    X_train, X_test, y_train, y_test = preprocess_data(data)
    random_forest_model(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()
