'''
4210 Group Project
Medium Complexity Code - Random Forest
Initial Code - Michael Chon
'''

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

# Load dataset
def load_data():
    data = pd.read_csv("employee_attrition_data.csv")

    if 'EmployeeNumber' not in data.columns:
        raise ValueError("The dataset must include an 'EmployeeID' column.")
    
    return data

# Preprocessing function (modify as needed based on dataset)
def preprocess_data(data):
    # Assuming the last column is the target and the rest are features
    #X = data.iloc[:, :-1]  # Features
    #y = data.iloc[:, -1]   # Target variable
    X = data.drop(['Attrition', 'EmployeeNumber'], axis=1)  # Remove Attrition and EmployeeID from features
    y = data['Attrition']

    # Store Employee IDs for final output
    employee_ids = data['EmployeeNumber']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test, employee_ids_train, employee_ids_test = train_test_split(
        X, y, employee_ids, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test, employee_ids_test

# Function to train and evaluate the Random Forest model
def random_forest_model(X_train, X_test, y_train, y_test):
    # Initialize the Random Forest Classifier
    model = RandomForestClassifier(random_state=42)
    
    # Fit the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
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

    top_features = feature_imp_df.head(5)

    print("\nTop Features by Importance:")
    print(top_features)  # Display top 5 features in the console

    # Plot the top five features
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=top_features, palette="viridis")
    plt.title("Top 5 Feature Importances")
    plt.show()
    
    # plt.figure(figsize=(10, 6))
    # sns.barplot(x="Importance", y="Feature", data=feature_imp_df)
    # plt.title("Feature Importances")
    # plt.show()
        # ROC curve
    if len(set(y_test)) == 2:  # Ensure binary classification
        y_prob = model.predict_proba(X_test)[:, 1]  # Probability for the positive class
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}", color="blue")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")  # Diagonal line
        plt.title("ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.show()
    else:
        print("ROC Curve cannot be computed for non-binary classification.")

    return y_pred, y_prob

def save_predictions(employee_ids_test, y_pred, y_prob):
    y_rf_pred = np.where(y_pred, 1, 0)
    # Create a DataFrame with Employee IDs, predictions, and probabilities
    output_df = pd.DataFrame({
        'EmployeeNumber': employee_ids_test,        # Employee IDs
        'PredictedAttrition': y_rf_pred,            # Predictions (1 = Leave, 0 = Stay)
        'AttritionProbability': y_prob              # Probability of Attrition
    })

    # Add a human-readable label for the prediction
    output_df['PredictionLabel'] = output_df['PredictedAttrition'].map({1: 'Leave', 0: 'Stay'})

    # Save the predictions to a CSV file for later review
    output_filename = 'employee_attrition_predictions.csv'
    output_df.to_csv(output_filename, index=False)

    print(f"Predictions saved to {output_filename}.")

# Calculate Department Morale
def calculate_department_morale(data):
    # Extract department columns
    if 'Department' not in data.columns:  
        data['Department'] = data.filter(like='Department_').idxmax(axis=1)

    # Calculate morale as 1 - attrition rate
    department_morale = data.groupby('Department')['Attrition'].mean().apply(lambda x: 1 - x)
    
    print("Department Morale (Higher is Better):")
    print(department_morale)
    
    # Plot morale
    plt.figure(figsize=(10, 6))
    department_morale.sort_values().plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title('Department Morale (Higher is Better)')
    plt.xlabel('Department')
    plt.ylabel('Morale')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def main():
    data = load_data()
    if data.empty:
        print("Please replace the load_data() function to load your actual dataset.")
        return
    
    X_train, X_test, y_train, y_test, employee_ids_test = preprocess_data(data)
    y_pred, y_prob = random_forest_model(X_train, X_test, y_train, y_test)

    save_predictions(employee_ids_test, y_pred, y_prob)

    print("\nCalculating Department Morale:")
    calculate_department_morale(data)

if __name__ == "__main__":
    main()
