import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
def load_data(filepath='employee_attrition_data.csv'):
    try:
        data = pd.read_csv(filepath)
        print("Data loaded successfully.")
    except FileNotFoundError:
        print("File not found. Please ensure the correct path is specified.")
        data = pd.DataFrame()  # Placeholder dataframe if file not found
    return data

# Preprocessing function
def preprocess_data(data):
    if data.empty:
        print("Data is empty. Please load a valid dataset.")
        return None, None, None, None, None
    
    # Separate features and target
    X = data.drop(columns='Attrition', errors='ignore')  # Drop target column
    y = data['Attrition']

    # Encoding for categorical features
    X = pd.get_dummies(X, drop_first=True)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)
    return X_train, X_test, y_train, y_test, data

# Evaluate the Decision Tree
def evaluate_decision_tree(X_train, X_test, y_train, y_test, max_depth=4):
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=7)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Accuracy and metrics
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm", cbar=False)
    plt.title("Decision Tree Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    # ROC Curve and AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve for Decision Tree Model")
    plt.legend(loc="lower right")
    plt.show()

# Feature Importance
    feature_importances = model.feature_importances_
    feature_names = X_train.columns
    feature_imp_df = pd.DataFrame({"Feature": feature_names, "Importance": feature_importances})
    feature_imp_df = feature_imp_df.sort_values(by="Importance", ascending=False)

    print("\nTop Features by Importance:")
    print(feature_imp_df.head(10))  # Display top 10 features in the console

    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=feature_imp_df.head(10), palette="viridis")
    plt.title("Top Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()

    return model

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

# Main Function
def main():
    data = load_data()
    if data.empty:
        print("Please load a valid dataset.")
        return

    X_train, X_test, y_train, y_test, data = preprocess_data(data)
    if X_train is None:
        return

    print("Evaluating Decision Tree:")
    model = evaluate_decision_tree(X_train, X_test, y_train, y_test)

    print("\nCalculating Department Morale:")
    calculate_department_morale(data)

if __name__ == "__main__":
    main()
