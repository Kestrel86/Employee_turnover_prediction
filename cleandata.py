import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


# Load dataset
df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")


# Quick overview
print(df.info())      # General information
print(df.describe())   # Summary stats for numerical columns
print(df.head())       # First few rows


# Drop unnecessary/redundant columns
df.drop(columns=['EmployeeNumber', 'EmployeeCount', 'StandardHours', 'Over18'], inplace=True)


# Check for missing values
print(df.isnull().sum())

df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
df['OverTime'] = df['OverTime'].map({'Yes': 1, 'No': 0})


# Convert categorical variables to one-hot encoding
df = pd.get_dummies(df, columns=['BusinessTravel', 'Department', 'EducationField', 
                                 'Gender', 'JobRole', 'MaritalStatus'], drop_first=True)


# Standardize numerical columns in dataset
scaler = StandardScaler()
numerical_cols = ['Age', 'MonthlyIncome', 'DistanceFromHome', 'TotalWorkingYears', 'YearsAtCompany']
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])


# Scale columns that use integers as labels (like JobSatisfaction and PerformanceRating)
# List of ordinal columns
ordinal_cols = ['Education', 'EnvironmentSatisfaction', 'JobInvolvement', 
                'JobSatisfaction', 'PerformanceRating', 'RelationshipSatisfaction', 
                'WorkLifeBalance']
# Apply MinMax scaling (scales each feature between 0 and 1)
scaler = MinMaxScaler()
df[ordinal_cols] = scaler.fit_transform(df[ordinal_cols])


# Save cleaned dataset to new file
df.to_csv("cleaned_dataset.csv", index=False)