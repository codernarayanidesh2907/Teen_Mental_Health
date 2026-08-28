# Teen Mental Health analysis using Libraries
#---------------- IMPORTING LIBRARIES ------------#
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import csv
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

#----------- Load the dataset -----------#
df = pd.read_csv('Teen_Mental_Health_Dataset.csv')
print(df.head())
print(df.info())
print(df.describe())

# Missing values
print(df.isnull().sum())
# currently, there are no missing values in the dataset.

# Search for duplicates 
print(df.duplicated().sum())
# Check for Unique values
print(df['gender'].nunique())
print(df['age'].nunique())
print(df['social_interaction_level'].unique())

# Count of each category
print(df['gender'].value_counts())
print(df['platform_usage'].value_counts())
print(df['depression_label'].value_counts())

# Percentage of each category
print(df['gender'].value_counts(normalize=True) * 100)
print(df['social_interaction_level'].value_counts(normalize=True) * 100)
print(df['depression_label'].value_counts(normalize=True) * 100)

# Grouping and Aggregation
print(df.groupby("depression_label")["sleep_hours"].mean())

print(df.groupby("gender")["stress_level"].mean())

print(df.groupby("platform_usage")["addiction_level"].mean())


#--------------- VISUALIZATION ----------------#
# Gender Distribution
plt.figure(figsize = (8,6))
sns.countplot(x = 'gender', data = df)
plt.title('Gender Distribution of Teenagers', fontsize = 16)
plt.xlabel('Gender', fontsize = 12)
plt.ylabel('Count', fontsize = 12)
plt.show()

# Social Interaction Level Distribution
plt.figure(figsize = (8,6))
sns.countplot(x='social_interaction_level',
              data=df,
              palette='Blues')
plt.title('Social Interaction Level Distribution', fontsize = 16)
plt.xlabel('Social Interaction Level', fontsize = 12)
plt.ylabel('Count', fontsize = 12)
plt.show()

# age Distribution
plt.figure(figsize=(8,5))
sns.histplot(df['age'], bins=10, kde=True)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

# Depression Label Distribution
plt.figure(figsize = (8,6))
sns.countplot(x = 'depression_label', data = df)
plt.title('Depression Label Distribution', fontsize = 16)
plt.xlabel('Depression Label', fontsize = 12)
plt.ylabel('Count', fontsize = 12)
plt.show()

# Sleep Hours Distribution
plt.figure(figsize = (8,6))
sns.histplot(x = 'sleep_hours', data = df, bins = 10, kde = True, color = 'lightgreen')
plt.title('Sleep Hours Distribution', fontsize = 16)
plt.xlabel('Sleep Hours', fontsize = 12)
plt.ylabel('Count', fontsize = 12)
plt.show()

# Academic Performace Distribution
plt.figure(figsize = (8,6))
sns.scatterplot(x = 'academic_performance', y = 'stress_level', data = df, hue = 'depression_label')
plt.title('Academic Performance vs Stress Level', fontsize = 16)
plt.xlabel('Academic Performance', fontsize = 12)
plt.ylabel('Stress Level', fontsize = 12)
plt.show()

# Addiction Level Distribution
plt.figure(figsize = (8,6))
sns.boxplot(x = 'addiction_level', data = df, color = 'salmon')
plt.title('Addiction Level Distribution', fontsize = 16)
plt.xlabel('Addiction Level', fontsize = 12)
plt.ylabel('Count', fontsize = 12)
plt.show()

# Platform Usage Distribution
plt.figure(figsize=(8,6))
sns.countplot(x='platform_usage', data=df)
plt.title('Platform Usage Distribution', fontsize=16)
plt.xlabel('Platform')
plt.ylabel('Count')
plt.xticks(rotation=30)
plt.show()

#  Daily Social Media Hours Distribution
plt.figure(figsize=(8,6))
sns.histplot(df['daily_social_media_hours'],
             bins=10,
             kde=True,
             color='orange')
plt.title('Daily Social Media Hours Distribution', fontsize=16)
plt.xlabel('Hours')
plt.ylabel('Count')
plt.show()

# Anxiety Level Distribution
plt.figure(figsize=(8,6))
sns.histplot(df['anxiety_level'],
             bins=10,
             kde=True,
             color='violet')
plt.title('Anxiety Level Distribution', fontsize=16)
plt.xlabel('Anxiety Level')
plt.ylabel('Count')
plt.show()


# Physical Activity Distribution
plt.figure(figsize=(8,6))
sns.histplot(df['physical_activity'],
             bins=10,
             kde=True,
             color='green')
plt.title('Physical Activity Distribution', fontsize=16)
plt.xlabel('Physical Activity')
plt.ylabel('Count')
plt.show()

# Sleep Hours vs Depression
plt.figure(figsize=(8,6))
sns.boxplot(x='depression_label',
            y='sleep_hours',
            data=df)
plt.title('Sleep Hours vs Depression')
plt.xlabel('Depression Label')
plt.ylabel('Sleep Hours')
plt.show()


# Stress Level vs Depression
plt.figure(figsize=(8,6))
sns.boxplot(x='depression_label',
            y='stress_level',
            data=df)
plt.title('Stress Level vs Depression')
plt.xlabel('Depression Label')
plt.ylabel('Stress Level')
plt.show()


# Anxiety Level vs Depression
plt.figure(figsize=(8,6))
sns.boxplot(x='depression_label',
            y='anxiety_level',
            data=df)
plt.title('Anxiety Level vs Depression')
plt.xlabel('Depression Label')
plt.ylabel('Anxiety Level')
plt.show()


# Social Media Hours vs Depression
plt.figure(figsize=(8,6))
sns.boxplot(x='depression_label',
            y='daily_social_media_hours',
            data=df)
plt.title('Social Media Usage vs Depression')
plt.xlabel('Depression Label')
plt.ylabel('Daily Social Media Hours')
plt.show()


# Crosstab Analysis
print("\nGender vs Depression")
print(pd.crosstab(df['gender'],
                  df['depression_label']))

print("\nPlatform vs Depression")
print(pd.crosstab(df['platform_usage'],
                  df['depression_label']))


# Pair Plot
sns.pairplot(
    df[
        [
            'daily_social_media_hours',
            'sleep_hours',
            'stress_level',
            'anxiety_level',
            'addiction_level',
            'depression_label'
        ]
    ],
    hue='depression_label'
)

plt.show()

# Correlation Heatmap 
corr = df.select_dtypes(include='number').corr()

plt.figure(figsize=(10,8))
sns.heatmap(corr,
            annot=True,
            cmap='coolwarm',
            fmt='.2f')

plt.title("Correlation Heatmap")
plt.show()

#------------------ Data Preprocessing ----------------#
print("\nData Preprocessing...")
print(df.head())

print("\nEncoding categorical variables...")
# Encoding categorical variables

encoder = LabelEncoder()
# Gender Encoder
gender_encoder = LabelEncoder()
df['gender'] = gender_encoder.fit_transform(df['gender'])

# Platform Encoder
platform_encoder = LabelEncoder()
df['platform_usage'] = platform_encoder.fit_transform(df['platform_usage'])

# Social Interaction Encoder
social_encoder = LabelEncoder()
df['social_interaction_level'] = social_encoder.fit_transform(df['social_interaction_level'])

# print the Mappings 
print("Gender:")
for i, value in enumerate(gender_encoder.classes_):
    print(i, "->", value)

print("\nPlatform:")
for i, value in enumerate(platform_encoder.classes_):
    print(i, "->", value)

print("\nSocial Interaction:")
for i, value in enumerate(social_encoder.classes_):
    print(i, "->", value)

# check again
print(df.head())

# ------------------ Splitting the dataset ----------------#
X = df.drop('depression_label', axis=1)
Y = df['depression_label']

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
print("\nTraining set size:", X_train.shape)
print("\nTesting set size:", X_test.shape)
print("\nTraining labels size:", Y_train.shape)
print("\nTesting labels size:", Y_test.shape)

print("\nFeatures used for prediction:")
print(X.columns)

# Check for class balance
print(Y.value_counts())

# ---------------- Random Forest Classifier ----------------#
print("\nTraining Random Forest Classifier...")
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42,  class_weight="balanced")
rf_classifier.fit(X_train, Y_train)

y_pred = rf_classifier.predict(X_test)
print("\nPredictions on test set:")
print(y_pred)

# Accuracy
accuracy = accuracy_score(Y_test, y_pred)
print("\nAccuracy of the model: {:.2f}%".format(accuracy * 100))

# Confusion Matrix
conf_matrix = confusion_matrix(Y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

# Classification Report
class_report = classification_report(Y_test, y_pred)
print("\nClassification Report:")
print(class_report)

# Confusion matrix Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
             xticklabels=['No Depression', 'Depression'],
             yticklabels=['No Depression', 'Depression'])
plt.title('Confusion Matrix Heatmap')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ----------------- Feature Importance ----------------#
feature_importances = pd.DataFrame(rf_classifier.feature_importances_, columns=['Importance'], index=X.columns)
feature_importances = feature_importances.sort_values('Importance', ascending=False)

print("\nFeature Importances:")
print(feature_importances)
# Visualizing Feature Importances
plt.figure(figsize=(10,6))
sns.barplot(x=feature_importances['Importance'], y=feature_importances.index, color='lightblue')
plt.title('Feature Importances from Random Forest Classifier')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.show()

#--------------- SAVE THE MODEL ----------------#
joblib.dump(rf_classifier, 'teen_mental_health_model.pkl')
print("Model saved as 'teen_mental_health_model.pkl'")
print("Saved Successfully!!!!")

# ------- ----------------SAVE THE ENCODERS ----------------#
joblib.dump(gender_encoder, "gender_encoder.pkl")
joblib.dump(platform_encoder, "platform_encoder.pkl")
joblib.dump(social_encoder, "social_encoder.pkl")
print("Encoders saved successfully!")