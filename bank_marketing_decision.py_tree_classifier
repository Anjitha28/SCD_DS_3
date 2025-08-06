#Bank Marketing Decision Tree Classifier

#import required library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load Dataset
df = pd.read_csv("/content/bank.csv", sep=';')
print(df)

# Data Preprocessing
df = df.drop_duplicates()
print("First 5 Rows:\n", df.head())

# Encode categorical variables
df_encoded = pd.get_dummies(df, drop_first=True)
df_encoded.head()

#Feature and Target Split
X = df_encoded.drop('y_yes', axis=1)  # Features
y = df_encoded['y_yes']               # Target

#Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Train Decision Tree Model
dtc = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=42)
dtc.fit(X_train, y_train)

#Model Predictions & Evaluation
y_pred = dtc.predict(X_test)

#Accuracy
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Display Confusion Matrix using seaborn heatmap
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')

#Plot and save confusion matrix
plt.savefig("confusion_matrix_plot.png")      #Save

# Download confusion matrix image
from google.colab import files
files.download("confusion_matrix_plot.png")
