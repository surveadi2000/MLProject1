# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix

# Load the dataset
dataset_path = "C:/Users/shrey/Desktop/NITT tiruchipali Project/ML Proejct 2/Financials.csv"
df = pd.read_csv(dataset_path)

# Display the first few rows of the dataset to understand its structure
print(df.head())

# Data Preprocessing
# (Note: Preprocessing steps may vary based on the specific characteristics of your dataset)

# Handling missing values (if any)
df.dropna(inplace=True)

# Feature scaling (if necessary)
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df.drop('Fraudulent', axis=1))

# Splitting the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(scaled_features, df['Fraudulent'], test_size=0.2, random_state=42)

# Model Development
# Example: Using Logistic Regression

model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC AUC Score:", roc_auc)
print("Confusion Matrix:\n", conf_matrix)
