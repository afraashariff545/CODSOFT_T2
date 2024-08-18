# CODSOFT_T2
CREDIT CARD FRAUD  DETECTION

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

data_path = "C:/Users/545af/Downloads/WORK/TASK_2/ML_CodSoft_Task_2/data.csv"
data = pd.read_csv(data_path)
data.fillna(method='ffill', inplace=True)

data['year'] = pd.to_datetime(data['trans_date_trans_time']).dt.year
data['month'] = pd.to_datetime(data['trans_date_trans_time']).dt.month
data['day'] = pd.to_datetime(data['trans_date_trans_time']).dt.day
data['hour'] = pd.to_datetime(data['trans_date_trans_time']).dt.hour  

numerical_features = ['amt', 'lat', 'long', 'city_pop', 'year', 'month', 'day', 'hour']
categorical_features = data.columns.difference(numerical_features)

for col in categorical_features:
  encoder = LabelEncoder()
  data[col] = encoder.fit_transform(data[col])

scaler = StandardScaler()
data[numerical_features] = scaler.fit_transform(data[numerical_features])

target = 'is_fraud'
y = data[target].copy()
X = data.drop(target, axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("AUC-ROC:", auc)
