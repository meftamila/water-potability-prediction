from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from imblearn.over_sampling import SMOTE
import pandas as pd

import kagglehub
adityakadiwal_water_potability_path = kagglehub.dataset_download('adityakadiwal/water-potability')

print('Data source import complete.')

df = pd.read_csv(adityakadiwal_water_potability_path + "/water_potability.csv")

df.head(3)

df.info()

df['ph'] = df['ph'].fillna(df['ph'].median())
df['Sulfate'] = df['Sulfate'].fillna(df['Sulfate'].median())
df['Trihalomethanes'] = df['Trihalomethanes'].fillna(df['Trihalomethanes'].median())

df.info()

df.head()

scaler = StandardScaler()
cols = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']
for col in cols:
  df[col] = scaler.fit_transform(df[col].values.reshape(-1, 1))
df.head(2)

X = df.drop('Potability', axis=1)
y = df['Potability']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=(len(y_train[y_train==0])/len(y_train[y_train==1])),
    eval_metric='logloss',
    random_state=42
)

# model.fit(X_train, y_train)


smote = SMOTE(random_state=42)


X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

model.fit(X_train_res, y_train_res)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X_train_res, y_train_res, cv=cv, scoring='f1')
print("Accuracy: ", accuracy)
print("F1 Score: ", f1)
print("Precision: ", precision)
print("Recall: ", recall)
print("ROC-AUC:", roc_auc)
print("Mean F1:", scores.mean())

import pickle

pickle_filename = 'water-potability-detection.pkl'

# Save the trained model to a pickle file
with open(pickle_filename, 'wb') as file:
    pickle.dump(model, file)

print(f"Object successfully saved to {pickle_filename}")

# Open the file in binary read mode ('rb') and load the object
with open(pickle_filename, 'rb') as file:
    loaded_data = pickle.load(file)

print(f"Object successfully loaded from {pickle_filename}:")
print(loaded_data)

