from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from imblearn.over_sampling import SMOTE
import pandas as pd
import pickle
import kagglehub

# Load dataset
adityakadiwal_water_potability_path = kagglehub.dataset_download('adityakadiwal/water-potability')
df = pd.read_csv(adityakadiwal_water_potability_path + "/water_potability.csv")

# Fill missing values
df['ph'] = df['ph'].fillna(df['ph'].median())
df['Sulfate'] = df['Sulfate'].fillna(df['Sulfate'].median())
df['Trihalomethanes'] = df['Trihalomethanes'].fillna(df['Trihalomethanes'].median())

# Scale features
scaler = StandardScaler()
cols = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate',
        'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']
df[cols] = scaler.fit_transform(df[cols])

# Save the scaler
with open('ai/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Prepare train-test
X = df.drop('Potability', axis=1)
y = df['Potability']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Train model
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

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
model.fit(X_train_res, y_train_res)

# Metrics
y_pred = model.predict(X_test)
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("F1 Score: ", f1_score(y_test, y_pred))
print("Precision: ", precision_score(y_test, y_pred))
print("Recall: ", recall_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, model.predict_proba(X_test)[:,1]))
scores = cross_val_score(model, X_train_res, y_train_res, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), scoring='f1')
print("Mean F1:", scores.mean())

# Save the model
with open('ai/water-potability-detection.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model and scaler saved successfully.")
