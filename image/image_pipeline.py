#!/usr/bin/env python
# coding: utf-8

# # Facial Recognition Model
# **Formative 2 -- Task 4: Model Creation (Facial Recognition piece)**
# 
# Loads `image_features.csv` (from Task 2), trains and compares several classifiers, evaluates the best one properly, and saves it as `face_model.pkl` for the rest of the team's pipeline (voice verification + product model) to build on.

# ## Imports

# In[1]:


import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             classification_report, log_loss)


# ## 1. Load the CSV

# In[2]:


import os
df = pd.read_csv(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "image_features.csv")
)
feature_cols = [c for c in df.columns if c.startswith(('color_hist_', 'gray_hist_', 'hog_'))]

print('Rows:', len(df), '| Feature columns:', len(feature_cols))
print('Members (classes):', sorted(df['member'].unique()))
df[['member', 'expression', 'augmentation', 'filename']].head(10)


# ## 2. Split into train/test sets -- done carefully to avoid leakage
# A plain random row split would leak: the augmented versions of a photo (rotated / flipped / grayscale) are near-duplicates of the original, so a random split can easily put an augmented copy of a "test" photo into training, making accuracy look far better than it really is.
# 
# Instead we hold out **one entire expression per person** (`surprised`, plus its augmentations) for testing, and train only on `neutral` and `smiling` (plus their augmentations). This tests the model on a genuinely unseen photo of each known person -- a fair test of whether it recognizes *the person*, not a near-identical feature vector it already saw.

# In[3]:


TEST_EXPRESSION = 'surprised'

train_df = df[df['expression'] != TEST_EXPRESSION].reset_index(drop=True)
test_df = df[df['expression'] == TEST_EXPRESSION].reset_index(drop=True)

print(f'Train rows: {len(train_df)} (neutral + smiling, + augmentations)')
print(f'Test rows:  {len(test_df)} (surprised only, + augmentations -- never seen in training)')

X_train, y_train = train_df[feature_cols].values, train_df['member'].values
X_test, y_test = test_df[feature_cols].values, test_df['member'].values

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# *(If your team would rather use a plain random `train_test_split` as the assignment literally describes, that version is below, commented out -- but be aware it will report inflated accuracy due to the leakage explained above.)*

# In[4]:


# Plain random split (kept for reference only -- NOT used below):
# X = df[feature_cols].values
# y = df['member'].values
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y
# )


# ## 3. Train candidate classifiers

# In[5]:


candidates = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'SVM (linear)': SVC(kernel='linear', probability=True, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42),
    'XGBoost': XGBClassifier(eval_metric='mlogloss', random_state=42),
}

# XGBoost needs integer-encoded labels, not strings
class_names = sorted(df['member'].unique())
label_to_idx = {name: i for i, name in enumerate(class_names)}
y_train_idx = np.array([label_to_idx[y] for y in y_train])
y_test_idx = np.array([label_to_idx[y] for y in y_test])

results = []
fitted_models = {}
for name, model in candidates.items():
    if name == 'XGBoost':
        model.fit(X_train_scaled, y_train_idx)
        preds_idx = model.predict(X_test_scaled)
        preds = np.array([class_names[i] for i in preds_idx])
    else:
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='macro')
    probs = model.predict_proba(X_test_scaled)
    if name == 'XGBoost':
        loss = log_loss(y_test_idx, probs, labels=list(range(len(class_names))))
    else:
        loss = log_loss(y_test, probs, labels=class_names)
    results.append({'Model': name, 'Accuracy': acc, 'F1 (macro)': f1, 'Log Loss': loss})
    fitted_models[name] = model

results_df = pd.DataFrame(results).sort_values('F1 (macro)', ascending=False).reset_index(drop=True)
results_df


# ## 4. Evaluate the best model in detail
# Accuracy, F1-score, confusion matrix, and full classification report -- plus log-loss as a bonus metric where the model supports probability estimates (loss is optional per the assignment for classical ML, but informative where available).

# In[6]:


best_name = results_df.iloc[0]['Model']
best_model = fitted_models[best_name]
print(f'Best model: {best_name}')

if best_name == 'XGBoost':
    y_pred_idx = best_model.predict(X_test_scaled)
    y_pred = np.array([class_names[i] for i in y_pred_idx])
    y_proba = best_model.predict_proba(X_test_scaled)
    loss = log_loss(y_test_idx, y_proba, labels=list(range(len(class_names))))
else:
    y_pred = best_model.predict(X_test_scaled)
    y_proba = best_model.predict_proba(X_test_scaled)
    loss = log_loss(y_test, y_proba, labels=class_names)

print(f'\nAccuracy: {accuracy_score(y_test, y_pred):.3f}')
print(f'F1 (macro): {f1_score(y_test, y_pred, average="macro"):.3f}')
print(f'Log loss: {loss:.3f}')

print('\nClassification report:')
print(classification_report(y_test, y_pred))


# In[7]:


labels = sorted(df['member'].unique())
cm = confusion_matrix(y_test, y_pred, labels=labels)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'Confusion Matrix -- {best_name}')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()


# **Honest read on these numbers:** with only 12 real photos across 4 people and hand-crafted (non-deep-learning) features, don't expect deep-learning-grade accuracy -- a modest score here reflects the small dataset, not a broken pipeline. The confusion matrix above shows exactly which people the model mixes up, which is more actionable than the accuracy number alone: if it's confusing two specific people, that's worth digging into (e.g. similar background/lighting in their photos) before assuming the model itself needs to change.

# ## 5. Save the trained model

# In[8]:


joblib.dump(best_model, 'face_model.pkl')
print(f'Saved {best_name} to face_model.pkl')


# **Important for whoever integrates this into the CLI app:** `face_model.pkl` alone is not enough to classify a brand-new photo -- you also need the exact same `StandardScaler` and feature-column order used here, or predictions will be silently wrong. Saving those alongside it:

# In[9]:


joblib.dump(scaler, 'face_scaler.pkl')
joblib.dump(feature_cols, 'face_feature_columns.pkl')
joblib.dump(class_names, 'face_class_names.pkl')
print('Also saved face_scaler.pkl, face_feature_columns.pkl, face_class_names.pkl')

eval_json = {
    r['Model']: {'accuracy': r['Accuracy'], 'f1_score': r['F1 (macro)'], 'log_loss': r['Log Loss']}
    for r in results
}
with open('model_evaluation.json', 'w') as f:
    json.dump(eval_json, f, indent=2)
print('Saved evaluation metrics -> model_evaluation.json')


# ## Summary
# 
# - **Data:** `image_features.csv` (48 rows: 12 original + 36 augmented, 184 feature columns) from Task 2.
# - **Split:** leakage-safe -- entire `surprised` expression held out for testing, never seen during training (a plain random split is shown commented-out for reference, but not used, since it would leak near-duplicate augmentations into the test set).
# - **Models compared:** Logistic Regression, SVM (linear), Random Forest, XGBoost -- best selected by macro-F1 on the held-out set.
# - **Evaluation:** accuracy, F1-score, log loss, full classification report, and confusion matrix, all computed on the held-out split.
# - **Saved:** `face_model.pkl` (the classifier itself) plus `face_scaler.pkl`, `face_feature_columns.pkl`, and `face_class_names.pkl` -- all four are needed together to correctly classify a new photo later.
