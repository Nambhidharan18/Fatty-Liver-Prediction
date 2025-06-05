import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score, roc_curve, auc ,cohen_kappa_score, recall_score, precision_score, log_loss
import lightgbm as lgb
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import label_binarize

#  Load Dataset
file_path = "synthetic_patient_data.csv"
df = pd.read_csv(file_path)


#  Define Features & Target
X = df.drop(columns=['Fatty Liver'])  # Features
y = df['Fatty Liver']  # Target: 0 (Healthy), 1 (NAFLD), 2 (AFLD), 3 (Non-Healthy)


#  Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#  Save the scaler for future predictions
joblib.dump(scaler, "scaler.pkl")

#  Define Base Models for Stacking
xgb_model = xgb.XGBClassifier(objective='multi:softmax', num_class=4, eval_metric='mlogloss', random_state=42)
svm_model = SVC(probability=True, kernel='rbf', random_state=42)
mlp_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)

base_models = [
    ('xgb', xgb_model),
    ('svm', svm_model),
    ('mlp', mlp_model)
]

#  Meta-Learner (Final Model in Stacking)
meta_learner = lgb.LGBMClassifier(objective='multiclass', num_class=4, random_state=42)

#  Stacking Classifier
stacked_model = StackingClassifier(estimators=base_models, final_estimator=meta_learner, passthrough=True)

#  Train Model
stacked_model.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)  # Train XGBoost separately

#  Save Models for Future Use
joblib.dump(stacked_model, "stacked_model.pkl")
joblib.dump(xgb_model, "xgboost_model.pkl")

#  Predict Probabilities
y_prob = stacked_model.predict_proba(X_test)
y_pred_proba = stacked_model.predict_proba(X_test)

# Convert Probabilities to Final Class Predictions
y_pred = np.argmax(y_pred_proba, axis=1)

# Multi-class metrics
sensitivity = recall_score(y_test, y_pred, average="weighted")  # Sensitivity = Recall
specificity = recall_score(y_test, y_pred, average="weighted", pos_label=0)  # Specificity for class 0
ppv = precision_score(y_test, y_pred, average="weighted")  # Positive Predictive Value
npv = (specificity * (y_test == 0).sum()) / ((specificity * (y_test == 0).sum()) + (1 - sensitivity) * (y_test == 1).sum())  # Negative Predictive Value
kappa = cohen_kappa_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob, multi_class="ovo", average="macro")
logloss_score = log_loss(y_test, y_pred_proba)

#  Model Evaluation
print("\n🔹 Accuracy Score:", accuracy_score(y_test, y_pred))
print("\n🔹 Classification Report:\n", classification_report(y_test, y_pred))
print(f"✅ Sensitivity (Recall): {sensitivity:.4f}")
print(f"✅ Specificity: {specificity:.4f}")
print(f"✅ Positive Predictive Value (PPV): {ppv:.4f}")
print(f"✅ Negative Predictive Value (NPV): {npv:.4f}")
print(f"✅ Kappa Score: {kappa:.4f}")
print(f"✅ ROC-AUC Score: {roc_auc:.4f}")
print(f"✅ Multi-Class Log Loss: {logloss_score:.4f}")


#  Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Healthy', 'NAFLD', 'AFLD', 'Non-Healthy'], yticklabels=['Healthy', 'NAFLD', 'AFLD', 'Non-Healthy'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

#  Multi-Class ROC Curve
classes = ["Healthy", "NAFLD", "AFLD", "Non-Healthy"]
n_classes = len(classes)

# Binarize labels
y_test_binarized = label_binarize(y_test, classes=[0, 1, 2, 3])

# Compute ROC Curve
fpr, tpr, roc_auc = dict(), dict(), dict()
colors = ['b', 'r', 'g', 'purple']  # Colors for each class
line_styles = ['-', '--', '-.', ':']  # Different line styles

plt.figure(figsize=(7, 5))

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_pred_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

    plt.plot(
        fpr[i], tpr[i], color=colors[i], linestyle=line_styles[i], lw=2, alpha=0.7,
        label=f'ROC curve for {classes[i]} (AUC = {roc_auc[i]:.2f})'
    )

# Reference diagonal line
plt.plot([0, 1], [0, 1], color='black', linestyle='dashed', lw=1)

# Final plot customization
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-Class ROC Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()
