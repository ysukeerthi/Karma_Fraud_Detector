
import json
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc



from fraud_detector import extract_features
from main import KarmaActivity  


with open("karma_fraud_dataset_enhanced.json", "r") as f:
    data = json.load(f)

X_raw = []
y = []

for user in data:
   
    activity_log = [KarmaActivity(**act) for act in user["karma_log"]]
    features, _, _ = extract_features(activity_log, config={})
    X_raw.append(features)
    y.append(1 if user["label"] == "karma-fraud" else 0)


vec = DictVectorizer()
X = vec.fit_transform(X_raw)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)


y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_proba))


joblib.dump(clf, "model.pkl")
joblib.dump(vec, "vectorizer.pkl")
print("Model and vectorizer saved.")

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig("roc_curve.png")
plt.show()

