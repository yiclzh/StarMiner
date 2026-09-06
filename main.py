import numpy as np
from noise import generate_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

from processing import X_train_scaled, y_train, y_test, X_test_scaled, X_train

time_days = np.linspace(-2, 2, 1000)
rng = np.random.default_rng(0)

model = LogisticRegression(max_iter=2000).fit(X_train_scaled, y_train)
joblib.dump(model, 'model.pkl')
y_pred = model.predict(X_test_scaled)
print("Accuracy: ", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


import matplotlib.pyplot as plt

# find one of each class in your training set
idx_transit = np.where(y_train == 1)[0][0]
idx_no_transit = np.where(y_train == 0)[0][0]

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(X_train[idx_transit])
axes[0].set_title("label = 1 (transit)")
axes[1].plot(X_train[idx_no_transit])
axes[1].set_title("label = 0 (no transit)")
plt.show()