import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from noise import generate_dataset
from sklearn.preprocessing import StandardScaler
import joblib

time_days = np.linspace(-2, 2, 1000)
rng = np.random.default_rng(0)
X, y = generate_dataset(3000, time_days, noise_std=0.001, rng=rng)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
scaler = StandardScaler()
joblib.dump(scaler, 'scaler.pkl')
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(X_train_scaled.shape)
print(X_test_scaled.shape)
print(np.bincount(y_train))
print(np.bincount(y_test))