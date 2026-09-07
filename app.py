import streamlit as st
import joblib

import  numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

from transit_physics import semi_major_axis_au, transit_depth, transit_duration_hours
from light_curve import simulate_light_curve, time_days


@st.cache_resource
def load_models():
    scaler = joblib.load('scaler.pkl')
    model = joblib.load('model.pkl')
    nn_model = load_model('nn_model.keras')
    return scaler, model, nn_model

scaler, model, nn_model = load_models()

st.title("Exoplanet Transit Simulator")

# Sliders for each physical parameter
star_mass = st.slider("Star Mass (solar masses)", 0.5, 1.5, 1.0)
star_radius = st.slider("Star Radius (solar radii)", 0.5, 1.5, 1.0)
planet_radius = st.slider("Planet Radius (Jupiter radii)", 0.05, 1.5, 0.5)
period = st.slider("Period (days)", 1.0, 50.0, 10.0)
impact_param = st.slider("Impact Parameter", 0.0, 0.99, 0.0)
noise_std = st.slider("Noise Std", 0.0005, 0.02, 0.001)

time_days = np.linspace(-2, 2, 1000)


a_au = semi_major_axis_au(period, star_mass)
depth = transit_depth(planet_radius, star_radius)
duration_hours = transit_duration_hours(period, star_radius, a_au, impact_param)
flux = simulate_light_curve(time_days, 0, duration_hours, depth)

# Add noise
rng = np.random.default_rng()
flux_noisy = rng.normal(0, noise_std, size=flux.shape)

fig, ax = plt.subplots()
ax.scatter(time_days, flux_noisy, s=3, alpha=0.5)
ax.set_xlabel("time (days)")
ax.set_ylabel("relative brightness")
st.pyplot(fig)


X_input = flux_noisy.reshape(1, -1)
X_input_scaled = scaler.transform(X_input)

logreg_prob = model.predict_proba(X_input_scaled)[0][1]
nn_prob = nn_model.predict(X_input_scaled)[0][0]

st.write(f"Logistic Regression prediction: {'Transit' if logreg_prob > 0.5 else 'No transit'} (probability: {logreg_prob:.2f})")
st.write(f"Neural Network prediction: {'Transit' if nn_prob > 0.5 else 'No transit'} (probability: {nn_prob:.2f})")