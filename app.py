import joblib


@st.cache_resource
def load_models():
    scaler = joblib.load('scaler.pkl')
    log_model = joblib.load('log_model.pkl')