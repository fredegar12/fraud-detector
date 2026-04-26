
import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

# ─────────────────────────────
# 1. PAGE CONFIG
# ─────────────────────────────
st.set_page_config(
    page_title="Fraud Detector",
    page_icon="🔍",
    layout="wide"
)

# ─────────────────────────────
# 2. LOAD PIPELINE
# ─────────────────────────────
@st.cache_resource
def load_pipeline():
    path = hf_hub_download(
        repo_id="FredFr3d/fraud-detector",
        filename="pipeline.pkl"
    )
    return joblib.load(path)

pipeline = load_pipeline()

# ─────────────────────────────
# 3. PREPROCESSING FUNCTION
# ─────────────────────────────
def preprocess(df):
    expected_cols = [
        'category', 'amt', 'gender', 'state', 'zip', 'lat', 'long',
        'city_pop', 'merch_lat', 'merch_long', 'hour', 'day_of_week'
    ]
    return df[expected_cols]

# ─────────────────────────────
# 4. SIDEBAR
# ─────────────────────────────
st.sidebar.title("🔍 Fraud Detector")
st.sidebar.write("Detect fraudulent credit card transactions using a Random Forest model.")
mode = st.sidebar.radio(
    "Choose mode:",
    ["Upload CSV", "Single Transaction"]
)

# ─────────────────────────────
# 5. CSV UPLOAD MODE
# ─────────────────────────────
if mode == "Upload CSV":
    st.title("Upload Transactions CSV")
    st.write("Upload a CSV file with transaction data to scan for fraud.")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("### Raw Data", df.head())

        try:
            X = preprocess(df.copy())

            df['prediction']        = pipeline.predict(X)
            df['fraud_probability'] = pipeline.predict_proba(X)[:, 1]

            flagged = df[df['prediction'] == 1]

            col1, col2 = st.columns(2)
            col1.metric("Total Transactions", len(df))
            col2.metric("Flagged As Fraud",   len(flagged))

            def highlight_fraud(row):
                color = 'background-color: #ffcccc' if row['prediction'] == 1 else ''
                return [color] * len(row)

            st.write("### Results")
            st.dataframe(df.style.apply(highlight_fraud, axis=1))

        except Exception as e:
            st.error(f"Error processing file: {e}")

# ─────────────────────────────
# 6. SINGLE TRANSACTION MODE
# ─────────────────────────────
elif mode == "Single Transaction":
    st.title("Check Single Transaction")
    st.write("Enter transaction details to check if it is fraudulent.")

    col1, col2, col3 = st.columns(3)

    with col1:
        amt      = st.number_input("Amount ($)", min_value=0.0, value=100.0)
        category = st.selectbox("Category", [
            'food_dining', 'gas_transport', 'grocery_net',
            'grocery_pos', 'health_fitness', 'home',
            'kids_pets', 'misc_net', 'misc_pos',
            'personal_care', 'shopping_net', 'shopping_pos',
            'entertainment', 'travel'
        ])
        gender   = st.selectbox("Gender", ["M", "F"])

    with col2:
        state    = st.text_input("State (e.g. CA)", value="CA")
        city_pop = st.number_input("City Population", min_value=0, value=50000)
        zip_code = st.number_input("Zip Code", min_value=0, value=90210)

    with col3:
        lat        = st.number_input("Customer Latitude",  value=37.77)
        long       = st.number_input("Customer Longitude", value=-122.41)
        merch_lat  = st.number_input("Merchant Latitude",  value=37.78)
        merch_long = st.number_input("Merchant Longitude", value=-122.42)

    hour        = st.slider("Hour of Day", 0, 23, 12)
    day_of_week = st.slider("Day of Week (0=Mon, 6=Sun)", 0, 6, 3)

    if st.button("Check Transaction"):
        input_data = pd.DataFrame([{
            'category':    category,
            'amt':         amt,
            'gender':      gender,
            'state':       state,
            'zip':         zip_code,
            'lat':         lat,
            'long':        long,
            'city_pop':    city_pop,
            'merch_lat':   merch_lat,
            'merch_long':  merch_long,
            'hour':        hour,
            'day_of_week': day_of_week
        }])

        X                 = preprocess(input_data)
        prediction        = pipeline.predict(X)[0]
        fraud_probability = pipeline.predict_proba(X)[0][1]

        st.write("---")
        if prediction == 1:
            st.error("🚨 FRAUDULENT TRANSACTION DETECTED")
        else:
            st.success("✅ TRANSACTION APPEARS LEGITIMATE")

        st.metric("Fraud Probability", f"{fraud_probability * 100:.1f}%")
