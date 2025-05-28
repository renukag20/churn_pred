import streamlit as st
import pandas as pd
import joblib           # ← use joblib
import base64
import os

# --------- Background Image Setup (Top Banner Only) ---------
# def set_background(image_file: str) -> None:
#     if not os.path.exists(image_file):
#         st.warning(f"⚠️ Background image not found: {image_file}")
#         return
#     with open(image_file, "rb") as image:
#         encoded = base64.b64encode(image.read()).decode()
#     st.markdown(
#         f"""
#         <style>
#         /* Top banner background */
#         .top-bg {{
#             background-image: url("data:image/jpeg;base64,{encoded}");
#             background-position: top center;
#             background-repeat: no-repeat;
#             background-size: contain;
#             height: 250px;
#             width: 100%;
#             margin-bottom: 20px;
#         }}
#         /* App background */
#         .stApp {{
#             background-color: black;
#         }}
#         </style>
#         <div class="top-bg"></div>
#         """,
#         unsafe_allow_html=True
#     )

# # Top banner image (edit path if you store it elsewhere)
# set_background("bg churn.jpg")

# ─────────────────────────────────────────────
# 1. Load trained model
# ─────────────────────────────────────────────
# MODEL_PATH = r"C:\Users\HARSH\Desktop\Streamlit\Fraud Analysis\best_rf_low.joblib"

# @st.cache_resource(show_spinner=False)
# def load_model(path: str):
#     if not os.path.exists(path):
#         st.error(f"❌ Model file not found: {path}")
#         st.stop()
#     return joblib.load(path)

model = joblib.load("best_rf_low.joblib")

# ─────────────────────────────────────────────
# 2. UI widgets
# ─────────────────────────────────────────────
st.title("📉 Customer Churn Prediction")
st.write("Fill in the customer details below, then click **Predict**.")

age              = st.number_input("Age",              18, 100, 30)
amount_spent     = st.number_input("Amount Spent",     0.0, 10000.0, 500.0)
login_frequency  = st.number_input("Login Frequency",  0,   100,    10)

gender           = st.selectbox("Gender",            ["F", "M"])
marital_status   = st.selectbox("Marital Status",    ["Widowed", "Divorced", "Married", "Single"])
income_level     = st.selectbox("Income Level",      ["High", "Medium", "Low"])
product_category = st.selectbox("Product Category",  ["Books", "Electronics", "Groceries", "Clothing", "Furniture"])
interaction_type = st.selectbox("Interaction Type",  ["Feedback", "Complaint", "Inquiry"])
resolution_status= st.selectbox("Resolution Status", ["Resolved", "Unresolved"])
service_usage    = st.selectbox("Service Usage",     ["Online Banking", "Mobile App", "Website"])

# ─────────────────────────────────────────────
# 3. Raw input DataFrame (for display)
# ─────────────────────────────────────────────
input_raw = pd.DataFrame([{
    "Age":              age,
    "AmountSpent":      amount_spent,
    "LoginFrequency":   login_frequency,
    "Gender":           gender,
    "MaritalStatus":    marital_status,
    "IncomeLevel":      income_level,
    "ProductCategory":  product_category,
    "InteractionType":  interaction_type,
    "ResolutionStatus": resolution_status,
    "ServiceUsage":     service_usage,
}])

# ─────────────────────────────────────────────
# 4. Manual label-encoding (must match training)
# ─────────────────────────────────────────────
gender_map       = {"F": 0, "M": 1}
marital_map      = {"Widowed": 3, "Divorced": 0, "Married": 1, "Single": 2}
income_map       = {"High": 0, "Medium": 2, "Low": 1}
product_map      = {"Books": 0, "Electronics": 2, "Groceries": 4,
                    "Clothing": 1, "Furniture": 3}
interaction_map  = {"Feedback": 1, "Complaint": 0, "Inquiry": 2}
resolution_map   = {"Resolved": 1, "Unresolved": 0}
service_map      = {"Online Banking": 0, "Mobile App": 1, "Website": 2}

input_enc = input_raw.copy()
input_enc["Gender"]           = input_raw["Gender"].map(gender_map)
input_enc["MaritalStatus"]    = input_raw["MaritalStatus"].map(marital_map)
input_enc["IncomeLevel"]      = input_raw["IncomeLevel"].map(income_map)
input_enc["ProductCategory"]  = input_raw["ProductCategory"].map(product_map)
input_enc["InteractionType"]  = input_raw["InteractionType"].map(interaction_map)
input_enc["ResolutionStatus"] = input_raw["ResolutionStatus"].map(resolution_map)
input_enc["ServiceUsage"]     = input_raw["ServiceUsage"].map(service_map)

# ─────────────────────────────────────────────
# 5. Prediction
# ─────────────────────────────────────────────
if st.button("Predict Churn"):
    try:
        pred = model.predict(input_enc)[0]

        st.subheader("Prediction")
        st.write("**Result:**", "Churn" if pred == 1 else "No Churn")

        with st.expander("🔍 See Input Data"):
            st.dataframe(input_raw)

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")

# C:\Users\HARSH\Desktop\Streamlit\Fraud Analysis\Churn>
