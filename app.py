"""
Customer Personality Analysis - Campaign Response Predictor
Streamlit web app.

Run locally with:   streamlit run app.py
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ------------------------------------------------------------------
# Page setup
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Campaign Response Predictor",
    page_icon="📊",
    layout="wide",
)


# ------------------------------------------------------------------
# Load the saved model bundle (cached so it loads only once)
# ------------------------------------------------------------------
@st.cache_resource
def load_bundle():
    return joblib.load("customer_response_model.pkl")


try:
    bundle = load_bundle()
except FileNotFoundError:
    st.error(
        "Could not find **customer_response_model.pkl**. "
        "Run the last cell of the notebook to create it, and keep it in this same folder."
    )
    st.stop()

model = bundle["model"]
model_name = bundle["model_name"]
scaler = bundle["scaler"]
feature_columns = bundle["feature_columns"]
defaults = bundle["defaults"]

# Work out which category columns exist, so the app matches the training data
edu_cols = [c for c in feature_columns if c.startswith("Education_")]
edu_options = [c.replace("Education_", "") for c in edu_cols]
# the category dropped by drop_first=True is not in the columns; offer it as "Other"
edu_options = edu_options + ["Other"]
has_living_with = "Living_With_Single" in feature_columns


# ------------------------------------------------------------------
# Header
# ------------------------------------------------------------------
st.title("Customer Personality Analysis")
st.subheader("Will this customer respond to the marketing campaign?")
st.caption(
    f"Prediction model: **{model_name}**  ·  "
    f"Test accuracy: **{bundle['accuracy']:.1%}**  ·  "
    f"ROC-AUC: **{bundle['roc_auc']:.3f}**"
)
st.divider()

st.write("Enter the customer's details below, then click **Predict response**.")

# ------------------------------------------------------------------
# Input form
# ------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["Customer profile", "Spending", "Purchases & history"])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        age = st.slider("Age", 18, 90, 45)
        income = st.number_input(
            "Yearly household income", min_value=0, max_value=200000, value=52000, step=1000
        )
        education = st.selectbox("Education", edu_options, index=0)
    with c2:
        kidhome = st.selectbox("Young children at home", [0, 1, 2], index=0)
        teenhome = st.selectbox("Teenagers at home", [0, 1, 2], index=0)
        if has_living_with:
            living = st.radio("Living situation", ["With partner", "Single"], horizontal=True)
        else:
            living = "With partner"

with tab2:
    st.caption("Amount spent in the last 2 years")
    c1, c2, c3 = st.columns(3)
    with c1:
        mnt_wines = st.number_input("Wines", 0, 2000, 300, step=10)
        mnt_fruits = st.number_input("Fruits", 0, 500, 25, step=5)
    with c2:
        mnt_meat = st.number_input("Meat products", 0, 2000, 160, step=10)
        mnt_fish = st.number_input("Fish products", 0, 500, 35, step=5)
    with c3:
        mnt_sweet = st.number_input("Sweet products", 0, 500, 27, step=5)
        mnt_gold = st.number_input("Gold products", 0, 500, 44, step=5)

with tab3:
    c1, c2 = st.columns(2)
    with c1:
        st.caption("Number of purchases")
        num_web = st.slider("Website purchases", 0, 30, 4)
        num_catalog = st.slider("Catalogue purchases", 0, 30, 2)
        num_store = st.slider("Store purchases", 0, 20, 5)
        num_deals = st.slider("Purchases using a discount", 0, 20, 2)
    with c2:
        st.caption("Engagement and history")
        web_visits = st.slider("Website visits last month", 0, 20, 5)
        recency = st.slider("Days since last purchase", 0, 100, 49)
        tenure = st.slider("Days as a customer", 0, 1000, 350)
        accepted_before = st.slider("Previous campaigns accepted (out of 5)", 0, 5, 0)
        complained = st.checkbox("Made a complaint in the last 2 years")

st.divider()

# ------------------------------------------------------------------
# Build the feature row - must match training columns exactly
# ------------------------------------------------------------------
def build_input_row():
    # start from the training medians so nothing is left empty
    row = {col: defaults.get(col, 0) for col in feature_columns}

    total_spending = mnt_wines + mnt_fruits + mnt_meat + mnt_fish + mnt_sweet + mnt_gold
    total_purchases = num_web + num_catalog + num_store + num_deals

    values = {
        "Income": income,
        "Kidhome": kidhome,
        "Teenhome": teenhome,
        "Recency": recency,
        "MntWines": mnt_wines,
        "MntFruits": mnt_fruits,
        "MntMeatProducts": mnt_meat,
        "MntFishProducts": mnt_fish,
        "MntSweetProducts": mnt_sweet,
        "MntGoldProds": mnt_gold,
        "NumDealsPurchases": num_deals,
        "NumWebPurchases": num_web,
        "NumCatalogPurchases": num_catalog,
        "NumStorePurchases": num_store,
        "NumWebVisitsMonth": web_visits,
        "Complain": int(complained),
        "Age": age,
        "Total_Spending": total_spending,
        "Total_Purchases": total_purchases,
        "Children": kidhome + teenhome,
        "Total_Accepted": accepted_before,
        "Customer_Days": tenure,
    }
    for k, v in values.items():
        if k in row:
            row[k] = v

    # spread "previous campaigns accepted" across the 5 campaign flags
    for i in range(1, 6):
        col = f"AcceptedCmp{i}"
        if col in row:
            row[col] = 1 if i <= accepted_before else 0

    # education one-hot
    for col in edu_cols:
        row[col] = 0
    chosen = f"Education_{education}"
    if chosen in row:
        row[chosen] = 1  # "Other" leaves all education columns at 0

    # living situation
    if has_living_with:
        row["Living_With_Single"] = 1 if living == "Single" else 0

    return pd.DataFrame([row])[feature_columns]


# ------------------------------------------------------------------
# Predict
# ------------------------------------------------------------------
if st.button("Predict response", type="primary", use_container_width=True):
    X_new = build_input_row()
    X_new_scaled = scaler.transform(X_new)

    prediction = model.predict(X_new_scaled)[0]
    probability = model.predict_proba(X_new_scaled)[0][1]

    c1, c2 = st.columns([1, 1])

    with c1:
        if prediction == 1:
            st.success("### Likely to respond")
            st.write("This customer is a good target for the campaign.")
        else:
            st.warning("### Unlikely to respond")
            st.write("This customer is a lower priority for the campaign.")

    with c2:
        st.metric("Chance of responding", f"{probability:.1%}")
        st.progress(float(probability))

    with st.expander("What the model was given"):
        st.dataframe(X_new.T.rename(columns={0: "Value"}), use_container_width=True)

    st.caption(
        "This is a decision-support tool based on historical data, not a guarantee. "
        "Predictions should support marketing decisions, not replace them."
    )

# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
with st.sidebar:
    st.header("About this project")
    st.write(
        "This app uses a machine learning model trained on the "
        "**Customer Personality Analysis** dataset to predict whether a customer "
        "will accept a marketing offer."
    )
    st.write("**How it was built**")
    st.write(
        "- Data cleaned and new features created\n"
        "- Five models trained and compared\n"
        "- Best model chosen by ROC-AUC\n"
        "- Model saved and served through this app"
    )
    st.divider()
    st.write(f"**Model in use:** {model_name}")
    st.write(f"**Accuracy:** {bundle['accuracy']:.1%}")
    st.write(f"**ROC-AUC:** {bundle['roc_auc']:.3f}")
    st.caption("Dataset source: Kaggle")
