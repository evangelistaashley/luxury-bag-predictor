import streamlit as st
import pandas as pd
import joblib

#Setup & Loading Pickles
st.set_page_config(page_title="Luxury Bag Valuator", layout="centered")

@st.cache_resource
def load_assets():
    model = joblib.load('bag_cagr_model.pkl')
    encoders = joblib.load('encoders_dict.pkl')
    condition_map = joblib.load('condition_map.pkl')
    brand_model_map = joblib.load('brand_model_map.pkl')
    return model, encoders, condition_map, brand_model_map

model, encoders, condition_map, brand_model_map = load_assets()

#Dashboard Header
st.title("👜 Luxury Bag Valuation Engine")
st.markdown("Select bag details to forecast the **Compound Annual Growth Rate (CAGR)**.")
st.divider()

# Valuation Engine Inputs
brand = st.selectbox("Brand / Designer", sorted(list(brand_model_map.keys())))

available_models = sorted(brand_model_map[brand])
model_name = st.selectbox("Model Silhouette", available_models)

color = st.selectbox("Primary Color", encoders['color'].classes_)

condition_label = st.selectbox("Item Condition", options=list(condition_map.keys()), index=2)

hype_status = st.selectbox("Market Status", options=["Classic", "Trending"], index=1)
hype_map = {"Classic": 100, "Trending": 5000}

wishlist_val = hype_map[hype_status]

#Column 1: Valuation Engine Model
input_df = pd.DataFrame({
    'vendor': [encoders['vendor'].transform([brand])[0]],
    'condition': [condition_map[condition_label]],
    'color': [encoders['color'].transform([color])[0]],
    'num_wish_list': [wishlist_val],
    'model_name': [encoders['model_name'].transform([model_name])[0]]
})

prediction = model.predict(input_df)[0]

#Valuation Engine Results
st.divider()
st.subheader(f"Predicted Annual Growth for a {brand} {model_name}")
st.title(f"{prediction:.2f}%")

if prediction > 10:
    st.success("🔥 Superior Girl Math: Outperforms the S&P 500")
elif prediction > 0:
    st.info("⭐ Slow Burn: Appreciates with Time")
else:
    st.warning("✨ Short Term Gratification: Decreases with Time, But Still Cute")

st.caption(f"Analysis based on a {brand} {model_name} in {condition_label} condition.")


