import streamlit as st
import pandas as pd
import joblib

#Setup & Loading Pickles
st.set_page_config(page_title="Luxury Bag Valuator", layout="wide")

@st.cache_resource
def load_assets():
    model = joblib.load('bag_cagr_model.pkl')
    encoders = joblib.load('encoders_dict.pkl')
    condition_map = joblib.load('condition_map.pkl')
    brand_model_map = joblib.load('brand_model_map.pkl')
    return model, encoders, condition_map, brand_model_map

model, encoders, condition_map, brand_model_map = load_assets()

st.title("👜 Luxury Market Intelligence & Valuation")
st.divider()

col1, col2 = st.columns([1, 2], gap="large")

#Column 1: Valuation Engine
with col1:  
  st.subheader("👜 Luxury Bag Valuation Engine")
  st.markdown("Select bag details to forecast the **Compound Annual Growth Rate (CAGR)**.")

# Valuation Engine Inputs
  with st.container(border=True):
    brand = st.selectbox("Brand / Designer", sorted(list(brand_model_map.keys())))

    available_models = sorted(brand_model_map[brand])
    model_name = st.selectbox("Model Silhouette", available_models)

    color = st.selectbox("Primary Color", encoders['color'].classes_)

    condition_label = st.selectbox("Item Condition", options=list(condition_map.keys()), index=2)

    hype_status = st.selectbox("Market Status", options=["Classic", "Trending"], index=1)
    hype_map = {"Classic": 100, "Trending": 1000}

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

with col2:
    # ROW 1: Feature Importance
    st.subheader("Key Value Drivers")
    feature_names = ['Brand', 'Condition', 'Color', 'Market Demand', 'Model']
    feat_data = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=True)
    
    st.bar_chart(feat_data, x='Importance', y='Feature', horizontal=True, color="#1f77b4")
    
    st.divider()
    
    # ROW 2: Top Performers
    st.subheader("Market-Wide Top Performers")
    top_10 = top_bags.sort_values(by='predicted_cagr', ascending=False).head(10)
    
    st.bar_chart(top_10, x='predicted_cagr', y='model_name', horizontal=True, color="#d33682")
