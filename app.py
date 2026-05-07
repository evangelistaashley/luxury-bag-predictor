import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

#Setup & Loading Pickles
st.set_page_config(page_title="Luxury Bag Valuator", layout="wide")

@st.cache_resource
def load_assets():
    model = joblib.load('bag_cagr_model.pkl')
    encoders = joblib.load('encoders_dict.pkl')
    condition_map = joblib.load('condition_map.pkl')
    brand_model_map = joblib.load('brand_model_map.pkl')

    top_bags_data = {
    'model_name': [
        'Chanel 25', 
        'Birkin', 
        'Kelly', 
        'Constance', 
        'Speedy',
        'Keepall',
        'Garden',
        'Herbag',
        'Wallet',
        'Alma'
    ],
    'predicted_cagr': [80.000000, 1.263145, 0.453377, -0.173121, -0.193158, 
                        -0.301972, -0.390606, -0.411376, -0.582067, -0.62863 ] 
    }
    top_bags = pd.DataFrame(top_bags_data)

    return model, encoders, condition_map, brand_model_map, top_bags

model, encoders, condition_map, brand_model_map, top_bags = load_assets()

st.title("👜 Luxury Bag Valuation & Market Intelligence")
st.divider()

col1, col2 = st.columns([1, 2], gap="large")

#Column 1: Valuation Engine
with col1:  
  st.subheader("📈 Luxury Bag Valuation Engine")
  st.markdown("Select bag features to predict the **Compound Annual Growth Rate (CAGR)**.")

# Valuation Engine Inputs
  with st.container(border=True):
    brand = st.selectbox("Brand / Designer", sorted(list(brand_model_map.keys())))

    available_models = sorted(brand_model_map[brand])
    model_name = st.selectbox("Model", available_models)

    color = st.selectbox("Primary Color", encoders['color'].classes_)

    condition_label = st.selectbox("Item Condition", options=list(condition_map.keys()), index=2)

    hype_status = st.selectbox("Market Status", options=["Classic", "Trending"], index=1)
    hype_map = {"Classic": 100, "Trending": 1000}

    wishlist_val = hype_map[hype_status]

#Valuation Engine Model
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
        st.info("🌱 Slow Burn: Appreciates with Time")
    else:
        st.warning("✨ Short Term Gratification: Decreases with Time, But Still Cute")

    st.caption(f"Analysis based on a {brand} {model_name} in {condition_label} condition.")

with col2:
    # ROW 1: Feature Importance Chart
    st.subheader("💎 Key Value Drivers")
    st.markdown(
        "Model has the highest impact on value appreciation, followed by brand and condition. "
    )
    
    feature_names = ['Brand / Designer', 'Condition', 'Primary Color', 'Market Status', 'Model']
    feat_data = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=True)

    # 2. Create the Figure & Axis natively & make background transparent
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    # 3. Create Vertical Bar Chart 
    bars = ax.barh(
        feat_data['Feature'], 
        feat_data['Importance'], 
        color="#ba6a36",
        align='center'
    )

    # 4. Update tick park color, set Labels, and axis range
    ax.tick_params(axis='x', colors='white', labelsize=10)
    ax.tick_params(axis='y', colors='white', labelsize=10)

    ax.set_ylabel("Bag Features", fontsize=12, color='white', fontweight='bold') 
    ax.set_xlabel("Importance Scores", fontsize=12, color='white', fontweight='bold')      
    ax.set_xlim(0, 1)    

    # 5. Add Data Labels (Requirement 5)
    for bar in bars:
            width = bar.get_width()
            if width > 0:
                # Place the label inside the bar slightly
                ax.annotate(format(width, '.2f'),
                            (width + 0.015, bar.get_y() + bar.get_height() / 2.),
                            ha='left', va='center',
                            xytext=(0, 0),
                            textcoords='offset points',
                            fontsize=10,
                            color='white',
                            fontweight='bold')
        
    # Clean up the top and right borders
    for spine in ax.spines.values():
            spine.set_visible(False)
    
    # REMOVE GRID LINES (Requirement)
    ax.grid(False)

    # 6. Display in Streamlit
    st.pyplot(fig, clear_figure=True)

    st.divider()

    # ROW 2: Top Performers Chart
    st.subheader("🌟 Top 10 Performing Models")
    st.markdown(
        "xxx. "
    )

    # 2. Create the Figure & Axis natively & make background transparent
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    # 3. Create Vertical Bar Chart 
    bars = ax.bar(
        top_bags['model_name'], 
        top_bags['predicted_cagr'], 
        color="#c3955b",
        align='center'
    )

    # 4. Update tick park color, set labels, and axis range
    ax.tick_params(axis='x', colors='white', labelsize=8)
    ax.tick_params(axis='y', colors='white', labelsize=8)

    ax.set_ylabel("Predicted CAGR", fontsize=10, color='white', fontweight='bold') 
    ax.set_xlabel("Model", fontsize=10, color='white', fontweight='bold')      
    ax.set_ylim(-85, 85)    

    ax.axhline(0, color='white', linewidth=0.8, alpha=0.5)

    # 5. Add data labels
    for bar in bars:
        height = bar.get_height()
            
        if height >= 0:
            va_direction = 'bottom'
            offset = 5
        else:
            va_direction = 'top'
            offset = -5
    
        ax.annotate(format(height, '.2f'),
                    (bar.get_x() + bar.get_width() / 2., height),
                    ha='center', 
                    va=va_direction,      # Dynamic anchor direction
                    xytext=(0, offset),   # Dynamic pixel offset shift
                    textcoords='offset points',
                    fontsize=10,
                    color='white',
                    fontweight='bold')
            
    # Clean up the top and right borders
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)
    
    # 6. Display in Streamlit
    st.pyplot(fig, clear_figure=True)
