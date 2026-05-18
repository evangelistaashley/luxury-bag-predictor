# 👜 Luxury Bag Price Predictor & Portfolio Analytics

An end-to-end machine learning pipeline and interactive dashboard designed to forecast the **Compound Annual Growth Rate (CAGR)** of luxury handbags. This project treats luxury handbags as an alternative asset, allowing insight into key value drivers and predicted bag appreciation based on 10,000+ historical resale market records from Fashionphile.

🔗 **[Live Streamlit App Link]** *https://luxury-bag-predictor-ae1209.streamlit.app/*

---

## 📱 App Features

* **Real-Time Valuation Forecasting:** Predicts luxury bag appreciation using Brand, Model Family, Condition, and Color.
* **Key Value Drivers:** Visualizes bag feature importance using `model.feature_importances_` to show which attributes contribute most to growth.
* **High Performers:** A dynamic tracker designed to identify the 5 models with the handle both positive gains and negative market deviations.

---

## 🧠 Model Development Lifecycle

### 1. Data Acquisition / Web Scraping 
The foundational dataset for this project was custom-built using a proprietary web scraper targeting **Fashionphile**, a leader in the secondary luxury market. 
* **Automated Extraction:** Developed a Python-based scraper using `Requests` and `BeautifulSoup` to parse multi-page product listings.
* **Feature Mining:** Extracted raw HTML data for over 25,000 unique listings, capturing attributes such as `product_id`, `brand`, `title` (product description), and `meta` data information that included:
    * Condition
    * Color
    * Material
    * Savings
    * Wishlist count

### 2. Exploratory Data Analysis (EDA) & Transformation
Dataset was deduped on `product_id` filtered to only include listings >$1,500 from the top 10 represented brands.
* **Data Imputation:**
   * `retail_savings`: Imputed null values using 2 methods (1. Median for vendor + model + color + material + condition & 2. Third-party & Google Search data mapping)
   * `num_wish_list`: Imputed null values with 0
* **Data Transformation:**
   * `model_name`: Performed complex string parsing using the regex function on the title to isolate the exact bag model (e.g., "Birkin") 
   * `color` & `material`: Flattened nested listing data to consolidate multi-attribute bags into a "Mixed" category to reduce noise
   * `cagr`: Created target variable to evaluate performance over time using `value_factor` and `bag_age`
     * `bag_age`: Created `original_year` variable using third-party market archives and Google Search data to get model release year 
     * `value_factor`: Created baseline for original asset value across all brands & models after converting `retail_savings` into a percent `retail_savings_percent`

### 3. Model Development
* **Ordinal Encoding:** Manually mapped the `condition` feature (Flawed=1 to New=6) to preserve the logical hierarchy of wear-and-tear.
* **Categorical Label Encoding:** High-cardinality nominal features (`vendor`, `model_name`,`color`,`material`) were transformed using Label Encoding, with artifacts serialized as `encoders_dict.pkl` for production use.
* **Matrix Analysis:** Pearson correlation analysis was used to ensure feature independence and reduce multicollinearity. Utilizing `Seaborn` heatmaps, the features were audited to identify redundant variables (`retail_savings_percent` & `material`). 

### 4. Model Selection, Optimization, & Evaluation 
* **Selection:** **Random Forest Regressor** was selected for its ability to capture non-linear relationships and its strength against extreme model outliers.
* **Optimized:** Tuned via `RandomizedSearchCV`, focusing on `n_estimators` for computational efficiency and `max_depth` to prevent overfitting on "unicorn" models.
* **Evaluation:**
    * **R-Squared Score: 0.9106**: The model explains over 91% of the variance in bag appreciation.
    * **RMSE (Root Mean Squared Error): 5.7840**: On average, the model's CAGR predictions are within ~5.8 percentage points of the actual historical growth rate.

### 5. Insights & ML Ops
* **Key Takeaways:**
     * Model is the primary driver of value, significantly outweighing aesthetic features like color and condition. The model acts as a price floor, maintaining appreciation rates even as   general market trends fluctuate. 
     * Newer (0–2 years) released bags, such as the Chanel 25, show volatile growth due to immediate secondary market premiums, while more tenured bags exhibit more stabilized, consistent CAGR.
     * The "Condition Gap", the CAGR difference between different levels, is brand dependent. For "Holy Grail" brands like Hermes, going from Excellent to New is exponentially higher than for contemporary brands like Gucci. This suggests that the cost of good preservation and storage methods provides a measurable return on investment (ROI).
* **CI/CD:** Automated deployment via Streamlit Cloud whenever the model is retrained with new listing data (~1x quarter) and pushed to GitHub.

---

## ⚙️ Tech Stack

* **Language:** Python 3.14
* **Data Analysis & Transformation:** Pandas, Numpy
* **Machine Learning:** Scikit-Learn
* **Data Visualization:** Matplotlib, Seaborn
* **Dashboard:** Streamlit
* **Serialization:** Joblib

---

## 🕸️ Repository Structure

```text
luxury-bag-predictor/
├── app.py                      # Main Streamlit Dashboard
├── requirements.txt            # Dependencies (pandas, sklearn, matplotlib)
├── bag_cagr_model.pkl          # Serialized Random Forest Model
├── encoders_dict.pkl           # Categorical Mapping Artifacts
├── condition_map.pkl           # Condition mapping
├── brand_model_map.pkl         # Brand to model mapping
└── Handbag_Valuation.ipynb     # Full Research, EDA & Training Audit Trail
