import os
 
import joblib
import numpy as np
import pandas as pd
import streamlit as st
 
# PAGE SETTINGS
st.set_page_config(
    page_title="HomePrice | Real Estate Price Prediction",
    page_icon="🏠",
    layout="wide"
)
 
# WEBSITE DESIGN
st.markdown("""
<style>
.stApp { background-color: #F7F9FC; }
.block-container { max-width: 1100px; padding-top: 2rem; padding-bottom: 3rem; }
h1, h2, h3 { color: #142D45; font-family: Arial, sans-serif; }
.hero { background-color: #142D45; padding: 50px 40px; border-radius: 16px; margin-bottom: 30px; }
.hero h1 { color: white; font-size: 42px; margin-bottom: 15px; }
.hero p { color: #DCE6F0; font-size: 17px; line-height: 1.7; }
.hero-label { color: #D9C8E5; font-size: 12px; letter-spacing: 2px; font-weight: bold; }
div.stButton > button {
    background-color: #142D45; color: white; border-radius: 8px;
    padding: 12px 25px; border: none; font-weight: bold;
}
div.stButton > button:hover { background-color: #244C70; color: white; }
.result-card { background-color: #142D45; border-radius: 16px; padding: 30px 35px; margin: 10px 0 20px 0; }
.result-card .label { color: #D9C8E5; font-size: 12px; letter-spacing: 2px; font-weight: bold; }
.result-card .price { color: white; font-size: 44px; font-weight: bold; margin: 5px 0; }
.result-card .range { color: #DCE6F0; font-size: 17px; }
</style>
""", unsafe_allow_html=True)
 
 
# LOAD TRAINED MODELS
@st.cache_resource
def load_models():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tool_models.joblib")
    return joblib.load(path)
 
 
A = load_models()
D = A["defaults"]
M = A["metrics"]
 
# HEADER
st.markdown("## 🏠 HomePrice")
st.caption("REAL ESTATE PRICE PREDICTION PLATFORM")
 
# HERO SECTION
st.markdown("""
<div class="hero">
<div class="hero-label">SMART REAL ESTATE ANALYTICS</div>
<h1>Discover What Your Home Could Be Worth.</h1>
<p>
A data-driven prediction tool designed to help
real estate agents and homebuyers make more
informed property pricing decisions.
</p>
</div>
""", unsafe_allow_html=True)
 
# PROPERTY INFORMATION
st.header("Estimate Your Property Value")
st.write(
    "Enter your property's characteristics below "
    "to begin your home value analysis. "
    "Covers single-family homes in Cook County, IL."
)
st.divider()
 
col1, col2 = st.columns(2)
 
with col1:
    sqft = st.number_input("Living Area (Square Feet)", min_value=400, max_value=10000,
                           value=int(round(D["sqft"], -1)), step=50)
    bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=10, value=3)
    full_baths = st.number_input("Number of Full Bathrooms", min_value=1, max_value=8, value=2)
 
with col2:
    land = st.number_input("Lot Size (Square Feet)", min_value=500, max_value=100000,
                           value=int(round(D["land"], -2)), step=250)
    age = st.number_input("Property Age (Years)", min_value=1, max_value=170, value=int(D["age"]))
    half_baths = st.number_input("Number of Half Bathrooms", min_value=0, max_value=4, value=0)
 
# LOCATION INFORMATION
st.subheader("Property Location")
zip_code = st.selectbox("Select ZIP Code", A["zip_choices"])
 
st.divider()
 
if st.button("Get Property Estimate"):
    row = pd.DataFrame([{
        "building_square_feet": float(sqft), "land_square_feet": float(land),
        "bedrooms": float(bedrooms), "full_baths": float(full_baths),
        "half_baths": float(half_baths), "age": float(age),
        "sale_year": float(A["tool_year"]), "zip": str(zip_code),
    }])
 
    # Features for Model 1 (linear) and Model 2 (logistic)
    lin = row.copy()
    lin["log_sqft"] = np.log(lin["building_square_feet"])
    lin["log_land"] = np.log(lin["land_square_feet"].clip(lower=1))
    lin["log_age"] = np.log(lin["age"])
    lin = lin[A["lin_cols"]]
 
    # Features for the extra LightGBM model
    gbm_X = row[A["gbm_cols"]].copy()
    gbm_X["zip"] = pd.Categorical(gbm_X["zip"], categories=A["zip_categories"])
 
    # Model 1: Multiple Linear Regression (price + 80% range)
    log1 = float(A["model1"].predict(lin)[0])
    price1 = float(np.exp(log1))
    low = float(np.exp(log1 + A["resid_lo"]))
    high = float(np.exp(log1 + A["resid_hi"]))
 
    # Model 2: Logistic Regression (above-median probability)
    p_above = float(A["model2"].predict_proba(lin)[0, 1])
 
    # Extra model: LightGBM (second opinion)
    price_g = float(np.exp(A["gbm"].predict(gbm_X)[0]))
 
    # RESULTS
    st.markdown(f"""
    <div class="result-card">
    <div class="label">ESTIMATED SALE PRICE</div>
    <div class="price">${price1:,.0f}</div>
    <div class="range">Likely range (80% confidence): ${low:,.0f} to ${high:,.0f}</div>
    </div>
    """, unsafe_allow_html=True)
 
    r1, r2 = st.columns(2)
 
    with r1:
        st.subheader("Market Position")
        st.progress(min(max(p_above, 0.0), 1.0))
        verdict = "above" if p_above >= 0.5 else "below"
        st.write(
            f"**{p_above:.0%} chance** this home sells above the county median "
            f"(${A['median_price']:,.0f}). It is more likely to sell **{verdict}** the median."
        )
 
    with r2:
        st.subheader("Second Opinion")
        diff = (price_g - price1) / price1
        st.metric("LightGBM estimate", f"${price_g:,.0f}", f"{diff:+.0%} vs main estimate",
                  delta_color="off")
        if low <= price_g <= high and abs(diff) <= 0.10:
            st.success("The models agree.")
        elif low <= price_g <= high:
            st.info("Within the range, but the estimates differ noticeably.")
        else:
            st.warning("The models disagree. Verify with comparable sales.")
 
    with st.expander("What drives this estimate?"):
        c = A["m1_coefs"]
        st.markdown(
            "**Price effect of each factor (holding the others equal):**\n"
            f"- +1% living area: about **{c['log_sqft']:+.2f}%** price\n"
            f"- +1% lot size: about **{c['log_land']:+.2f}%** price\n"
            f"- +1% age: about **{c['log_age']:+.2f}%** price\n"
            f"- +1 bedroom: about **{(np.exp(c['bedrooms']) - 1) * 100:+.1f}%** price\n"
            f"- +1 full bath: about **{(np.exp(c['full_baths']) - 1) * 100:+.1f}%** price\n"
            f"- +1 half bath: about **{(np.exp(c['half_baths']) - 1) * 100:+.1f}%** price"
        )
 
    with st.expander("About the models and their limits"):
        m1, m2, mg = M["model1"], M["model2"], M["gbm"]
        st.write(
            f"- Trained on {M['n_train']:,} Cook County single-family sales (2013-2019); "
            f"estimates are as of {A['tool_year']}.\n"
            f"- **Main estimate (Linear Regression):** typical error about {m1['median_pct_error']:.0%}; "
            f"the 80% range held the true price {m1['range_coverage']:.0%} of the time on held-out homes.\n"
            f"- **Market position (Logistic Regression):** classifies above/below median correctly "
            f"{m2['accuracy']:.0%} of the time (AUC {m2['auc']:.2f}).\n"
            f"- **Second opinion (LightGBM):** typical error about {mg['median_pct_error']:.0%}.\n"
            "- Condition, renovations, and the exact street are not captured. "
            "Use this as a starting point, not an appraisal."
        )
 
# FOOTER
st.divider()
st.caption(
    "HomePrice | BUSA310 Project 1 | DePauw University"
)
