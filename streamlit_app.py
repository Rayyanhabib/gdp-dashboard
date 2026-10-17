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

# =====================================================================
# USER RESEARCH - EDIT THIS BLOCK WITH YOUR OWN INTERVIEW NOTES
# Write everything in your own words, then change SHOW_RESEARCH to True.
# Use roles only (no names) and no sensitive personal information.
# =====================================================================
SHOW_RESEARCH = False
NUM_INTERVIEWS = 5
INTERVIEWEE_ROLES = [
    "EDIT: role 1 (for example, realtor)",
    "EDIT: role 2",
]
KEY_NEEDS = [
    "EDIT: main need or concern you heard",
    "EDIT: another need or concern",
]
DESIGN_DECISIONS = [   # (what interviewees said, what you changed in the tool)
    ("EDIT: what you heard", "EDIT: the specific change you made to the inputs, outputs, or design"),
    ("EDIT: a second finding", "EDIT: a second change"),
]

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

    # Features for the linear and logistic models
    lin = row.copy()
    lin["log_sqft"] = np.log(lin["building_square_feet"])
    lin["log_land"] = np.log(lin["land_square_feet"].clip(lower=1))
    lin["log_age"] = np.log(lin["age"])
    lin = lin[A["lin_cols"]]

    # Features for the LightGBM model
    gbm_X = row[A["gbm_cols"]].copy()
    gbm_X["zip"] = pd.Categorical(gbm_X["zip"], categories=A["zip_categories"])

    # MAIN MODEL - Model 2: Logistic Regression (above-median probability)
    p_above = float(A["model2"].predict_proba(lin)[0, 1])

    # BACKGROUND MODELS
    log1 = float(A["model1"].predict(lin)[0])
    price1 = float(np.exp(log1))
    low = float(np.exp(log1 + A["resid_lo"]))
    high = float(np.exp(log1 + A["resid_hi"]))
    price_g = float(np.exp(A["gbm"].predict(gbm_X)[0]))

    if p_above >= 0.995:
        p_text = ">99%"
    elif p_above <= 0.005:
        p_text = "<1%"
    else:
        p_text = f"{p_above:.0%}"
    verdict = "ABOVE" if p_above >= 0.5 else "BELOW"

    # RESULT (logistic regression)
    st.markdown(f"""
    <div class="result-card">
    <div class="label">MARKET POSITION</div>
    <div class="price">{p_text} chance of selling above the median</div>
    <div class="range">This home is more likely to sell <b>{verdict}</b> the county median price of ${A['median_price']:,.0f}.</div>
    </div>
    """, unsafe_allow_html=True)
    st.progress(min(max(p_above, 0.0), 1.0))
    if 0.35 <= p_above <= 0.65:
        st.info("Close call: this home sits near the median, so the prediction is less certain.")

    # BACKGROUND MODELS (supporting evidence)
    with st.expander("Predicted sale price (linear regression and LightGBM)", expanded=True):
        st.markdown("**Linear regression estimate**")
        st.metric("Estimated sale price", f"${price1:,.0f}")
        st.write(f"Likely range (80% confidence): ${low:,.0f} to ${high:,.0f}")
        st.markdown("**LightGBM second opinion**")
        diff = (price_g - price1) / price1
        st.metric("LightGBM estimate", f"${price_g:,.0f}", f"{diff:+.0%} vs linear regression",
                  delta_color="off")
        if low <= price_g <= high and abs(diff) <= 0.10:
            st.success("The price models agree.")
        elif low <= price_g <= high:
            st.info("Within the range, but the price estimates differ noticeably.")
        else:
            st.warning("The price models disagree. Verify with comparable sales.")
        above_prices = [p for p in (price1, price_g)]
        agree = all((p >= A["median_price"]) == (p_above >= 0.5) for p in above_prices)
        st.caption("Both price estimates fall on the same side of the median as the logistic result."
                   if agree else
                   "A price estimate falls on the other side of the median from the logistic result.")

    with st.expander("What drives the odds? (logistic regression)"):
        labels = {"log_sqft": "Living area (sq ft)", "log_land": "Lot size", "log_age": "Property age",
                  "bedrooms": "Bedrooms", "full_baths": "Full bathrooms",
                  "half_baths": "Half bathrooms", "sale_year": "Sale year (market trend)"}
        o = pd.Series(A["m2_odds"]).sort_values(ascending=False)
        o.index = [labels.get(i, i) for i in o.index]
        st.write("Each value is the multiplier on the odds of selling above the median for a "
                 "one-standard-deviation increase, holding the other factors equal. "
                 "Above 1 raises the odds, below 1 lowers them.")
        st.dataframe(o.round(2).rename("Odds multiplier"))

    with st.expander("About the models and their limits"):
        m1, m2, mg = M["model1"], M["model2"], M["gbm"]
        st.write(
            f"- Trained on {M['n_train']:,} Cook County single-family sales (2013-2019); "
            f"estimates are as of {A['tool_year']}.\n"
            f"- **Logistic regression (main):** classifies above/below median correctly "
            f"{m2['accuracy']:.0%} of the time on held-out homes (precision {m2['precision']:.0%}, "
            f"recall {m2['recall']:.0%}, AUC {m2['auc']:.2f}).\n"
            f"- **Linear regression (supporting):** typical price error about {m1['median_pct_error']:.0%}; "
            f"the 80% range held the true price {m1['range_coverage']:.0%} of the time.\n"
            f"- **LightGBM (supporting):** typical price error about {mg['median_pct_error']:.0%}.\n"
            "- Condition, renovations, and the exact street are not captured. "
            "Use this as a starting point, not an appraisal."
        )

# USER RESEARCH SECTION
if SHOW_RESEARCH:
    st.divider()
    st.header("Built From User Research")
    st.write(f"Before designing this tool, we interviewed {NUM_INTERVIEWS} people: "
             + ", ".join(INTERVIEWEE_ROLES) + ".")
    st.subheader("What we heard")
    for need in KEY_NEEDS:
        st.markdown(f"- {need}")
    st.subheader("How it shaped the tool")
    for heard, changed in DESIGN_DECISIONS:
        c1, c2 = st.columns(2)
        c1.markdown(f"**We heard:** {heard}")
        c2.markdown(f"**We changed:** {changed}")

# FOOTER
st.divider()
st.caption(
    "HomePrice | BUSA310 Project 1 | DePauw University"
)
