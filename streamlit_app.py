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
    .stApp {
        background-color: #F7F9FC;
    }

    .block-container {
        max-width: 1100px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        color: #142D45;
        font-family: Arial, sans-serif;
    }

    .hero {
        background-color: #142D45;
        padding: 50px 40px;
        border-radius: 16px;
        margin-bottom: 30px;
    }

    .hero h1 {
        color: white;
        font-size: 42px;
        margin-bottom: 15px;
    }

    .hero p {
        color: #DCE6F0;
        font-size: 17px;
        line-height: 1.7;
    }

    .hero-label {
        color: #D9C8E5;
        font-size: 12px;
        letter-spacing: 2px;
        font-weight: bold;
    }

    div.stButton > button {
        background-color: #142D45;
        color: white;
        border-radius: 8px;
        padding: 12px 25px;
        border: none;
        font-weight: bold;
    }

    div.stButton > button:hover {
        background-color: #244C70;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

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
    "to begin your home value analysis."
)

st.divider()

col1, col2 = st.columns(2)

with col1:
    sqft = st.number_input(
        "Living Area (Square Feet)",
        min_value=1,
        value=2000
    )

    bedrooms = st.number_input(
        "Number of Bedrooms",
        min_value=0,
        value=3
    )

with col2:
    bathrooms = st.number_input(
        "Number of Bathrooms",
        min_value=0,
        value=2
    )

    age = st.number_input(
        "Property Age (Years)",
        min_value=0,
        value=15
    )

# LOCATION INFORMATION
st.subheader("Property Location")

neighborhood = st.selectbox(
    "Select Neighborhood",
    ["Neighborhood A", "Neighborhood B", "Neighborhood C"]
)
st.divider()

if st.button("Get Property Estimate"):
    st.info(
        "Your prediction model will be connected here."
    )

# FOOTER
st.divider()

st.caption(
    "HomePrice | BUSA310 Project 1 | DePauw University"
)