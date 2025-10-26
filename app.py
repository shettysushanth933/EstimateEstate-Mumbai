import streamlit as st
import pandas as pd
import pickle
import os
from data_preprocessor import DataPreprocessor  # Assumes data_preprocessor.py is present
import numpy as np
import folium
from streamlit_folium import st_folium

# Page configuration
st.set_page_config(
    page_title="EstimateEstate - Mumbai",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "# EstimateEstate Mumbai\nAI-powered house price prediction for Mumbai"
    }
)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'encoder' not in st.session_state:
    st.session_state.encoder = None

@st.cache_data
def load_dataset():
    """Load the Mumbai house price dataset"""
    if not os.path.exists('mumbai-house-price-data-cleaned.csv'):
        st.error("Dataset file 'mumbai-house-price-data-cleaned.csv' not found.")
        return pd.DataFrame()
    df = pd.read_csv('mumbai-house-price-data-cleaned.csv')
    return df

def load_model():
    """Load the trained model and encoder"""
    model_path = 'model.pkl'
    encoder_path = 'encoder.pkl'
    
    if os.path.exists(model_path) and os.path.exists(encoder_path):
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            with open(encoder_path, 'rb') as f:
                encoder = pickle.load(f)
            return model, encoder
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return None, None
    return None, None

def format_price(price):
    """Format price in Indian currency"""
    if price >= 1e7:
        return f"₹{price/1e7:.2f} Cr"
    elif price >= 1e5:
        return f"₹{price/1e5:.2f} Lakh"
    else:
        return f"₹{price/1e3:.2f} K"

def main():
    # --- Main Header ---
    st.title("🏠 EstimateEstate Mumbai")
    st.caption("AI-Powered House Price Prediction for Mumbai")
    
    # Load dataset for reference
    df = load_dataset()
    if df.empty:
        st.stop()
    
    # --- Sidebar ---
    with st.sidebar:
        st.header("📊 Dataset Overview")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Properties", f"{len(df):,}", help="Total properties in dataset")
        with col2:
            st.metric("Localities", f"{df['locality'].nunique()}", help="Number of unique localities")
        
        st.metric("Property Types", df['property_type'].nunique(), help="Different property types available")
        
        st.divider()
        
        # Model status
        model, encoder = load_model()
        if model is not None:
            st.success("✅ Model Loaded Successfully")
        else:
            st.warning("⚠️ Model Not Found. Run `train_model.py` first.")
        
        st.divider()
        
        st.markdown("### 📋 How to Use")
        st.markdown("""
        1. Fill in property details
        2. Click **Predict Price** button
        3. View estimated price and insights
        4. Adjust features to see price changes
        """)
        
        st.divider()
        
        st.markdown("### 💡 Tips")
        st.markdown("""
        - Select locality for accurate pricing
        - More bedrooms/bathrooms = Higher price
        - Location greatly affects price
        - Check similar properties for context
        """)
    
    # --- Main Content Tabs ---
    tab1, tab2, tab3 = st.tabs(["🏠 Price Predictor", "📊 Data Analysis", "ℹ️ About"])
    
    with tab1:
        st.header("🏠 Enter Property Details")
        
        col1, col2 = st.columns([0.6, 0.4]) # Give map column a bit more space
        
        with col1:
            # --- Location & Type Container ---
            with st.container(border=True):
                st.markdown("#### 📍 Location & Type")
                
                c1, c2 = st.columns(2)
                with c1:
                    # Property Type
                    property_type = st.selectbox(
                        "Property Type *",
                        options=sorted(df['property_type'].unique()),
                        help="Select the type of property you want to estimate"
                    )
                with c2:
                    # Area
                    area = st.number_input(
                        "Area (sqft) *",
                        min_value=100, 
                        max_value=5000,
                        value=1000,
                        step=50,
                        help="Total area of the property in square feet"
                    )
                
                # Location (Locality)
                localities = sorted(df['locality'].unique())
                locality = st.selectbox(
                    "Locality *",
                    options=localities,
                    help="Select the locality/area in Mumbai. This greatly affects pricing."
                )
                
                # Get coordinates for selected locality
                locality_data = df[df['locality'] == locality].iloc[0]
                latitude = float(locality_data['latitude'])
                longitude = float(locality_data['longitude'])

                # --- Map in Expander ---
                with st.expander("Show Location Map"):
                    map_center = [latitude, longitude]
                    m = folium.Map(
                        location=map_center,
                        zoom_start=12,
                        tiles='CartoDB positron'
                    )
                    folium.Marker(
                        location=map_center,
                        popup=f"<b>{locality}</b>",
                        tooltip=f"📍 {locality}",
                        icon=folium.Icon(color='blue', icon='home', prefix='fa')
                    ).add_to(m)
                    
                    st_folium(m, width='100%', height=250, returned_objects=[])
                    st.caption(f"📍 Coordinates: {latitude:.6f}, {longitude:.6f}")

            # --- Property Features Container ---
            with st.container(border=True):
                st.markdown("#### 🛏️ Property Features")
                
                c1, c2 = st.columns(2)
                with c1:
                    # Bedrooms
                    bedroom_num = st.slider(
                        "Bedrooms *",
                        min_value=0, max_value=15, value=2, step=1,
                        help="Number of bedrooms"
                    )
                    
                    # Bathrooms
                    bathroom_num = st.slider(
                        "Bathrooms *",
                        min_value=1, max_value=15, value=2, step=1,
                        help="Number of bathrooms"
                    )
                    
                    # Balconies
                    balcony_num = st.slider(
                        "Balconies",
                        min_value=0, max_value=15, value=1, step=1,
                        help="Number of balconies"
                    )
                
                with c2:
                    # Furnished status
                    furnished = st.selectbox(
                        "Furnishing Status *",
                        options=['Unfurnished', 'Semi-Furnished', 'Furnished'],
                        help="Furnishing status affects property value"
                    )
                    # Age
                    age = st.slider(
                        "Property Age (years)",
                        min_value=0, max_value=50, value=0,
                        help="Age of the property in years"
                    )
                    # Total Floors
                    total_floors = st.number_input(
                        "Total Floors",
                        min_value=1, max_value=50, value=1,
                        help="Total number of floors in the building"
                    )

        with col2:
            # --- Prediction Button & Output ---
            st.markdown("#### 🔮 Get Estimate")
            predict_button = st.button("Predict Price", type="primary", use_container_width=True)
            
            st.markdown("---")

            if predict_button:
                if model is None or encoder is None:
                    st.error("❌ Model not available. Please run `python train_model.py` first.")
                else:
                    try:
                        # Prepare input data
                        input_data = pd.DataFrame({
                            'property_type': [property_type], 'area': [area],
                            'locality': [locality], 'bedroom_num': [bedroom_num],
                            'bathroom_num': [bathroom_num], 'balcony_num': [balcony_num],
                            'furnished': [furnished], 'age': [age],
                            'total_floors': [total_floors],
                            'latitude': [latitude], 'longitude': [longitude]
                        })
                        
                        # Encode categorical variables
                        encoded_data = encoder.transform(input_data)
                        
                        # Make prediction
                        predicted_price = model.predict(encoded_data)[0]
                        
                        # --- Display Prediction ---
                        st.subheader("Estimated Price")
                        st.header(f"{format_price(predicted_price)}")
                        st.caption(f"(≈ ${predicted_price*0.012:,.2f} USD)")
                        
                        st.markdown("---")
                        
                        # --- Additional Metrics ---
                        st.subheader("📊 Price Analysis")
                        c1, c2, c3 = st.columns(3)
                        
                        price_per_sqft = predicted_price / area
                        c1.metric("Price per Sqft", f"₹{price_per_sqft:,.0f}")
                        
                        avg_price = df['price'].mean()
                        c2.metric("Market Average", format_price(avg_price))
                        
                        price_comparison = ((predicted_price / avg_price) - 1) * 100
                        c3.metric("vs Market Avg", f"{price_comparison:+.1f}%")
                        
                        st.markdown("---")
                        
                        # --- Market Insights ---
                        st.subheader("💡 Market Insights")
                        similar_props = df[
                            (df['property_type'] == property_type) &
                            (df['locality'] == locality) &
                            (df['bedroom_num'] == bedroom_num)
                        ]
                        
                        if len(similar_props) > 0:
                            min_price = similar_props['price'].min()
                            max_price = similar_props['price'].max()
                            mean_price = similar_props['price'].mean()
                            
                            st.markdown(f"**Similar Properties in {locality}**")
                            c1, c2, c3 = st.columns(3)
                            c1.info(f"**Minimum:**\n{format_price(min_price)}")
                            c2.info(f"**Average:**\n{format_price(mean_price)}")
                            c3.info(f"**Maximum:**\n{format_price(max_price)}")
                        else:
                            st.info("No similar properties found in the dataset for this combination.")
                        
                    except Exception as e:
                        st.error(f"Error making prediction: {str(e)}")

    with tab2:
        st.header("📊 Dataset Analysis & Visualizations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Price Distribution (Sampled)")
            st.bar_chart(df['price'].sample(min(1000, len(df))))
            
            st.markdown("#### Property Types")
            prop_type_counts = df['property_type'].value_counts()
            st.bar_chart(prop_type_counts)
        
        with col2:
            st.markdown("#### Top 10 Localities by Average Price")
            top_localities = df.groupby('locality')['price'].mean().sort_values(ascending=False).head(10)
            st.bar_chart(top_localities)
            
            st.markdown("#### Furnishing Status")
            furnished_counts = df['furnished'].value_counts()
            st.bar_chart(furnished_counts)
        
        st.divider()
        st.markdown("### 📈 Statistics Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric("Minimum Price", format_price(df['price'].min()))
        col2.metric("Average Price", format_price(df['price'].mean()))
        col3.metric("Median Price", format_price(df['price'].median()))
        col4.metric("Maximum Price", format_price(df['price'].max()))
        
        st.divider()
        st.markdown("### 📋 Sample Data")
        num_samples = st.slider("Number of samples", 5, 50, 10)
        st.dataframe(df.sample(num_samples), use_container_width=True)
    
    with tab3:
        st.header("ℹ️ About EstimateEstate Mumbai")
        
        st.markdown(f"""
        #### 🎯 Overview
        **EstimateEstate Mumbai** is an AI-powered house price prediction application that uses advanced 
        machine learning regression models to accurately estimate property prices in Mumbai, India.
        
        #### 🔧 Technology Stack
        - **Frontend**: Streamlit
        - **Machine Learning**: Scikit-learn (Random Forest, Gradient Boosting)
        - **Data Processing**: Pandas, NumPy
        - **Visualization**: Folium, Streamlit charts
        
        #### 📊 Model Performance
        The application uses an ensemble of regression models, with the best performing model 
        (typically Random Forest) achieving an **R² score of ~0.91** on the test set.
        
        #### ✨ Features
        - 🎯 Accurate price predictions
        - 📍 Location-based analysis across **{df['locality'].nunique()}** localities
        - 📊 Interactive dataset exploration
        - 🎨 Clean, responsive user interface
        - 💰 Real-time price comparisons
        - 📈 Market insights and trends
        
        #### 📝 Dataset
        - **Total Properties**: {len(df):,}
        - **Features**: Area, Location, Bedrooms, Bathrooms, Balconies, Furnishing, Age, Floors, Coordinates
        """)
        
        st.divider()
        st.markdown("### 🚀 Getting Started")
        st.markdown("""
        1. Navigate to the **Price Predictor** tab
        2. Fill in property details
        3. Click **Predict Price** to get an estimate
        4. Explore **Data Analysis** for market insights
        """)
        
        st.divider()
        st.markdown("Made with ❤️ for Mumbai real estate")

if __name__ == "__main__":
    main()