import streamlit as st
import pandas as pd
import pickle
import os
from data_preprocessor import DataPreprocessor
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

# Enhanced Custom CSS
st.markdown("""
    <style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Root Variables */
    :root {
        --primary-color: #6366f1;
        --secondary-color: #8b5cf6;
        --accent-color: #ec4899;
        --success-color: #10b981;
        --warning-color: #f59e0b;
        --dark-bg: #0f172a;
        --light-bg: #f8fafc;
    }
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Streamlit Default Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main Header */
    .main-header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
        text-align: center;
    }
    
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        color: white;
        text-align: center;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .main-subtitle {
        font-size: 1.3rem;
        color: rgba(255, 255, 255, 0.95);
        margin: 0.5rem 0 0 0;
        font-weight: 300;
    }
    
    /* Prediction Box */
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
        animation: fadeIn 0.5s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .prediction-price {
        font-size: 3.5rem;
        font-weight: 800;
        margin: 1rem 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .prediction-label {
        font-size: 1.2rem;
        opacity: 0.9;
        margin-bottom: 0.5rem;
    }
    
    /* Cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        border-left: 5px solid #667eea;
        transition: transform 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.15);
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8fafc 0%, #e2e8f0 100%);
    }
    
    .sidebar-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 1rem;
    }
    
    /* Input Styling */
    .stSelectbox > label, .stNumberInput > label {
        font-weight: 600;
        color: #334155;
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.8rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        border-radius: 15px;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Info Boxes */
    .info-box {
        background: linear-gradient(135deg, #e0e7ff 0%, #f3e8ff 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #6366f1;
        margin: 1rem 0;
    }
    
    .success-box {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        border-left-color: #10b981;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border-left-color: #f59e0b;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 12px 24px;
        border-radius: 10px;
        font-weight: 600;
    }
    
    /* Chart containers */
    div[data-testid="stVerticalBlock"] {
        gap: 1rem;
    }
    
    /* Responsive */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem;
        }
        .prediction-price {
            font-size: 2.5rem;
        }
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'encoder' not in st.session_state:
    st.session_state.encoder = None

@st.cache_data
def load_dataset():
    """Load the Mumbai house price dataset"""
    df = pd.read_csv('mumbai-house-price-data-cleaned.csv')
    return df

def load_model():
    """Load the trained model and encoder"""
    if os.path.exists('model.pkl') and os.path.exists('encoder.pkl'):
        try:
            with open('model.pkl', 'rb') as f:
                model = pickle.load(f)
            with open('encoder.pkl', 'rb') as f:
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
    # Enhanced Header
    st.markdown("""
        <div class="main-header-container">
            <h1 class="main-header">🏠 EstimateEstate Mumbai</h1>
            <p class="main-subtitle">AI-Powered House Price Prediction for Mumbai</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Load dataset for reference
    df = load_dataset()
    
    # Enhanced Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-header">📊 Dataset Overview</div>', unsafe_allow_html=True)
        
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
            st.markdown('<div class="success-box">✅ <strong>Model Status:</strong> Loaded Successfully</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="warning-box">⚠️ <strong>Model Status:</strong> Not Found. Run train_model.py first.</div>', unsafe_allow_html=True)
        
        st.divider()
        
        st.markdown("### 📋 How to Use")
        st.markdown("""
        1. Fill in property details below
        2. Click Predict Price button
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
    
    # Main content area with enhanced tabs
    tab1, tab2, tab3 = st.tabs(["🏠 Price Predictor", "📊 Data Analysis", "ℹ️ About"])
    
    with tab1:
        st.markdown("### 🏠 Enter Property Details")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### Location & Type")
            
            # Property Type
            property_type = st.selectbox(
                "Property Type *",
                options=sorted(df['property_type'].unique()),
                help="Select the type of property you want to estimate"
            )
            
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
            
            # Create map for the locality
            st.markdown("#### 📍 Location Map")
            
            # Create a map centered on the locality
            map_center = [latitude, longitude]
            m = folium.Map(
                location=map_center,
                zoom_start=12,
                tiles='CartoDB positron'
            )
            
            # Add marker for the locality with custom icon
            folium.Marker(
                location=map_center,
                popup=f"<b>{locality}</b><br>Mumbai, India",
                tooltip=f"📍 {locality}",
                icon=folium.Icon(color='blue', icon='home', prefix='fa', icon_color='white')
            ).add_to(m)
            
            # Add a circle to show the area
            folium.CircleMarker(
                location=map_center,
                radius=100,
                popup=f"{locality}",
                color='#6366f1',
                fill=True,
                fillColor='#8b5cf6',
                fillOpacity=0.2,
                weight=2
            ).add_to(m)
            
            # Display the map
            map_data = st_folium(m, width=350, height=280, returned_objects=["last_clicked"])
            
            # Display coordinates info
            st.caption(f"📍 Coordinates: {latitude:.6f}, {longitude:.6f}")
        
        with col2:
            st.markdown("#### Property Features")
            
            # Bedrooms
            bedroom_num = st.selectbox(
                "Bedrooms *",
                options=list(range(0, 16)),
                index=2,
                help="Number of bedrooms in the property"
            )
            
            # Bathrooms
            bathroom_num = st.selectbox(
                "Bathrooms *",
                options=list(range(1, 16)),
                help="Number of bathrooms in the property"
            )
            
            # Balconies
            balcony_num = st.selectbox(
                "Balconies",
                options=list(range(0, 16)),
                help="Number of balconies"
            )
            
            # Furnished status
            furnished = st.selectbox(
                "Furnishing Status *",
                options=['Unfurnished', 'Semi-Furnished', 'Furnished'],
                help="Furnishing status affects property value"
            )
            
            st.markdown("#### Property Details")
            
            # Age
            age = st.number_input(
                "Property Age (years)",
                min_value=0,
                max_value=50,
                value=0,
                help="Age of the property in years"
            )
            
            # Total Floors
            total_floors = st.number_input(
                "Total Floors",
                min_value=1,
                max_value=50,
                value=1,
                help="Total number of floors in the building"
            )
        
        # Prediction button
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            predict_button = st.button(
                "🔮 Predict Price",
                type="primary"
            )
        
        if predict_button:
            if model is None or encoder is None:
                st.error("❌ Model not available. Please run `python train_model.py` first.")
                st.stop()
            
            try:
                # Prepare input data
                input_data = pd.DataFrame({
                    'property_type': [property_type],
                    'area': [area],
                    'locality': [locality],
                    'bedroom_num': [bedroom_num],
                    'bathroom_num': [bathroom_num],
                    'balcony_num': [balcony_num],
                    'furnished': [furnished],
                    'age': [age],
                    'total_floors': [total_floors],
                    'latitude': [latitude],
                    'longitude': [longitude]
                })
                
                # Encode categorical variables
                encoded_data = encoder.transform(input_data)
                
                # Make prediction
                predicted_price = model.predict(encoded_data)[0]
                
                # Display prediction
                st.markdown(f"""
                <div class="prediction-box">
                    <p class="prediction-label">Estimated Price</p>
                    <h1 class="prediction-price">{format_price(predicted_price)}</h1>
                    <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">(≈ ${predicted_price*0.012:,.2f})</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Additional metrics
                st.markdown("### 📊 Price Analysis")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    price_per_sqft = predicted_price / area
                    st.metric(
                        "Price per Sqft", 
                        f"₹{price_per_sqft:,.0f}",
                        help="Price per square foot"
                    )
                
                with col2:
                    avg_price = df['price'].mean()
                    st.metric(
                        "Market Average", 
                        format_price(avg_price),
                        help="Average price in the dataset"
                    )
                
                with col3:
                    price_comparison = ((predicted_price / avg_price) - 1) * 100
                    st.metric(
                        "vs Market Average", 
                        f"{price_comparison:+.1f}%",
                        help="Difference from market average"
                    )
                
                # Price range info
                st.markdown("---")
                st.markdown("### 💡 Market Insights")
                
                # Find similar properties
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
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.info(f"📉 **Minimum:** {format_price(min_price)}")
                    with col2:
                        st.info(f"📊 **Average:** {format_price(mean_price)}")
                    with col3:
                        st.info(f"📈 **Maximum:** {format_price(max_price)}")
                else:
                    st.info("📊 No similar properties found in the dataset for this combination.")
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                st.code(str(e))
    
    with tab2:
        st.markdown("### 📊 Dataset Analysis & Visualizations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Price Distribution")
            st.bar_chart(df['price'].sample(min(10000, len(df))).value_counts().head(30))
            
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
        
        # Detailed statistics
        st.markdown("---")
        st.markdown("### 📈 Statistics Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Minimum Price", format_price(df['price'].min()))
        with col2:
            st.metric("Average Price", format_price(df['price'].mean()))
        with col3:
            st.metric("Median Price", format_price(df['price'].median()))
        with col4:
            st.metric("Maximum Price", format_price(df['price'].max()))
        
        # Data table
        st.markdown("---")
        st.markdown("### 📋 Sample Data")
        num_samples = st.slider("Number of samples", 5, 50, 20)
        st.dataframe(df.sample(num_samples), height=400)
    
    with tab3:
        st.markdown("### ℹ️ About EstimateEstate Mumbai")
        
        st.markdown("""
        #### 🎯 Overview
        **EstimateEstate Mumbai** is an AI-powered house price prediction application that uses advanced 
        machine learning regression models to accurately estimate property prices in Mumbai, India.
        
        #### 🔧 Technology Stack
        - **Frontend**: Streamlit - Modern web UI framework
        - **Machine Learning**: Scikit-learn (Random Forest, Gradient Boosting)
        - **Data Processing**: Pandas, NumPy
        - **Visualization**: Interactive Streamlit charts
        
        #### 📊 Model Performance
        The application uses an ensemble of regression models:
        - Random Forest Regressor (Best Performing)
        - Gradient Boosting Regressor
        - Linear, Ridge, and Lasso Regression
        - Decision Tree Regressor
        
        **Best Model**: Random Forest with R² score of 0.91
        
        #### ✨ Features
        - 🎯 Accurate price predictions
        - 📍 Location-based analysis across 400+ localities
        - 📊 Interactive dataset exploration
        - 🎨 Beautiful, modern user interface
        - 💰 Real-time price comparisons
        - 📈 Market insights and trends
        
        #### 📝 Dataset
        - **Total Properties**: {:,}
        - **Localities**: {}
        - **Property Types**: {}
        - **Features**: Area, Location, Bedrooms, Bathrooms, Balconies, Furnishing, Age, Floors, Coordinates
        """.format(len(df), df['locality'].nunique(), df['property_type'].nunique()))
        
        st.markdown("---")
        st.markdown("### 🚀 Getting Started")
        st.markdown("""
        1. Navigate to the **Price Predictor** tab
        2. Fill in property details
        3. Click **Predict Price** to get an estimate
        4. Explore **Data Analysis** for market insights
        """)
        
        st.markdown("---")
        st.markdown("Made with ❤️ for Mumbai real estate")

if __name__ == "__main__":
    main()
