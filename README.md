# 🏠 EstimateEstate - Mumbai

An AI-powered house price prediction application for Mumbai, India. This application uses machine learning regression models to accurately estimate property prices based on various features.

## 📋 Features

- **AI Price Prediction**: Predict house prices based on multiple features using advanced regression models
- **Interactive Maps**: Visual location display with Folium integration showing exact locality
- **Beautiful UI**: Modern, gradient-based design with animations and hover effects
- **Location Analysis**: Support for 400+ localities across Mumbai
- **Dataset Exploration**: Interactive visualizations and data analysis
- **Multiple Models**: Ensemble of regression algorithms (Random Forest, Gradient Boosting, etc.)
- **Real-time Insights**: Compare predicted prices with market averages and similar properties

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone or download this repository**

```bash
cd EstimateEstate-Mumbai
```

2. **Install required packages**

```bash
pip install -r requirements.txt
```

### Running the Application

1. **Train the model** (if not already trained)

```bash
python train_model.py
```

This will:
- Load the dataset
- Preprocess the data
- Train multiple regression models
- Select the best performing model
- Save the model and encoder as `model.pkl` and `encoder.pkl`

2. **Run the Streamlit application**

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

## 📊 Dataset

The application uses a cleaned Mumbai house price dataset with the following features:

- **Title**: Property name
- **Area**: Property area in square feet
- **Locality**: Location/area in Mumbai
- **Property Type**: Apartment, Villa, Independent House, etc.
- **Bedrooms**: Number of bedrooms
- **Bathrooms**: Number of bathrooms
- **Balconies**: Number of balconies
- **Furnished Status**: Unfurnished, Semi-Furnished, or Furnished
- **Age**: Property age in years
- **Total Floors**: Total floors in the building
- **Latitude/Longitude**: GPS coordinates

## 🎯 How to Use

1. **Navigate to the "Price Predictor" tab**
2. **Fill in the property details:**
   - Select property type
   - Enter area in square feet
   - Choose locality from the dropdown
   - Adjust number of bedrooms, bathrooms, and balconies
   - Set furnished status
   - Enter property age and total floors
3. **Click "Predict Price"** to get the estimated price
4. **Review insights** including price per square foot and market comparisons

## 📈 Model Performance

The application trains and evaluates multiple regression models:

- Linear Regression
- Ridge Regression
- Lasso Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor

The best performing model is automatically selected based on R² score, MAE, and RMSE metrics.

## 📁 Project Structure

```
EstimateEstate-Mumbai/
├── app.py                      # Main Streamlit application
├── train_model.py              # Model training script
├── model.pkl                   # Trained model (generated)
├── encoder.pkl                 # Data encoder (generated)
├── mumbai-house-price-data-cleaned.csv  # Dataset
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🛠️ Technologies Used

- **Streamlit**: Web application framework
- **Scikit-learn**: Machine learning library
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Pickle**: Model serialization

## 📝 License

This project is open source and available for personal and educational use.

## 🤝 Contributing

Contributions are welcome! Feel free to submit issues or pull requests.

## 📞 Support

For questions or issues, please refer to the repository documentation or create an issue.

---

Made with ❤️ for Mumbai real estate

