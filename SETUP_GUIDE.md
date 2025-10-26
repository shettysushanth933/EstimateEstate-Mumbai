# Setup Guide - EstimateEstate Mumbai

## Step-by-Step Commands

### 1. Create Virtual Environment (Already Done ✅)
```bash
python -m venv venv
```

### 2. Activate Virtual Environment

**For PowerShell:**
```powershell
.\venv\Scripts\Activate.ps1
```

**For Command Prompt:**
```bash
venv\Scripts\activate.bat
```

**For Git Bash:**
```bash
source venv/Scripts/activate
```

If you get an execution policy error in PowerShell, run this first:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

**New Dependencies:**
- `folium` - For interactive maps
- `streamlit-folium` - For map integration with Streamlit

### 4. Train the Model (First Time Only)
```bash
python train_model.py
```

This will take a few minutes to train the models. You'll see progress output and the best model will be saved.

### 5. Run the Application
```bash
streamlit run app.py
```

The app will automatically open in your browser at `http://localhost:8501`

### 6. Deactivate Virtual Environment (When Done)
```bash
deactivate
```

---

## Quick Start (All Commands at Once)

Open a terminal in the project directory and run:

```bash
# For PowerShell
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train_model.py
streamlit run app.py

# For Command Prompt
venv\Scripts\activate.bat
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

---

## Troubleshooting

### If Streamlit Command Not Found
```bash
pip install streamlit
```

### If Permission Denied
Run terminal as Administrator or use:
```bash
python -m pip install -r requirements.txt
```

### If model.pkl not found
Make sure you've run `python train_model.py` at least once.

---

## Project Structure After Setup

```
EstimateEstate-Mumbai/
├── venv/                    # Virtual environment
├── app.py                   # Main Streamlit app
├── train_model.py           # Model training script
├── model.pkl                # Trained model (after training)
├── encoder.pkl              # Data encoder (after training)
├── mumbai-house-price-data-cleaned.csv
├── requirements.txt
├── README.md
└── SETUP_GUIDE.md           # This file
```

