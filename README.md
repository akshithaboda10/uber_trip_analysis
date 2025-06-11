# 🚕 Uber Trip Analysis (NYC 2014)

This project analyzes and forecasts Uber trip demand using 2014 NYC pickup data.  
It includes time series forecasting, exploratory data analysis, and an interactive Streamlit dashboard.

---

## 📌 Objectives

- Analyze Uber pickups by hour, day, and week
- Discover peak ride times and trends
- Build machine learning models to forecast hourly trip demand
- Visualize predictions vs actuals
- Deploy insights with a Streamlit dashboard

---

## 📁 Dataset

- Source: [Uber NYC FOIL Data 2014](https://drive.google.com/file/d/1uj0xGqt3t7w6AgoTNq8SksR2Ci3bbWJ1/view)
- Files used:
uber-raw-data-apr14.csv
uber-raw-data-may14.csv
uber-raw-data-jun14.csv
uber-raw-data-jul14.csv
uber-raw-data-aug14.csv
uber-raw-data-sep14.csv

## 🗂️ Folder Structure

uber_trip_analysis/
├── app/ # Streamlit dashboard
│ ├── app.py
│ └── requirements.txt
├── data/ # Raw data (CSV files)
├── notebooks/ # Jupyter notebook
│ └── Uber_Trip_Analysis.ipynb
├── scripts/ # Python script version
│ └── uber_analysis.py
├── output/ # Output plots
│ └── plots/
├── report/ # Final PDF report
│ └── Uber_Trip_Report.pdf
├── README.md # Project overview
└── .gitignore

## ⚙️ How to Run the Project

### 📘 Jupyter Notebook
```bash
cd notebooks
jupyter notebook Uber_Trip_Analysis.ipynb

### 🐍 Python Script
cd scripts
python uber_analysis.py

### 🌐 Streamlit App
cd app
streamlit run app.py

📊 Model Performance

| Model             | MAPE (%)     |
| ----------------- | ------------ |
| XGBoost           | \~8.0%       |
| Random Forest     | \~9.5%       |
| Gradient Boosting | \~10.0%      |
| **Ensemble**      | **\~8.2%** ✅ |

✅ Key Insights
- Trip demand peaks in the evening and on weekends
- Friday and Saturday nights show the highest trip volume
- Ensemble model performs best for hourly forecasting


📜 License
This project is intended for educational and portfolio purposes only.
---

## ✅ Next Step

1. Create or open `README.md` in VS Code
2. **Paste everything above**
3. Save the file (`Ctrl + S`)
4. Push to GitHub:

git add README.md
git commit -m "📄 Added final README.md"
git push origin main

