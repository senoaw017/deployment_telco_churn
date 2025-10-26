# 📊 Customer Churn Prediction App

Aplikasi web interaktif untuk memprediksi customer churn menggunakan Machine Learning.

## 🚀 Features

- **Prediction Interface**: Input data customer dan prediksi churn probability
- **Real-time Visualization**: Gauge chart, metrics, dan visualisasi interaktif
- **Business Recommendations**: Rekomendasi aksi berdasarkan risk level
- **Model Performance**: Confusion matrix dan metrics evaluation
- **Feature Insights**: Analisis feature importance

## 📦 Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## ▶️ How to Run

```bash
streamlit run app.py
```

App akan terbuka di browser: `http://localhost:8501`

## 📁 Required Files

Pastikan file-file ini ada di folder yang sama:

1. **app.py** - Main Streamlit application
2. **churn_model.joblib** - Trained model
3. **customerchurn.csv** - Dataset asli (untuk template)
4. **requirements.txt** - Python dependencies

## 🎯 Usage

1. Isi informasi customer di form
2. Klik tombol **Predict Churn**
3. Lihat hasil prediksi:
   - Risk level (High/Low)
   - Churn probability
   - Business recommendations
4. Explore tabs lain untuk model performance dan insights

## 📊 Model Info

- **Algorithm**: LogisticRegression
- **Preprocessing**: RobustScaler + SMOTEENN
- **Accuracy**: 77.1%
- **Recall**: 80.0%
- **F2 Score**: 0.738

## 🛠️ Tech Stack

- **Streamlit** - Web framework
- **Plotly** - Interactive visualizations
- **Scikit-learn** - ML model
- **Pandas/NumPy** - Data processing

## 📝 Notes

- Model membutuhkan 19 features customer
- Pastikan format data input sesuai dengan training data
- Churn probability > 50% = High Risk

## 🎨 Customization

Edit `app.py` untuk customize:
- Colors & styling (CSS section)
- Layout & components
- Recommendations logic
- Visualizations

---

**Developed with ❤️ for Customer Retention**
