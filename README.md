# Fatty-Liver-Prediction
A full-stack AI/ML project that predicts Non-Alcoholic (NAFLD) and Alcoholic Fatty Liver Disease (AFLD) using a synthetically generated medical dataset and a stacked ensemble learning model. Built using Python, it features a GUI, SHAP explainability, PDF report generation, and local database storage with SQLite.

# **🚀 Features**

📊 Synthetic Dataset Generator
Simulates 1 million realistic patient records across 4 liver conditions with gender-specific clinical parameters.

🤖 Stacked Ensemble Model
Combines XGBoost, SVM, and ANN with LightGBM as meta-learner for accurate multi-class classification.

📈 SHAP Explainability
Interprets each prediction with feature importance graphs for better clinical understanding.

🖥️ Desktop GUI (Tkinter)
User-friendly interface to input patient data, get predictions, and download medical reports.

📄 PDF Report Generator (FPDF)
Automatically generates a detailed patient report with diagnosis, parameter analysis, and recommendations.

🗃️ Database Storage (SQLite)
Stores patient data and PDFs securely in a local database with export features.

# **🧰 Technologies Used**


Languages: Python

Libraries: Pandas, NumPy, scikit-learn, XGBoost, LightGBM, TensorFlow/Keras, SHAP, FPDF

GUI: Tkinter

Database: SQLite

IDE: Google Colab & VS Code

# **📌 How It Works**


Generate synthetic patient data using defined medical parameter ranges.

Train a stacked ensemble ML model to classify liver conditions.

Input patient data via GUI.

Predict and generate an explainable PDF report.

Save results into a secure SQLite database.

# **📦 Future Enhancements**


🌐 Web deployment with Flask or Streamlit

📷 Image-based prediction using ultrasound and CNNs

📱 WhatsApp/Email integration for sharing reports

📊 Doctor dashboard with patient history

🗣️ Multilingual and voice-enabled support
