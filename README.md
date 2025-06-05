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

# **📄 Demo Screenshot**

**Programmer Defining the Sample Size:**

![image](https://github.com/user-attachments/assets/95978a75-a362-44b8-bb66-ee018c872ffd)

**Dataset Generated:**

![image](https://github.com/user-attachments/assets/fd4300bd-884a-431b-bb33-c9dccf58e314)


![image](https://github.com/user-attachments/assets/6a84308c-44e9-4777-8321-b665cb4bc3b7)

**SQLite Database to Store Patient Report**

![image](https://github.com/user-attachments/assets/47ec5bca-26d5-437d-9110-d08848f3d9b2)


**Saving the Stack Model with Scaler file After Training**

![image](https://github.com/user-attachments/assets/8115ec32-0e85-4f81-8c87-d80339650d3b)


**Model Deployee for Patient Report Generation**

![image](https://github.com/user-attachments/assets/69d7d3bc-5a98-41f7-b5d8-a66be0391fed)

**After Generating the Patient Report it automatically Stored in a database Managed by a GUI System**

![image](https://github.com/user-attachments/assets/1ffcb834-0fba-4414-9bba-f90c82dc7983)











