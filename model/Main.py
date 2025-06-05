import joblib
import numpy as np
import matplotlib.pyplot as plt
import shap
from fpdf import FPDF
import os
import pandas as pd
import sqlite3

# Load the pre-trained model and scaler
model = joblib.load("stacked_model.pkl")
scaler = joblib.load("scaler.pkl")
graph = joblib.load("xgboost_model.pkl")


# Function to get valid float input
def get_valid_float(prompt, min_value=None, max_value=None):
    while True:
        try:
            value = float(input(prompt))
            if min_value is not None and value < min_value:
                print(f"Value must be at least {min_value}.")
                continue
            if max_value is not None and value > max_value:
                print(f"Value must be at most {max_value}.")
                continue
            return value
        except ValueError:
            print("Invalid input. Please enter a valid number.")

# Function to get valid integer input
def get_valid_int(prompt, min_value=None, max_value=None):
    while True:
        try:
            value = int(input(prompt))
            if min_value is not None and value < min_value:
                print(f"Value must be at least {min_value}.")
                continue
            if max_value is not None and value > max_value:
                print(f"Value must be at most {max_value}.")
                continue
            return value
        except ValueError:
            print("Invalid input. Please enter a valid integer.")

# Function to classify risk
def classify_risk(category, alt, ast, bmi, alcohol_consumption,
                  waist_circumference, triglycerides, hdl, fbg, sbp, ldl,
                  diabetes, ggt, cdt, mcv, gender):
    if category == 1:  # NAFLD Classification
        if alt < 45 and ast < 40 and bmi < 30:
            return "Mild NAFLD"
        elif 45 <= alt < 80 or 40 <= ast < 70 or 30 <= bmi < 35:
            return "Moderate NAFLD"
        else:
            return "Severe NAFLD"

    elif category == 2:  # AFLD Classification
        if alcohol_consumption < 210 and alt < 50 and ast < 45:
            return "Mild AFLD"
        elif 210 <= alcohol_consumption < 350 or 50 <= alt < 90 or 45 <= ast < 80:
            return "Moderate AFLD"
        else:
            return "Severe AFLD"

    elif category == 3:  # Non-Healthy (Not Fatty Liver)
        metabolic_syndrome = (
            (bmi >= 30) or
            (waist_circumference > 102 if gender == 0 else waist_circumference > 88) or
            (triglycerides > 150) or
            (hdl < 40 if gender == 0 else hdl < 50) or
            (fbg >= 126) or
            (sbp > 130)
        )

        liver_dysfunction = (
            (alt > 2 * 35) or  # Assuming 35 U/L as normal ALT range
            (ast > 2 * 40) or  # Assuming 40 U/L as normal AST range
            (ggt > 80)
        )

        alcohol_related_risk = (
            (alcohol_consumption >= 210 if gender == 0 else alcohol_consumption >= 140) or
            (cdt > 2.5) or
            (mcv > 100)
        )

        diabetes_cardio_risk = (
            (diabetes == 1) or
            (fbg >= 126) or
            (ldl > 130 and (hdl < 40 if gender == 0 else hdl < 50)) or
            (sbp > 140)
        )

        risks = []
        if metabolic_syndrome:
            risks.append("Metabolic Syndrome")
        if liver_dysfunction:
            risks.append("Liver Dysfunction (Non-Fatty)")
        if alcohol_related_risk:
            risks.append("Alcohol-Related Liver Risk")
        if diabetes_cardio_risk:
            risks.append("Diabetes & Cardiovascular Risk")

        return ", ".join(risks) if risks else "General Non-Healthy Condition"

    else:
        return None  # For Healthy category


# Function to get recommendations

def get_recommendations(condition):
    recommendations = {
        "Mild NAFLD": [
            "Adopt a Balanced Diet – Increase fiber intake, reduce processed foods, and focus on lean proteins.",
            "Regular Exercise – Aim for at least 30 minutes of moderate activity 5 times a week.",
            "Routine Monitoring – Get liver function tests (ALT, AST) every 6–12 months."
        ],
        "Moderate NAFLD": [
            "Weight Management Plan – Reduce body weight by 5–10% through diet and exercise.",
            "Liver-Supportive Supplements – Consider vitamin E and omega-3 fatty acids.",
            "Medical Evaluation – Schedule a liver ultrasound and check for insulin resistance."
        ],
        "Severe NAFLD": [
            "Immediate Medical Consultation – Visit a hepatologist for fibrosis or cirrhosis evaluation.",
            "Strict Lifestyle Overhaul – Eliminate sugary drinks, high-fat foods, and refined carbs.",
            "Medication Review – Discuss alternative treatments with a doctor if on liver-impacting medication."
        ],
        "Mild AFLD": [
            "Limit Alcohol Intake – Reduce alcohol to less than 10g/day or eliminate it completely.",
            "Hydration & Nutrition – Increase water intake and consume liver-friendly foods.",
            "Liver Checkups – Monitor ALT, AST, and GGT levels every 6 months."
        ],
        "Moderate AFLD": [
            "Controlled Alcohol Reduction – Gradually reduce alcohol under medical supervision.",
            "Nutritional Counseling – Increase B vitamins, zinc, and antioxidants.",
            "Early Screening for Fibrosis – Consider FibroScan or elastography to check for liver scarring."
        ],
        "Severe AFLD": [
            "Complete Alcohol Abstinence – Immediate cessation is necessary to prevent cirrhosis.",
            "Hospital-Based Detox – Seek medically supervised detox if withdrawal symptoms are severe.",
            "Liver Transplant Evaluation – Discuss long-term treatment options with a specialist."
        ],
        "Metabolic Syndrome": [
            "Reduce Sugar & Carbs – Cut processed sugars and refined carbs to manage blood sugar and triglycerides.",
            "Regular Cardiovascular Checkups – Monitor blood pressure, cholesterol, and glucose levels every 6 months.",
            "Increased Physical Activity – Engage in strength training and aerobic exercise 5 days a week."
        ],
        "Liver Dysfunction (Non-Fatty)": [
            "Eliminate Liver Toxins – Avoid excessive medications, herbal supplements, and alcohol.",
            "Liver-Boosting Foods – Increase cruciferous vegetables, coffee (in moderation), and healthy fats.",
            "Regular Liver Function Tests – Get ALT, AST, and GGT tests every 3–6 months."
        ],
        "Alcohol-Related Liver Risk": [
            "Immediate Alcohol Reduction – Reduce intake below risk thresholds (≤140ml for females, ≤210ml for males).",
            "Regular CDT & MCV Tests – Monitor blood markers of alcohol impact every 3 months.",
            "Seek Support – Consult a liver specialist or join alcohol reduction programs if needed."
        ],
        "Diabetes & Cardiovascular Risk": [
            "Strict Glucose & Lipid Control – Follow a low-GI diet and increase fiber intake.",
            "Monitor BP & Lipids – Check blood pressure, LDL, and HDL every 3–6 months.",
            "Medications if Required – If uncontrolled, discuss statins, metformin, or BP medication with a doctor."
        ],
        "Healthy": [
            "Maintain a Balanced Diet – Continue eating whole foods, lean proteins, and healthy fats.",
            "Regular Health Screenings – Get annual checkups to ensure liver and metabolic health remains stable.",
            "Stay Active & Hydrated – Exercise 150 minutes/week and drink at least 2L of water daily."
        ],
        "General Non-Healthy Condition": [
            "Consult a General Physician – Identify underlying health risks and develop a personalized health plan.",
            "Follow a Healthy Lifestyle – Adopt a balanced diet, regular exercise, and stress management.",
            "Regular Health Monitoring – Track key health markers (BMI, BP, blood sugar) every 6 months."
        ]
    }

    return recommendations.get(condition, ["Continue Living Your Healthy Life Style."])

# Function to get parameters

def parameter(gender, bmi, triglycerides, alt, ast, fbg, waist_circumference, sbp, hdl, ldl, mcv, cdt, ggt, rbc_size, diabetes, alcohol_consumption, year_alcohol_consumption):
    if gender == 0:  # Male
        return [
            ("BMI", "18.5-24.9 kg/m²", bmi, 18.5, 24.9),
            ("Triglycerides", "40-160 mg/dL", triglycerides, 40, 160),
            ("ALT", "10-50 U/L", alt, 10, 50),
            ("AST", "10-40 U/L", ast, 10, 40),
            ("FBG", "70-99 mg/dL", fbg, 70, 99),
            ("Waist Circumference", "<102 cm", waist_circumference, None, 102),
            ("SBP", "90-120 mmHg", sbp, 90, 120),
            ("HDL-C", ">40 mg/dL", hdl, 40, None),
            ("LDL-C", "<130 mg/dL", ldl, None, 130),
            ("MCV", "80-100 fL", mcv, 80, 100),
            ("CDT", "<2.5%", cdt, None, 2.5),
            ("GGT", "8-61 U/L", ggt, 8, 61),
            ("RBC Size", "6-8 µm", rbc_size, 6, 8),
            ("Diabetes Type 2", "0 = No, 1 = Yes", diabetes, 0, 1),
            ("Pure Alcohol Consumption", "0-20 g/week", alcohol_consumption, None, 20),
            ("Year Alcohol Consumption", "0-3 years", year_alcohol_consumption, None, 3)

        ]
    else:  # Female
        return [
            ("BMI", "18.5-24.9 kg/m²", bmi, 18.5, 24.9),
            ("Triglycerides", "35-135 mg/dL", triglycerides, 35, 135),
            ("ALT", "7-35 U/L", alt, 7, 35),
            ("AST", "7-35 U/L", ast, 7, 35),
            ("FBG", "70-99 mg/dL", fbg, 70, 99),
            ("Waist Circumference", "<88 cm", waist_circumference, None, 88),
            ("SBP", "90-120 mmHg", sbp, 90, 120),
            ("HDL-C", ">50 mg/dL", hdl, 50, None),
            ("LDL-C", "<130 mg/dL", ldl, None, 130),
            ("MCV", "80-100 fL", mcv, 80, 100),
            ("CDT", "<2.5%", cdt, None, 2.5),
            ("GGT", "8-61 U/L", ggt, 8, 61),
            ("RBC Size", "6-8 µm", rbc_size, 6, 8),
            ("Diabetes Type 2", "0 = No, 1 = Yes", diabetes, 0, 1),
            ("Pure Alcohol Consumption", "0-15 g/weeks", alcohol_consumption, None, 15),
            ("Year Alcohol Consumption", "0-2 years", year_alcohol_consumption, None, 2)
        ]


# Collect user inputs
#print("Enter patient details:")

#name = input("Enter your name: ")
'''
gender = get_valid_int("Gender (0 = Male, 1 = Female): ", 0, 1)
age = get_valid_int("Age in years: ", 1, 120)
bmi = get_valid_float("BMI: ", 10, 50)
triglycerides = get_valid_float("Triglycerides (TG) in mg/dL: ", 50, 500)
alt = get_valid_float("Alanine Aminotransferase (ALT) in U/L: ", 5, 300)
ast = get_valid_float("Aspartate Aminotransferase (AST) in U/L: ", 5, 300)
fbg = get_valid_float("Fasting Blood Glucose (FBG) in mg/dL: ", 50, 300)
waist_circumference = get_valid_float("Waist Circumference in cm: ", 50, 150)
sbp = get_valid_float("Systolic Blood Pressure (SBP) in mmHg: ", 80, 200)
hdl = get_valid_float("High-Density Lipoprotein (HDL-C) in mg/dL: ", 10, 100)
ldl = get_valid_float("Low-Density Lipoprotein (LDL-C) in mg/dL: ", 50, 200)
mcv = get_valid_float("Mean Corpuscular Volume (MCV) in fL: ", 50, 120)
cdt = get_valid_float("Carbohydrate-Deficient Transferrin (CDT) %: ", 0, 10)
ggt = get_valid_float("Gamma-Glutamyl Transferase (GGT) in U/L: ", 5, 500)
rbc_size = get_valid_float("RBC Size (µm): ", 5, 10)
alcohol_consumption = get_valid_float("Alcohol Consumption in g/weeks: ", 0, 600)
year_alcohol_consumption = get_valid_int("Year of  Alcohol Consumption in years: ", 0, 60)
diabetes = get_valid_int("Diabetes Status (0 = Non-Diabetic, 1 = Type 2 Diabetes): ", 0, 1)
'''
# Create feature vector

import tkinter as tk
from tkinter import messagebox

def collect_inputs():
    try:
        global name, gender, age, bmi, triglycerides, alt, ast, fbg, waist_circumference, sbp
        global hdl, ldl, mcv, cdt, ggt, rbc_size, alcohol_consumption, year_alcohol_consumption, diabetes

        name = name_var.get()
        gender = int(gender_var.get())
        age = int(age_var.get())
        bmi = float(bmi_var.get())
        triglycerides = float(triglycerides_var.get())
        alt = float(alt_var.get())
        ast = float(ast_var.get())
        fbg = float(fbg_var.get())
        waist_circumference = float(waist_var.get())
        sbp = float(sbp_var.get())
        hdl = float(hdl_var.get())
        ldl = float(ldl_var.get())
        mcv = float(mcv_var.get())
        cdt = float(cdt_var.get())
        ggt = float(ggt_var.get())
        rbc_size = float(rbc_var.get())
        alcohol_consumption = float(alcohol_var.get())
        year_alcohol_consumption = int(year_alcohol_var.get())
        diabetes = int(diabetes_var.get())

        root.destroy()  # Close the form window after successful validation

    except ValueError:
        messagebox.showerror("Invalid input", "Please enter valid numerical values.")

root = tk.Tk()
root.title("Patient Details Input Form")

# Variables
fields = [
    ("Name", "name_var"), ("Gender (0=Male, 1=Female)", "gender_var"), ("Age", "age_var"), 
    ("BMI", "bmi_var"), ("Triglycerides (mg/dL)", "triglycerides_var"), ("ALT (U/L)", "alt_var"),
    ("AST (U/L)", "ast_var"), ("FBG (mg/dL)", "fbg_var"), ("Waist Circumference (cm)", "waist_var"),
    ("SBP (mmHg)", "sbp_var"), ("HDL (mg/dL)", "hdl_var"), ("LDL (mg/dL)", "ldl_var"),
    ("MCV (fL)", "mcv_var"), ("CDT (%)", "cdt_var"), ("GGT (U/L)", "ggt_var"),
    ("RBC Size (µm)", "rbc_var"), ("Alcohol Consumption (g/week)", "alcohol_var"),
    ("Year Alcohol Consumption", "year_alcohol_var"), ("Diabetes (0=No, 1=Yes)", "diabetes_var")
]

entries = {}
for idx, (label, varname) in enumerate(fields):
    tk.Label(root, text=label).grid(row=idx, column=0, sticky='w')
    entries[varname] = tk.StringVar()
    tk.Entry(root, textvariable=entries[varname]).grid(row=idx, column=1)

# Assign variables globally
for varname in entries:
    globals()[varname] = entries[varname]

tk.Button(root, text="Submit", command=collect_inputs).grid(row=len(fields), columnspan=2, pady=10)

root.mainloop()

input_data = [[
    gender, age, bmi, triglycerides, alt, ast, fbg, waist_circumference, sbp,
    hdl, ldl, mcv, cdt, ggt, rbc_size, alcohol_consumption, year_alcohol_consumption, diabetes
]]


# Scale the input data
scaled_input = scaler.transform(input_data)

# Make prediction
predicted_probs = model.predict_proba(scaled_input)
predicted_class = model.predict(scaled_input)[0]
confidence_score = np.max(predicted_probs) * 100

# Class mapping
class_labels = {0: "Healthy", 1: "NAFLD", 2: "AFLD", 3: "Non-Healthy (Not Fatty Liver)"}

# Risk classification
risk_livel=classify_risk(predicted_class , alt, ast, bmi, alcohol_consumption,
                  waist_circumference, triglycerides, hdl, fbg, sbp, ldl,
                  diabetes, ggt, cdt, mcv, gender)
# Generate PDF Report
pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()
pdf.add_font("DejaVu", "", "DejaVuSans.ttf", uni=True)
pdf.set_font("DejaVu", "", 16)
pdf.cell(200, 10, "Fatty Liver Disease Risk Report", ln=True, align='C')
pdf.ln(10)

# Patient Information
pdf.set_font("Arial", "B", 12)
pdf.cell(200, 10, "Patient Information:", ln=True)
pdf.set_font("DejaVu", size=12)
pdf.cell(200, 10, f"Name: {name}", ln=True)
pdf.cell(200, 10, f"Age: {age}", ln=True)
pdf.cell(200, 10, f"Gender: {'Male' if gender == 0 else 'Female'}", ln=True)
pdf.cell(200, 10, f"Predicted Category: {class_labels[predicted_class]}", ln=True)
pdf.cell(200, 10, f"Confidence Score: {confidence_score:.2f}%", ln=True)
pdf.ln(5)

# Parameters Table
pdf.set_font("Arial", "B", 12)
pdf.cell(200, 10, "Patient Parameter Analysis:", ln=True)
pdf.set_font("DejaVu", size=12)
#pdf.ln(2)


# Table for Parameter Analysis
pdf.set_font("Arial", "B", 12)
pdf.cell(60, 10, "Parameter", border=1)
pdf.cell(70, 10, "Normal Range", border=1)
pdf.cell(60, 10, "Patient Value", border=1, ln=True)
pdf.set_font("DejaVu", size=12)

def check_status(value, normal_min, normal_max):
    if normal_min is not None and normal_max is not None:
        return "✔" if normal_min <= value <= normal_max else "✘"
    elif normal_max is not None:
        return "✔" if value <= normal_max else "✘"
    return "✔"  # Default to normal if no strict range exists


parameters = parameter(gender, bmi, triglycerides, alt, ast, fbg, waist_circumference, sbp, hdl, ldl, mcv, cdt, ggt, rbc_size, diabetes, alcohol_consumption, year_alcohol_consumption)

for param, normal, value, min_val, max_val in parameters:
    pdf.cell(60, 10, param, border=1)
    pdf.cell(70, 10, normal, border=1)
    pdf.cell(40, 10, str(value), border=1)
    pdf.cell(20, 10, check_status(value, min_val, max_val), border=1, ln=True)

pdf.ln(5)


# SHAP Feature Importance
explainer = shap.TreeExplainer(graph)
shap_values = explainer.shap_values(scaled_input)
shap_values_class = shap_values[0, :, predicted_class]
shap_values_mean = np.abs(shap_values_class)

feature_names = [
    "Gender", "Age", "BMI", "Triglycerides", "ALT", "AST", "FBG", "Waist Circumference", "SBP",
    "HDL", "LDL", "MCV", "CDT", "GGT", "RBC Size", "Diabetes", "Pure Alcohol Consumption", "Year alcohol consumption"
]

plt.figure(figsize=(8, 6))
plt.barh(feature_names, shap_values_mean, color='skyblue')
plt.xlabel("Mean |SHAP Value|")
plt.ylabel("Features")
plt.title("Feature Contribution to Prediction")
plt.gca().invert_yaxis()
plt.savefig(f"C:/Users/ELCOT/Desktop/Final Year Project/Picture/shap_analysis_of_{name}.png", bbox_inches='tight')
plt.close()

pdf.set_font("Arial", "B", 12)
pdf.cell(200, 10, "Importance of Parameters in Prediction:", ln=True)
pdf.image(f"C:/Users/ELCOT/Desktop/Final Year Project/Picture/shap_analysis_of_{name}.png", x=10, w=180)

# Risk Classification
pdf.set_font("Arial", "B", 12)
pdf.cell(200, 10, "Risk Classification:", ln=True)
pdf.set_font("DejaVu", size=12)
pdf.cell(200, 10, f"Risk Level: {risk_livel}", ln=True)
pdf.ln(5)

# Recommendations
recommendation = get_recommendations(risk_livel)
formatted_recommendation = "\n".join(recommendation)
pdf.set_font("Arial", "B", 12)
pdf.cell(200, 10, "Health Recommendations:", ln=True)
pdf.set_font("DejaVu", size=12)
pdf.multi_cell(200, 10, formatted_recommendation)

save_directory = r"C:\Users\ELCOT\Desktop\Final Year Project\report"
# Save PDF
if gender==1:
  
  pdf.output(os.path.join(save_directory, f"{name}_Report.pdf"))
  print(f"PDF Report Generated for Ms.{name} as {name}_Report.pdf")
else:
  pdf.output(os.path.join(save_directory, f"{name}_Report.pdf"))
  print(f"PDF Report Generated for Mr.{name} as {name}_Report.pdf")

print(f"You are Predicted as {class_labels[predicted_class]} Category")


# === Define SQLite connection ===
db_path = "C:/Users/ELCOT/Desktop/Final Year Project/Database/reports.db"  # Make sure this path is correct for your environment
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# === Generate next patient ID ===
cursor.execute("SELECT COUNT(*) FROM reports")
record_count = cursor.fetchone()[0]
patient_id = f"P{record_count + 1:03d}"  # e.g., P001, P002, ...

# === Read PDF file content ===
pdf_file_path = os.path.join(save_directory, f"{name}_Report.pdf")
with open(pdf_file_path, "rb") as file:
    pdf_data = file.read()

# === Insert patient data and report into database ===
cursor.execute("""
INSERT INTO reports (patient_id, name, age, gender, disease, risk_level, report)
VALUES (?, ?, ?, ?, ?, ?, ?)
""", (
    patient_id,
    name,
    age,
    'Female' if gender == 1 else 'Male',
    class_labels[predicted_class],  # Disease
    risk_livel,                     # Risk Level
    pdf_data                        # PDF Report
))

conn.commit()
conn.close()

print(f"Patient data and report for {name} (ID: {patient_id}) stored in the database.")


