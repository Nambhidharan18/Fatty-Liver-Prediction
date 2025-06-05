import numpy as np
import pandas as pd
import random

# Define parameter ranges with gender variations where applicable
PARAMETER_RANGES = {
    "BMI": {"Healthy": (18.5, 24.9), "NAFLD": (25, 35), "AFLD": (22, 32), "Non-Healthy": (22, 29)},
    "Triglycerides": {"Male": {"Healthy": (50, 140), "NAFLD": (141, 250), "AFLD": (151, 300), "Non-Healthy": (130, 170)},
                      "Female": {"Healthy": (50, 130), "NAFLD": (131, 230), "AFLD": (140, 290), "Non-Healthy": (120, 160)}},
    "ALT": {"Male": {"Healthy": (10, 40), "NAFLD": (41, 100), "AFLD": (41, 120), "Non-Healthy": (35, 55)},
            "Female": {"Healthy": (10, 30), "NAFLD": (31, 80), "AFLD": (31, 110), "Non-Healthy": (30, 50)}},
    "AST": {"Male": {"Healthy": (10, 40), "NAFLD": (41, 90), "AFLD": (41, 110), "Non-Healthy": (35, 55)},
            "Female": {"Healthy": (10, 35), "NAFLD": (36, 80), "AFLD": (36, 100), "Non-Healthy": (30, 50)}},
    "Fasting Blood Glucose": {"Healthy": (70, 99), "NAFLD": (100, 126), "AFLD": (100, 130), "Non-Healthy": (95, 110)},
    "Waist Circumference": {"Male": {"Healthy": (70, 94), "NAFLD": (95, 120), "AFLD": (95, 125), "Non-Healthy": (90, 100)},
                             "Female": {"Healthy": (60, 80), "NAFLD": (81, 110), "AFLD": (81, 115), "Non-Healthy": (75, 95)}},
    "SBP": {"Healthy": (90, 119), "NAFLD": (120, 140), "AFLD": (120, 145), "Non-Healthy": (115, 130)},
    "HDL-C": {"Male": {"Healthy": (40, 55), "NAFLD": (30, 39), "AFLD": (25, 38), "Non-Healthy": (35, 45)},
              "Female": {"Healthy": (50, 65), "NAFLD": (40, 49), "AFLD": (35, 48), "Non-Healthy": (40, 55)}},
    "LDL-C": {"Healthy": (50, 100), "NAFLD": (101, 160), "AFLD": (101, 180), "Non-Healthy": (95, 120)},
    "MCV": {"Healthy": (80, 95), "NAFLD": (96, 105), "AFLD": (100, 110), "Non-Healthy": (85, 100)},
    "CDT": {"Healthy": (0, 1.6), "NAFLD": (1.7, 3.0), "AFLD": (2.5, 5.0), "Non-Healthy": (1.5, 2.5)},
    "GGT": {"Healthy": (10, 50), "NAFLD": (51, 150), "AFLD": (100, 300), "Non-Healthy": (50, 120)},
    "RBC Size": {"Healthy": (6.0, 8.0), "NAFLD": (6.5, 9.0), "AFLD": (7.0, 10.0), "Non-Healthy": (6.2, 8.5)},
    "Pure Alcohol Consumption": {"Male": {"Healthy": (0, 20), "NAFLD": (10, 70), "AFLD": (210, 600), "Non-Healthy": (20, 140)},
                            "Female": {"Healthy": (0, 15), "NAFLD": (5, 60), "AFLD": (140, 450), "Non-Healthy": (10, 120)}},
    "Year of Alcohol Consumption": {"Male": {"Healthy": (0, 3), "NAFLD": (1, 5), "AFLD": (5, 40), "Non-Healthy": (1, 10)},
                            "Female": {"Healthy": (0, 2), "NAFLD": (1, 4), "AFLD": (3, 30), "Non-Healthy": (1, 8)}},
    "Diabetes Probability": {"Healthy": 0, "NAFLD": 0.3, "AFLD": 0.25, "Non-Healthy": 0.2},
    "Fatty Liver": {"Healthy": 0, "NAFLD": 1, "AFLD": 2, "Non-Healthy": 3},
}

# Function to generate synthetic patient data
def generate_samples(n_samples, category):
    data = []

    while len(data) < n_samples:
        gender = random.choice(["Male", "Female"])

        sample = {
            "Gender": gender,
            "Age": random.randint(18, 80),
            "BMI": round(np.random.uniform(*PARAMETER_RANGES["BMI"][category]), 2),
            "Triglycerides": round(np.random.uniform(*PARAMETER_RANGES["Triglycerides"][gender][category]), 1),
            "ALT": round(np.random.uniform(*PARAMETER_RANGES["ALT"][gender][category]), 1),
            "AST": round(np.random.uniform(*PARAMETER_RANGES["AST"][gender][category]), 1),
            "Fasting Blood Glucose": round(np.random.uniform(*PARAMETER_RANGES["Fasting Blood Glucose"][category]), 1),
            "Waist Circumference": round(np.random.uniform(*PARAMETER_RANGES["Waist Circumference"][gender][category]), 1),
            "SBP": round(np.random.uniform(*PARAMETER_RANGES["SBP"][category]), 1),
            "HDL-C": round(np.random.uniform(*PARAMETER_RANGES["HDL-C"][gender][category]), 1),
            "LDL-C": round(np.random.uniform(*PARAMETER_RANGES["LDL-C"][category]), 1),
            "MCV": round(np.random.uniform(*PARAMETER_RANGES["MCV"][category]), 1),
            "CDT": round(np.random.uniform(*PARAMETER_RANGES["CDT"][category]), 2),
            "GGT": round(np.random.uniform(*PARAMETER_RANGES["GGT"][category]), 1),
            "RBC Size": round(np.random.uniform(*PARAMETER_RANGES["RBC Size"][category]), 2),
            "Pure Alcohol Consumption": round(np.random.uniform(*PARAMETER_RANGES["Pure Alcohol Consumption"][gender][category]), 1),
            "Year of Alcohol Consumption": random.randint( int(PARAMETER_RANGES["Year of Alcohol Consumption"][gender][category][0]),
                                                          int(PARAMETER_RANGES["Year of Alcohol Consumption"][gender][category][1])),
           "Diabetes": 1 if random.random() < PARAMETER_RANGES["Diabetes Probability"][category] else 0,
            "Fatty Liver": PARAMETER_RANGES["Fatty Liver"][category]
        }

        if sample not in data:  # Avoid duplicates
            data.append(sample)

    return pd.DataFrame(data)

# Function to generate dataset for all categories
def generate_dataset():
    sample_sizes = {}

    # User-defined sample sizes
    print("Enter sample size for each category:")
    for category in ["Healthy", "NAFLD", "AFLD", "Non-Healthy"]:
        sample_sizes[category] = int(input(f"{category}: "))

    df_list = []

    for category, n_samples in sample_sizes.items():
        df_list.append(generate_samples(n_samples, category))

    final_dataset = pd.concat(df_list, ignore_index=True)

    # Convert gender to 0 (Male) and 1 (Female) before saving
    final_dataset["Gender"] = final_dataset["Gender"].map({"Male": 0, "Female": 1})

    # Save dataset
    final_dataset.to_csv("synthetic_patient_data.csv", index=False)

    print("Dataset generated and saved as 'synthetic_patient_data.csv'.")

# Run the function
generate_dataset()
