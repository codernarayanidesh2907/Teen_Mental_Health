# GUI for Teen Mental Health Analysis
import tkinter as tk
from tkinter import ttk
import joblib
from tkinter import messagebox

# Load the trained model
model = joblib.load("teen_mental_health_model.pkl")
gender_encoder = joblib.load("gender_encoder.pkl")
platform_encoder = joblib.load("platform_encoder.pkl")
social_encoder = joblib.load("social_encoder.pkl")

root = tk.Tk()
root.title("Teen Mental Health Prediction")
root.geometry("800x700")
root.configure(bg="#3ABEF9")  # light blue background color
root.resizable(False, False)  # Prevent resizing

title_label = tk.Label(root, text="Teen Mental Health Prediction", font=("Helvetica", 20, "bold"), bg="#3ABEF9")
title_label.pack(pady=20)

input_frame = tk.Frame(root, bg="#3ABEF9")
input_frame.pack(pady = 10)

fields = [
    "Age",
    "Daily Social Media Hours",
    "Sleep Hours",
    "Screen Time Before Sleep",
    "Academic Performance",
    "Physical Activity",
    "Stress Level",
    "Anxiety Level",
    "Addiction Level"
]
entries = {}

for i, field in enumerate(fields):
    label = tk.Label(
        input_frame,
        text=field,
        font=("Arial", 12),
        bg="#3ABEF9",
    )
    label.grid(row=i, column=0, padx=20, pady=10, sticky="w")

    entry = ttk.Entry(
        input_frame,
        width=25
    )
    entry.grid(row=i, column=1, padx=20, pady=10)
    entries[field] = entry

# Add Dropdown Menus

# gender
gender_label = tk.Label(
    input_frame,
    text="Gender",
    font=("Arial",11),
    bg="#3ABEF9"
)

gender_label.grid(row=9, column=0, padx=15, pady=8, sticky="w")

gender_combo = ttk.Combobox(
    input_frame,
    values=["male", "female"],
    state="readonly",
    width=22
)

gender_combo.grid(row=9, column=1, padx=15, pady=8)

# Platform Usage 
platform_label = tk.Label(
    input_frame,
    text="Platform Usage",
    font=("Arial",11),
    bg="#3ABEF9"
)

platform_label.grid(row=10, column=0, padx=15, pady=8, sticky="w")

platform_combo = ttk.Combobox(
    input_frame,
    values=[
        "Facebook",
        "Instagram",
        "Snapchat",
        "TikTok",
        "YouTube"
    ],
    state="readonly",
    width=22
)

platform_combo.grid(row=10, column=1, padx=15, pady=8)

# Social Interaction level
social_label = tk.Label(
    input_frame,
    text="Social Interaction",
    font=("Arial",11),
    bg="#3ABEF9"
)

social_label.grid(row=11, column=0, padx=15, pady=8, sticky="w")

social_combo = ttk.Combobox(
    input_frame,
    values=["high", "medium", "low"],
    state="readonly",
    width=22
)

social_combo.grid(row=11, column=1, padx=15, pady=8)

#------------ PREDICTION FUNCTION ----------------#
def predict():
    try:
        # Get input values
        age = float(entries["Age"].get())
        gender = gender_combo.get()
        daily_social_media_hours = float(entries["Daily Social Media Hours"].get())
        platform_usage = platform_combo.get()
        sleep_hours = float(entries["Sleep Hours"].get())
        screen_time_before_sleep = float(entries["Screen Time Before Sleep"].get())
        academic_performance = float(entries["Academic Performance"].get())
        physical_activity = float(entries["Physical Activity"].get())
        social_interaction = social_combo.get()
        stress_level = float(entries["Stress Level"].get())
        anxiety_level = float(entries["Anxiety Level"].get())
        addiction_level = float(entries["Addiction Level"].get())

        
       
        

        # Encode categorical variables
        gender_encoded = gender_encoder.transform([gender])[0]
        platform_encoded = platform_encoder.transform([platform_usage])[0]
        social_encoded = social_encoder.transform([social_interaction])[0]

        # Create feature vector
        features = [
            age,
            gender_encoded,
            daily_social_media_hours,
             platform_encoded,
            sleep_hours,
            screen_time_before_sleep,
            academic_performance,
            physical_activity,
            social_encoded,
            stress_level,
            anxiety_level,
            addiction_level,
            
        ]

        # Make prediction
        prediction = model.predict([features])[0]
        probability = model.predict_proba([features])[0]

        # Display results
        messagebox.showinfo("Prediction", f"Predicted Outcome: {prediction}")
        messagebox.showinfo("Probabilities", f"Probability of No Depression: {probability[0]:.2f}\nProbability of Depression: {probability[1]:.2f}")

    except Exception as e:
       messagebox.showerror(
        "Error",
        str(e)
    )

#------------ PREDICT BUTTON ----------------#
predict_button = tk.Button(
    root,
    text="Predict",
    font=("Arial", 14, "bold"),
    bg="#4CAF50",
    fg="white",
    command=predict
)
predict_button.pack(pady=20)

#----------- Result Label ----------------#
result_label = tk.Label(
    root,
    text="Predictions will appear here......",
    font=("Arial", 14),
    bg="#3ABEF9"
)
result_label.pack(pady=10)

root.mainloop()
