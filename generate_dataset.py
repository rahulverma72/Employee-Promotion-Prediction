import pandas as pd
import numpy as np
import os

np.random.seed(42)

n = 300

# -----------------------------------
# Employee Names
# -----------------------------------

first_names = [
    "Rahul", "Anish", "Amit", "Priya", "Neha",
    "Rohit", "Anjali", "Arjun", "Sneha", "Karan",
    "Pooja", "Vikas", "Riya", "Aditya", "Simran",
    "Nikhil", "Kavya", "Saurabh", "Meera", "Varun",
    "Ishita", "Manish", "Shreya", "Akash", "Naina",
    "Vivek", "Sakshi", "Mohit", "Ayesha", "Harsh"
]

last_names = [
    "Verma", "Yadav", "Sharma", "Singh", "Gupta",
    "Kumar", "Mishra", "Pandey", "Tiwari", "Agarwal"
]

names = []

for i in range(n):
    first = first_names[i % len(first_names)]
    last = last_names[(i * 3) % len(last_names)]

    name = f"{first} {last}"

    if name in names:
        name = f"{first} {last} {i + 1}"

    names.append(name)


# -----------------------------------
# Employee Features
# -----------------------------------

experience = np.random.randint(1, 16, n)

performance = np.random.randint(50, 101, n)

attendance = np.random.randint(70, 101, n)

projects = np.random.randint(0, 11, n)


# -----------------------------------
# REALISTIC PROMOTION SCORE
# -----------------------------------

score = (
    (experience / 15) * 25 +
    ((performance - 50) / 50) * 30 +
    ((attendance - 70) / 30) * 20 +
    (projects / 10) * 25
)

# Add some randomness
noise = np.random.normal(0, 8, n)

final_score = score + noise


# -----------------------------------
# Promotion
# -----------------------------------

promotion_probability = 1 / (
    1 + np.exp(-(final_score - 55) / 8)
)

promotion = (
    np.random.random(n) < promotion_probability
).astype(int)


# -----------------------------------
# Create Dataset
# -----------------------------------

df = pd.DataFrame({
    "Employee_Name": names,
    "Experience_Years": experience,
    "Performance_Score": performance,
    "Attendance": attendance,
    "Projects_Completed": projects,
    "Promotion": promotion
})


# -----------------------------------
# Save CSV
# -----------------------------------

folder = os.path.dirname(
    os.path.abspath(__file__)
)

file_path = os.path.join(
    folder,
    "employee_promotion_dataset.csv"
)

df.to_csv(
    file_path,
    index=False
)


# -----------------------------------
# Display
# -----------------------------------

print("DATASET CREATED SUCCESSFULLY!")
print()

print("Location:")
print(file_path)

print()

print("Number of records:", len(df))

print()

print(df.head(10))

print()

print("Promotion distribution:")
print(df["Promotion"].value_counts())