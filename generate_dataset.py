import pandas as pd
import numpy as np
import os

np.random.seed(42)

n = 300

experience = np.random.randint(1, 16, n)

performance = np.random.randint(50, 101, n)

attendance = np.random.randint(70, 101, n)

projects = np.random.randint(0, 11, n)

# Promotion logic
promotion = (
    (experience >= 5) &
    (performance >= 70) &
    (attendance >= 80) &
    (projects >= 4)
).astype(int)

df = pd.DataFrame({
    "Experience_Years": experience,
    "Performance_Score": performance,
    "Attendance": attendance,
    "Projects_Completed": projects,
    "Promotion": promotion
})

# Get the folder where this Python file is located
folder = os.path.dirname(os.path.abspath(__file__))

# Create CSV in the same folder
file_path = os.path.join(
    folder,
    "employee_promotion_dataset.csv"
)

df.to_csv(file_path, index=False)

print("======================================")
print("DATASET CREATED SUCCESSFULLY!")
print("======================================")
print("Location:")
print(file_path)
print()
print("Number of records:", len(df))
print()
print(df.head())
print()
print("Promotion distribution:")
print(df["Promotion"].value_counts())