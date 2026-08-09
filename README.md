 #SmartPromote AI

#Employee Promotion Prediction & Analytics System

SmartPromote AI is a machine learning-based employee promotion prediction system developed using Python, Scikit-learn, and Streamlit.

The system predicts whether an employee is likely to be promoted based on:

- Experience
- Performance Score
- Attendance
- Projects Completed

It uses a **Decision Tree Classification** algorithm and provides an interactive dashboard for HR analytics.

# Features

 Employee promotion prediction
 Decision Tree Classification
 Interactive Streamlit dashboard
 CSV dataset upload
 Automatic data validation
 Automatic model training
 Accuracy, Precision, Recall and F1 Score
 Confusion Matrix
 Decision Tree visualization
 Feature importance analysis
 Interactive charts
 Individual employee prediction
 Prediction probability
 Dataset preview and download

## 🛠️ Technologies Used

| Technology    | Purpose                     |
| ------------- | --------------------------- |
| Python        | Programming                 |
| Pandas        | Data processing             |
| NumPy         | Numerical operations        |
| Scikit-learn  | Machine Learning            |
| Decision Tree | Classification              |
| Streamlit     | Interactive Dashboard       |
| Plotly        | Interactive Visualization   |
| Matplotlib    | Decision Tree Visualization |

##  Dataset

The project uses a self-created synthetic dataset containing employee records.

Required columns:

```text
Experience_Years
Performance_Score
Attendance
Projects_Completed
Promotion
```

Where:

```text
0 = Not Promoted
1 = Promoted
```

##  Machine Learning Workflow

```text
Employee Dataset
       ↓
Data Validation
       ↓
Data Cleaning
       ↓
Feature Selection
       ↓
Train/Test Split
       ↓
Decision Tree Training
       ↓
Model Evaluation
       ↓
Promotion Prediction
```

##  Run Locally

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Go to the project directory:

```bash
cd Employee-Promotion-Prediction
```

Create and activate a virtual environment:

```bash
python -m venv venv
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

##  Live Demo

Live application:

**To be added after Streamlit deployment**

##  Project Purpose

This project is developed as an academic machine learning project to demonstrate the complete ML workflow from dataset creation and preprocessing to model training, evaluation, visualization and deployment.

The system is intended as a **decision-support tool** and not as an autonomous replacement for HR decision-making.

## Author

Rahul Verma

B.Tech – Computer Science Engineering (AI/ML)
