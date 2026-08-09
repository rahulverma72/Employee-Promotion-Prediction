import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# -------------------------------
# Page settings
# -------------------------------

st.set_page_config(
    page_title="SmartPromote AI",
    page_icon="🎯",
    layout="wide"
)

# Keep prediction history during the current Streamlit session.
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []


# -------------------------------
# Title
# -------------------------------

st.title("🎯 SmartPromote AI")
st.subheader("Employee Promotion Prediction & Analytics System")

st.write(
    "This application uses a Decision Tree algorithm "
    "to predict employee promotion status."
)

st.divider()


# -------------------------------
# Required columns
# -------------------------------

required_columns = [
    "Experience_Years",
    "Performance_Score",
    "Attendance",
    "Projects_Completed",
    "Promotion"
]

features = [
    "Experience_Years",
    "Performance_Score",
    "Attendance",
    "Projects_Completed"
]


# -------------------------------
# Sidebar
# -------------------------------

st.sidebar.header("⚙️ Settings")

uploaded_file = st.sidebar.file_uploader(
    "Upload Employee CSV",
    type=["csv"]
)

test_size = st.sidebar.slider(
    "Test Data (%)",
    10,
    40,
    20
)

tree_depth = st.sidebar.slider(
    "Decision Tree Depth",
    2,
    10,
    5
)


# -------------------------------
# Load dataset
# -------------------------------

if uploaded_file is not None:

    try:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("CSV uploaded successfully")

    except Exception:
        st.error("Unable to read the uploaded CSV file.")
        st.stop()

else:

    try:
        df = pd.read_csv(
            "employee_promotion_dataset.csv"
        )

    except FileNotFoundError:
        st.error(
            "employee_promotion_dataset.csv "
            "was not found."
        )
        st.stop()


# -------------------------------
# Check columns
# -------------------------------

missing_columns = []

for column in required_columns:

    if column not in df.columns:
        missing_columns.append(column)


if missing_columns:

    st.error("Some required columns are missing:")

    for column in missing_columns:
        st.write("❌", column)

    st.stop()


# -------------------------------
# Clean data
# -------------------------------

# Employee_Name is optional and used only for display.
# It is never used as a machine-learning feature.
if "Employee_Name" in df.columns:
    employee_names = df["Employee_Name"].astype(str).fillna("Unknown")
else:
    employee_names = pd.Series(
        [f"Employee {i + 1}" for i in range(len(df))],
        index=df.index
    )

df = df[required_columns].copy()

for column in required_columns:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

df = df.dropna()

df["Promotion"] = df["Promotion"].astype(int)

# Keep names for display only.
employee_names = employee_names.loc[df.index].reset_index(drop=True)
df = df.reset_index(drop=True)
df.insert(0, "Employee_Name", employee_names)


# Check Promotion values

if not set(df["Promotion"].unique()).issubset({0, 1}):

    st.error(
        "Promotion column must contain only 0 and 1."
    )

    st.stop()


if len(df["Promotion"].unique()) < 2:

    st.error(
        "Dataset must contain both promoted "
        "and not promoted employees."
    )

    st.stop()


# -------------------------------
# Prepare data
# -------------------------------

X = df[features]
y = df["Promotion"]


# -------------------------------
# Train test split
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_size / 100,
    random_state=42,
    stratify=y
)


# -------------------------------
# Create and train model
# -------------------------------

model = DecisionTreeClassifier(
    max_depth=tree_depth,
    random_state=42
)

model.fit(
    X_train,
    y_train
)


# -------------------------------
# Model prediction
# -------------------------------

y_pred = model.predict(X_test)


# -------------------------------
# Model evaluation
# -------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)


# -------------------------------
# Dashboard statistics
# -------------------------------

total_employees = len(df)

promoted = len(
    df[df["Promotion"] == 1]
)

not_promoted = len(
    df[df["Promotion"] == 0]
)

promotion_rate = (
    promoted / total_employees
) * 100


# -------------------------------
# KPI Cards
# -------------------------------

st.header("📊 Employee Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Employees",
    total_employees
)

col2.metric(
    "Promoted",
    promoted
)

col3.metric(
    "Not Promoted",
    not_promoted
)

col4.metric(
    "Promotion Rate",
    f"{promotion_rate:.1f}%"
)


st.divider()


# -------------------------------
# Tabs
# -------------------------------

overview, model_tab, tree_tab, prediction_tab, data_tab = st.tabs(
    [
        "📊 Overview",
        "🤖 Model Performance",
        "🌳 Decision Tree",
        "🔮 Prediction",
        "📂 Dataset"
    ]
)


# ==================================================
# OVERVIEW
# ==================================================

with overview:

    st.header("📊 Employee Analytics")

    # Promotion chart

    col1, col2 = st.columns(2)

    with col1:

        promotion_data = pd.DataFrame({
            "Status": [
                "Not Promoted",
                "Promoted"
            ],
            "Employees": [
                not_promoted,
                promoted
            ]
        })

        fig = px.pie(
            promotion_data,
            names="Status",
            values="Employees",
            hole=0.4,
            title="Promotion Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # Performance chart

    with col2:

        fig = px.histogram(
            df,
            x="Performance_Score",
            color="Promotion",
            title="Performance Score Distribution",
            nbins=15
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # Experience vs performance

    col1, col2 = st.columns(2)

    with col1:

        fig = px.scatter(
            df,
            x="Experience_Years",
            y="Performance_Score",
            color="Promotion",
            size="Projects_Completed",
            title="Experience vs Performance",
            hover_data=["Attendance"]
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # Attendance chart

    with col2:

        fig = px.box(
            df,
            x="Promotion",
            y="Attendance",
            color="Promotion",
            title="Attendance and Promotion"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # Project analysis

    st.subheader("📌 Projects and Promotion")

    project_data = (
        df.groupby("Projects_Completed")["Promotion"]
        .mean()
        .reset_index()
    )

    project_data["Promotion_Rate"] = (
        project_data["Promotion"] * 100
    )

    fig = px.bar(
        project_data,
        x="Projects_Completed",
        y="Promotion_Rate",
        title="Promotion Rate by Projects Completed",
        labels={
            "Projects_Completed": "Projects",
            "Promotion_Rate": "Promotion Rate (%)"
        }
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ==================================================
# MODEL PERFORMANCE
# ==================================================

with model_tab:

    st.header("🤖 Model Performance")

    st.write(
        "The Decision Tree is evaluated using "
        "the testing dataset."
    )


    # Metrics

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Accuracy",
        f"{accuracy * 100:.2f}%"
    )

    col2.metric(
        "Precision",
        f"{precision * 100:.2f}%"
    )

    col3.metric(
        "Recall",
        f"{recall * 100:.2f}%"
    )

    col4.metric(
        "F1 Score",
        f"{f1 * 100:.2f}%"
    )


    st.divider()


    # Confusion matrix

    st.subheader("📌 Confusion Matrix")

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    fig = px.imshow(
        cm,
        text_auto=True,
        x=[
            "Not Promoted",
            "Promoted"
        ],
        y=[
            "Not Promoted",
            "Promoted"
        ],
        title="Confusion Matrix"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


    st.info(
        f"Training records: {len(X_train)} | "
        f"Testing records: {len(X_test)}"
    )


# ==================================================
# DECISION TREE
# ==================================================

with tree_tab:

    st.header("🌳 Decision Tree")

    st.write(
        "The following tree shows how the model "
        "makes promotion decisions."
    )


    fig, ax = plt.subplots(
        figsize=(20, 10)
    )

    plot_tree(
        model,
        feature_names=features,
        class_names=[
            "Not Promoted",
            "Promoted"
        ],
        filled=True,
        rounded=True,
        ax=ax
    )

    st.pyplot(fig)


    st.subheader("⭐ Feature Importance")

    importance = pd.DataFrame({
        "Feature": features,
        "Importance": model.feature_importances_
    })

    importance = importance.sort_values(
        "Importance",
        ascending=False
    )

    fig = px.bar(
        importance,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Feature Importance"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ==================================================
# PREDICTION
# ==================================================

with prediction_tab:

    st.header("🔮 Employee Promotion Prediction")

    st.write(
        "Enter the details of a new employee."
    )


    col1, col2 = st.columns(2)


    with col1:

        employee_name = st.text_input(
            "Employee Name",
            placeholder="Enter employee name",
            max_chars=50
        )

        experience = st.number_input(
            "Experience (Years)",
            min_value=0,
            max_value=20,
            value=5
        )

        performance = st.slider(
            "Performance Score",
            0,
            100,
            75
        )


    with col2:

        attendance = st.slider(
            "Attendance (%)",
            0,
            100,
            85
        )

        projects = st.number_input(
            "Projects Completed",
            min_value=0,
            max_value=15,
            value=4
        )


    if st.button(
        "🚀 Predict Promotion",
        use_container_width=True
    ):

        if not employee_name.strip():
            st.warning("⚠️ Please enter the employee name first.")
        else:
            new_employee = pd.DataFrame({
                "Experience_Years": [experience],
                "Performance_Score": [performance],
                "Attendance": [attendance],
                "Projects_Completed": [projects]
            })

            prediction = model.predict(new_employee)[0]
            probability = model.predict_proba(new_employee)[0]

            class_probabilities = dict(zip(model.classes_, probability))
            promoted_probability = class_probabilities.get(1, 0) * 100
            not_promoted_probability = class_probabilities.get(0, 0) * 100

            status = (
                "Eligible for Promotion"
                if prediction == 1
                else "Not Eligible"
            )

            record = {
                "Employee Name": employee_name.strip(),
                "Experience": experience,
                "Performance": performance,
                "Attendance": attendance,
                "Projects": projects,
                "Promotion %": round(promoted_probability, 2),
                "Status": status
            }

            # Update the employee if the same name is tested again.
            st.session_state.prediction_history = [
                row for row in st.session_state.prediction_history
                if row["Employee Name"].lower() != employee_name.strip().lower()
            ]
            st.session_state.prediction_history.append(record)

            st.divider()

            if prediction == 1:
                st.success(
                    f"🎉 {employee_name.strip()} is likely to be promoted."
                )
            else:
                st.warning(
                    f"⚠️ {employee_name.strip()} is not likely to be promoted."
                )

            col1, col2 = st.columns(2)

            col1.metric(
                "Promotion Probability",
                f"{promoted_probability:.2f}%"
            )

            col2.metric(
                "Not Promoted Probability",
                f"{not_promoted_probability:.2f}%"
            )

            probability_data = pd.DataFrame({
                "Status": ["Not Promoted", "Promoted"],
                "Probability": [
                    not_promoted_probability,
                    promoted_probability
                ]
            })

            fig = px.bar(
                probability_data,
                x="Status",
                y="Probability",
                text_auto=".1f",
                title=f"Prediction Probability — {employee_name.strip()}"
            )

            st.plotly_chart(fig, use_container_width=True)

            st.subheader("👤 Employee Result")

            result_col1, result_col2, result_col3 = st.columns(3)

            result_col1.metric("Employee", employee_name.strip())
            result_col2.metric(
                "Status",
                "✅ Eligible" if prediction == 1 else "❌ Not Eligible"
            )
            result_col3.metric(
                "Promotion Percentage",
                f"{promoted_probability:.2f}%"
            )

            st.subheader("💡 Employee Profile")

            if performance >= 85:
                st.write("✅ Strong performance score")
            elif performance < 70:
                st.write("⚠️ Performance score can be improved")
            else:
                st.write("ℹ️ Average performance score")

            if attendance >= 90:
                st.write("✅ Excellent attendance")
            elif attendance < 80:
                st.write("⚠️ Attendance can be improved")

            if projects >= 7:
                st.write("✅ Strong project contribution")
            elif projects < 3:
                st.write("⚠️ More project involvement may help")

    # ==================================================
    # ELIGIBLE EMPLOYEES
    # ==================================================
    st.divider()
    st.subheader("🏆 Eligible Employees")

    history_df = pd.DataFrame(st.session_state.prediction_history)

    if history_df.empty:
        st.info(
            "No predictions yet. Enter an employee name and details above."
        )
    else:
        eligible_df = history_df[
            history_df["Status"] == "Eligible for Promotion"
        ].copy()

        if eligible_df.empty:
            st.warning(
                "No employees are currently predicted as eligible for promotion."
            )
        else:
            eligible_df = eligible_df.sort_values(
                "Promotion %",
                ascending=False
            )

            display_eligible = eligible_df[
                [
                    "Employee Name",
                    "Experience",
                    "Performance",
                    "Attendance",
                    "Projects",
                    "Promotion %",
                    "Status"
                ]
            ].copy()

            display_eligible["Status"] = "✅ Eligible"

            st.dataframe(
                display_eligible,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Promotion %": st.column_config.ProgressColumn(
                        "Promotion %",
                        min_value=0,
                        max_value=100,
                        format="%.2f%%"
                    )
                }
            )
            st.caption(
                "Eligible means the Decision Tree predicted Promotion = 1. "
                "Promotion % is the model's predicted probability."
            )

        with st.expander("📋 View All Prediction History"):
            all_history = history_df.sort_values(
                "Promotion %",
                ascending=False
            ).copy()

            all_history["Status"] = all_history["Status"].replace({
                "Eligible for Promotion": "✅ Eligible",
                "Not Eligible": "❌ Not Eligible"
            })

            st.dataframe(
                all_history,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Promotion %": st.column_config.ProgressColumn(
                        "Promotion %",
                        min_value=0,
                        max_value=100,
                        format="%.2f%%"
                    )
                }
            )

        if st.button("🗑️ Clear Prediction History"):
            st.session_state.prediction_history = []
            st.rerun()


# ==================================================
# DATASET
# ==================================================

with data_tab:

    st.header("📂 Dataset Management")


    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Records",
        len(df)
    )

    col2.metric(
        "Input Features",
        len(features)
    )

    col3.metric(
        "Target",
        "Promotion"
    )


    st.divider()


    st.subheader("👀 Dataset Preview")

    number_of_rows = st.slider(
        "Number of rows",
        5,
        min(50, len(df)),
        min(10, len(df))
    )

    st.dataframe(
        df.head(number_of_rows),
        use_container_width=True
    )


    # ==================================================
    # PREDICT ALL EMPLOYEES IN DATASET
    # ==================================================
    st.divider()
    st.subheader("🏆 Eligible Employees — Model Prediction")

    # Predict promotion probability for every employee in the loaded CSV.
    dataset_features = df[features]
    dataset_predictions = model.predict(dataset_features)
    dataset_probabilities = model.predict_proba(dataset_features)

    class_probabilities = {
        cls: dataset_probabilities[:, i]
        for i, cls in enumerate(model.classes_)
    }

    full_prediction_df = df.copy()
    full_prediction_df["Promotion Probability"] = (
        class_probabilities.get(1, [0] * len(df)) * 100
    )
    full_prediction_df["Predicted Promotion"] = dataset_predictions
    full_prediction_df["Prediction Status"] = full_prediction_df[
        "Predicted Promotion"
    ].map({
        1: "Eligible",
        0: "Not Eligible"
    })

    eligible_employees = full_prediction_df[
        full_prediction_df["Predicted Promotion"] == 1
    ].copy()

    eligible_employees = eligible_employees.sort_values(
        "Promotion Probability",
        ascending=False
    )

    if eligible_employees.empty:
        st.warning("No employees are predicted as eligible for promotion.")
    else:
        st.success(
            f"{len(eligible_employees)} employee(s) are predicted as eligible "
            "for promotion."
        )

        eligible_display = eligible_employees[
            [
                "Employee_Name",
                "Experience_Years",
                "Performance_Score",
                "Attendance",
                "Projects_Completed",
                "Promotion Probability",
                "Prediction Status"
            ]
        ].copy()

        eligible_display["Promotion Probability"] = (
            eligible_display["Promotion Probability"].round(2)
        )

        st.dataframe(
            eligible_display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Employee_Name": st.column_config.TextColumn(
                    "Employee Name"
                ),
                "Promotion Probability": st.column_config.ProgressColumn(
                    "Promotion %",
                    min_value=0,
                    max_value=100,
                    format="%.2f%%"
                ),
                "Prediction Status": st.column_config.TextColumn(
                    "Status"
                )
            }
        )

        # Download only the eligible employees.
        eligible_csv = eligible_display.to_csv(index=False).encode("utf-8")

        st.download_button(
            "📥 Download Eligible Employees",
            data=eligible_csv,
            file_name="eligible_employees_prediction.csv",
            mime="text/csv"
        )

        st.caption(
            "Promotion % is the Decision Tree's predicted probability for "
            "Promotion = 1. This is a model prediction, not a final HR decision."
        )

    st.subheader("📈 Dataset Statistics")

    st.dataframe(
        df.describe().round(2),
        use_container_width=True
    )


    st.subheader("🧹 Data Quality")

    quality = pd.DataFrame({
        "Column": df.columns,
        "Missing Values": [
            df[col].isnull().sum()
            for col in df.columns
        ],
        "Unique Values": [
            df[col].nunique()
            for col in df.columns
        ]
    })

    st.dataframe(
        quality,
        use_container_width=True
    )


    # Download button

    csv = df.to_csv(
        index=False
    ).encode("utf-8")


    st.download_button(
        "📥 Download Dataset",
        data=csv,
        file_name="employee_promotion_dataset.csv",
        mime="text/csv"
    )


# -------------------------------
# Footer
# -------------------------------

st.divider()

st.caption(
    "SmartPromote AI | "
    "Employee Promotion Prediction System | "
    "Python + Scikit-learn + Streamlit"
)