# -*- coding: utf-8 -*-
"""
Cardiovascular Risk Prediction Dashboard
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ==========================================================
# PAGE CONFIGURATION — MUST BE FIRST STREAMLIT COMMAND
# ==========================================================

st.set_page_config(
    page_title="Cardiovascular Risk Dashboard",
    page_icon="❤️",
    layout="wide",
)


# ==========================================================
# PATHS
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "outputs" / "models"
TRAJECTORY_DIR = BASE_DIR / "outputs" / "trajectories"


# ==========================================================
# CONSTANTS
# ==========================================================

DISPLAY_NAMES = {
    "age": "Age",
    "gender": "Gender",
    "Ethnicity": "Ethnicity",
    "IMD": "Index of Multiple Deprivation",
    "bmi": "Body mass index",
    "sbp": "Systolic blood pressure",
    "dbp": "Diastolic blood pressure",
    "hba1c": "HbA1c",
    "hdl": "HDL cholesterol",
    "ldl": "LDL cholesterol",
    "triglyceride": "Triglycerides",
    "egfr": "eGFR",
    "CKD": "Chronic kidney disease",
    "Hypertension": "Hypertension",
    "Atrial_fibrillation": "Atrial fibrillation",
    "COPD": "COPD",
    "Asthma": "Asthma",
    "Hyperlipidemia": "Hyperlipidaemia",
    "T1DM": "Type 1 diabetes",
    "T2DM": "Type 2 diabetes",
}

FEATURE_UNITS = {
    "age": "years",
    "bmi": "kg/m²",
    "sbp": "mmHg",
    "dbp": "mmHg",
    "hba1c": "mmol/mol",
    "hdl": "mmol/L",
    "ldl": "mmol/L",
    "triglyceride": "mmol/L",
    "egfr": "mL/min/1.73 m²",
}

BINARY_FEATURES = {
    "CKD",
    "Hypertension",
    "Atrial_fibrillation",
    "COPD",
    "Asthma",
    "Hyperlipidemia",
    "T1DM",
    "T2DM",
}

CLINICAL_LIMITS = {
    "age": (18, 110),
    "IMD": (1, 5),
    "bmi": (10.0, 70.0),
    "sbp": (60.0, 260.0),
    "dbp": (30.0, 160.0),
    "hba1c": (20.0, 150.0),
    "hdl": (0.2, 3.5),
    "ldl": (0.2, 8.0),
    "triglyceride": (0.2, 10.0),
    "egfr": (5.0, 200.0),
}


# ==========================================================
# LOADERS
# ==========================================================

@st.cache_data(show_spinner="Loading clinical dataset...")
def load_feature_data() -> pd.DataFrame:
    path = BASE_DIR / "feature_dataset.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Could not find {path}")
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def load_training_reference() -> pd.DataFrame:
    path = BASE_DIR / "X_train.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Could not find {path}")
    return pd.read_parquet(path)


@st.cache_resource(show_spinner="Loading prediction model...")
def load_model_resources() -> dict:
    model_path = MODEL_DIR / "final_lightgbm_model.pkl"
    threshold_path = MODEL_DIR / "optimal_threshold.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"Could not find {model_path}")

    model = joblib.load(model_path)
    threshold = (
        float(joblib.load(threshold_path))
        if threshold_path.exists()
        else 0.10
    )

    resources = {
        "model": model,
        "threshold": threshold,
        "features": list(model.feature_name_),
    }

    ethnicity_encoder_path = MODEL_DIR / "ethnicity_encoder.pkl"
    if ethnicity_encoder_path.exists():
        resources["ethnicity_encoder"] = joblib.load(
            ethnicity_encoder_path
        )

    return resources


@st.cache_data(show_spinner="Loading trajectory summaries...")
def load_trajectory_outputs() -> dict:
    paths = {
        "trajectories": (
            TRAJECTORY_DIR / "trajectory_summary_dashboard.parquet"
        ),
        "transitions": (
            TRAJECTORY_DIR / "transition_summary_dashboard.parquet"
        ),
        "onset_summary": (
            TRAJECTORY_DIR / "condition_onset_summary.parquet"
        ),
        "yearly_counts": (
            TRAJECTORY_DIR / "yearly_onset_counts.parquet"
        ),
        "length_summary": (
            TRAJECTORY_DIR / "trajectory_length_summary.parquet"
        ),
    }

    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing trajectory files:\n" + "\n".join(missing)
        )

    return {
        name: pd.read_parquet(path)
        for name, path in paths.items()
    }


# ==========================================================
# HELPERS
# ==========================================================

def display_name(feature: str) -> str:
    return DISPLAY_NAMES.get(
        feature,
        feature.replace("_", " ").title(),
    )


def format_feature_label(feature: str) -> str:
    unit = FEATURE_UNITS.get(feature)
    name = display_name(feature)
    return f"{name} ({unit})" if unit else name


def dataframe_memory_mb(dataframe: pd.DataFrame) -> float:
    return dataframe.memory_usage(deep=True).sum() / (1024 ** 2)


def create_numeric_input(
    feature: str,
    training_reference: pd.DataFrame,
):
    lower, upper = CLINICAL_LIMITS[feature]
    median = float(
        pd.to_numeric(
            training_reference[feature],
            errors="coerce",
        ).median()
    )
    median = min(max(median, lower), upper)

    if feature in {"age", "IMD"}:
        return st.number_input(
            format_feature_label(feature),
            min_value=int(lower),
            max_value=int(upper),
            value=int(round(median)),
            step=1,
        )

    return st.number_input(
        format_feature_label(feature),
        min_value=float(lower),
        max_value=float(upper),
        value=float(round(median, 2)),
        step=0.1,
    )


def prepare_patient_dataframe(
    patient_values: dict,
    model_features: list[str],
    resources: dict,
) -> pd.DataFrame:
    patient = pd.DataFrame([patient_values])

    if "gender" in patient.columns:
        patient["gender"] = (
            patient["gender"]
            .map({1: 0, 2: 1})
            .astype(int)
        )

    if "Ethnicity" in patient.columns:
        if "ethnicity_encoder" not in resources:
            raise ValueError(
                "The model expects Ethnicity, but the saved "
                "ethnicity encoder was not found."
            )

        encoder = resources["ethnicity_encoder"]
        ethnicity_value = str(
            patient.loc[0, "Ethnicity"]
        ).strip().lower()

        encoder_classes = [str(value) for value in encoder.classes_]

        if ethnicity_value not in encoder_classes:
            ethnicity_value = (
                "unknown"
                if "unknown" in encoder_classes
                else encoder_classes[0]
            )

        patient.loc[0, "Ethnicity"] = int(
            encoder.transform([ethnicity_value])[0]
        )

    missing_features = [
        feature
        for feature in model_features
        if feature not in patient.columns
    ]

    if missing_features:
        raise ValueError(
            "Missing model inputs: " + ", ".join(missing_features)
        )

    patient = patient.reindex(columns=model_features)

    for column in patient.columns:
        patient[column] = pd.to_numeric(
            patient[column],
            errors="raise",
        )

    return patient


def clinical_summary(
    patient: pd.DataFrame,
    probability: float,
    threshold: float,
) -> str:
    findings = []

    def get_value(feature: str):
        if feature not in patient.columns:
            return None
        return float(patient[feature].iloc[0])

    age = get_value("age")
    hba1c = get_value("hba1c")
    sbp = get_value("sbp")
    ldl = get_value("ldl")
    egfr = get_value("egfr")
    bmi = get_value("bmi")

    if age is not None and age >= 70:
        findings.append("older age")
    if hba1c is not None and hba1c > 58:
        findings.append("elevated HbA1c")
    if sbp is not None and sbp >= 140:
        findings.append("elevated systolic blood pressure")
    if ldl is not None and ldl > 3:
        findings.append("raised LDL cholesterol")
    if egfr is not None and egfr < 60:
        findings.append("reduced kidney function")
    if bmi is not None and bmi >= 30:
        findings.append("BMI in the obesity range")

    for feature, label in [
        ("CKD", "chronic kidney disease"),
        ("Hypertension", "hypertension"),
        ("Atrial_fibrillation", "atrial fibrillation"),
    ]:
        if (
            feature in patient.columns
            and int(patient[feature].iloc[0]) == 1
        ):
            findings.append(label)

    position = "above" if probability >= threshold else "below"

    if findings:
        findings_text = ", ".join(findings)
    else:
        findings_text = (
            "few major rule-based risk factors were identified"
        )

    return (
        f"The estimated five-year cardiovascular risk is "
        f"{probability:.1%}, which is {position} the model "
        f"threshold of {threshold:.1%}. Relevant entered "
        f"characteristics include: {findings_text}."
    )


# ==========================================================
# LOAD APP RESOURCES
# ==========================================================

try:
    feature_data = load_feature_data()
    training_reference = load_training_reference()
    resources = load_model_resources()
    trajectory_outputs = load_trajectory_outputs()
except Exception as error:
    st.error(f"Application could not start: {error}")
    st.stop()

model = resources["model"]
optimal_threshold = resources["threshold"]
model_features = list(model.feature_name_)

trajectory_summary = trajectory_outputs["trajectories"]
transition_summary = trajectory_outputs["transitions"]
condition_onset_summary = trajectory_outputs["onset_summary"]
yearly_onset_counts = trajectory_outputs["yearly_counts"]


# ==========================================================
# HEADER
# ==========================================================

st.title("Five-Year Cardiovascular Risk Dashboard")

st.caption(
    "Exploratory analysis, individual risk prediction and "
    "post-diabetes disease trajectories."
)

st.warning(
    "Research and educational use only. This dashboard is "
    "not a substitute for clinical judgement or a validated "
    "medical device."
)


# ==========================================================
# TABS
# ==========================================================

tab_home, tab_eda, tab_prediction, tab_events = st.tabs(
    [
        "🏠 Home",
        "📊 Exploratory Data Analysis",
        "❤️ Individual Prediction",
        "📈 Disease Trajectories",
    ]
)


# ==========================================================
# HOME
# ==========================================================

with tab_home:
    st.header("Cardiovascular Disease Risk Prediction Dashboard")

    st.markdown(
        """
This dashboard predicts the **five-year risk of a cardiovascular
event** following diabetes diagnosis using an optimised LightGBM
model.

### Model summary

- Algorithm: **LightGBM**
- Hyperparameter optimisation: **Optuna**
- Missing-data method: **MICE**
- Explainability: **SHAP**
- Cohort size: **447,776 patients**

### Model performance

| Metric | Value |
|---|---:|
| ROC-AUC | 0.761 |
| Sensitivity | 68.9% |
| Specificity | 70.3% |
| Precision | 20.2% |
| F1-score | 0.313 |
"""
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Patients", f"{len(feature_data):,}")
    col2.metric("Model features", f"{len(model_features):,}")
    col3.metric(
        "Five-year event rate",
        (
            f"{feature_data['cv_within_5y'].mean():.2%}"
            if "cv_within_5y" in feature_data.columns
            else "Unavailable"
        ),
    )
    col4.metric(
        "Decision threshold",
        f"{optimal_threshold:.1%}",
    )


# ==========================================================
# TAB 1 — EDA
# ==========================================================

with tab_eda:
    st.header("Exploratory Data Analysis")

    available_features = [
        feature
        for feature in model_features
        if feature in feature_data.columns
    ]

    selected_feature = st.selectbox(
        "Select a feature",
        options=available_features,
        format_func=display_name,
        key="eda_feature",
    )

    selected_series = feature_data[selected_feature]
    missing_percentage = selected_series.isna().mean() * 100

    col_summary, col_chart = st.columns([1, 2])

    with col_summary:
        st.subheader("Feature summary")
        st.metric("Missing values", f"{missing_percentage:.2f}%")

        is_continuous = (
            pd.api.types.is_numeric_dtype(selected_series)
            and selected_feature not in BINARY_FEATURES
        )

        if is_continuous:
            summary = selected_series.describe()
            st.metric("Mean", f"{summary['mean']:.2f}")
            st.metric("Median", f"{selected_series.median():.2f}")
            st.metric("Standard deviation", f"{summary['std']:.2f}")
            st.dataframe(
                summary.to_frame("Value"),
                use_container_width=True,
            )
        else:
            counts = (
                selected_series.astype(str)
                .value_counts(dropna=False)
                .rename_axis("Category")
                .reset_index(name="Patients")
            )
            st.dataframe(
                counts,
                use_container_width=True,
                hide_index=True,
            )

    with col_chart:
        st.subheader(
            f"Distribution of {display_name(selected_feature)}"
        )

        if is_continuous:
            plotting_data = feature_data[
                [selected_feature]
            ].dropna()

            figure = px.histogram(
                plotting_data,
                x=selected_feature,
                nbins=40,
                marginal="box",
                labels={
                    selected_feature: format_feature_label(
                        selected_feature
                    )
                },
            )
        else:
            counts = (
                selected_series.astype(str)
                .value_counts(dropna=False)
                .rename_axis("Category")
                .reset_index(name="Patients")
            )

            figure = px.bar(
                counts,
                x="Category",
                y="Patients",
            )

        figure.update_layout(template="plotly_white")
        st.plotly_chart(figure, use_container_width=True)

    if "cv_within_5y" in feature_data.columns:
        st.subheader(
            "Feature comparison by cardiovascular outcome"
        )

        comparison_data = feature_data[
            [selected_feature, "cv_within_5y"]
        ].dropna()

        comparison_data["Outcome"] = comparison_data[
            "cv_within_5y"
        ].map({0: "No event", 1: "CV event"})

        if is_continuous:
            comparison_figure = px.box(
                comparison_data,
                x="Outcome",
                y=selected_feature,
                points=False,
                labels={
                    selected_feature: format_feature_label(
                        selected_feature
                    )
                },
            )
        else:
            grouped = (
                comparison_data.groupby(
                    ["Outcome", selected_feature],
                    observed=True,
                )
                .size()
                .reset_index(name="Patients")
            )

            comparison_figure = px.bar(
                grouped,
                x=selected_feature,
                y="Patients",
                color="Outcome",
                barmode="group",
            )

        comparison_figure.update_layout(template="plotly_white")
        st.plotly_chart(
            comparison_figure,
            use_container_width=True,
        )


# ==========================================================
# TAB 2 — INDIVIDUAL PREDICTION
# ==========================================================



with tab_prediction:
    st.header("Individual Patient Prediction")

    st.write(
        "Enter patient characteristics to estimate the "
        "probability of a cardiovascular event within five years."
        "Diabetes type is included because it was one of the predictors used "
        "during model development. In this model, age, kidney function, blood pressure "
         "and lipid measures contribute more strongly to the estimated risk than "
        "diabetes type alone."
    )

    with st.form("patient_prediction_form"):
        patient_values = {}

        st.subheader("Demographics")
        demographic_col_1, demographic_col_2 = st.columns(2)

        with demographic_col_1:
            if "age" in model_features:
                patient_values["age"] = create_numeric_input(
                    "age",
                    training_reference,
                )

            if "gender" in model_features:
                patient_values["gender"] = st.selectbox(
                    "Gender",
                    options=[1, 2],
                    format_func=lambda value: (
                        "Category 1"
                        if value == 1
                        else "Category 2"
                    ),
                )

        with demographic_col_2:
            if "IMD" in model_features:
                patient_values["IMD"] = create_numeric_input(
                    "IMD",
                    training_reference,
                )

            if "Ethnicity" in model_features:
                patient_values["Ethnicity"] = st.selectbox(
                    "Ethnicity",
                    [
                        "white",
                        "south asian",
                        "black",
                        "other asian",
                        "other",
                        "unknown",
                    ],
                )

        st.subheader("Clinical measurements")
        clinical_columns = st.columns(3)

        continuous_features = [
            "bmi",
            "sbp",
            "dbp",
            "hba1c",
            "hdl",
            "ldl",
            "triglyceride",
            "egfr",
        ]

        for index, feature in enumerate(continuous_features):
            if feature not in model_features:
                continue

            with clinical_columns[index % 3]:
                patient_values[feature] = create_numeric_input(
                    feature,
                    training_reference,
                )
                
        # ==========================================================
        # Diabetes type
        # ==========================================================

        st.subheader("Diabetes type")

        diabetes_col_1, diabetes_col_2 = st.columns(2)


        with diabetes_col_1:
            if "T1DM" in model_features:
                patient_values["T1DM"] = int(
                    st.checkbox(
                        "Type 1 diabetes",
                        value=False,
                        key="prediction_t1dm",
                    )
            )

        with diabetes_col_2:
            if "T2DM" in model_features:
                patient_values["T2DM"] = int(
                    st.checkbox(
                        "Type 2 diabetes",
                        value=True,
                        key="prediction_t2dm",
                    )
                )
        
        
              
        
        
        st.subheader("Medical history")

        history_columns = st.columns(3)

        history_features = [
            feature
            for feature in sorted(BINARY_FEATURES)
            if feature not in {"T1DM", "T2DM"}
            ]
       
        for index, feature in enumerate(history_features):
            if feature not in model_features:
                continue

            with history_columns[index % 3]:
                patient_values[feature] = int(
                    st.checkbox(
                        display_name(feature),
                        value=False,
                    )
                )

        submitted = st.form_submit_button(
            "Calculate five-year risk",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        try:
            patient_frame = prepare_patient_dataframe(
                patient_values,
                model_features,
                resources,
            )
            st.write(patient_frame[["T1DM","T2DM"]])
           # st.write(patient_frame.T)
            #print(model.feature_name_)
            

            predicted_probability = float(
                model.predict_proba(patient_frame)[0, 1]
            )

            is_high_risk = (
                predicted_probability >= optimal_threshold
            )

            st.divider()

            result_col_1, result_col_2 = st.columns(2)

            result_col_1.metric(
                "Predicted five-year risk",
                f"{predicted_probability:.1%}",
            )

            result_col_2.metric(
                "Classification threshold",
                f"{optimal_threshold:.1%}",
            )

            if is_high_risk:
                st.error("Higher-risk classification")
            else:
                st.success("Lower-risk classification")

            risk_percent = predicted_probability * 100
            threshold_percent = optimal_threshold * 100

            gauge_figure = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=risk_percent,
                    number={
                        "suffix": "%",
                        "valueformat": ".1f",
                    },
                    title={
                        "text": "Predicted five-year CVD risk"
                    },
                    gauge={
                        "axis": {"range": [0, 100]},
                        "steps": [
                            {
                                "range": [0, 10],
                                "color": "lightgreen",
                            },
                            {
                                "range": [10, 20],
                                "color": "gold",
                            },
                            {
                                "range": [20, 100],
                                "color": "tomato",
                            },
                        ],
                        "threshold": {
                            "line": {
                                "color": "black",
                                "width": 4,
                            },
                            "thickness": 0.8,
                            "value": threshold_percent,
                        },
                    },
                )
            )

            gauge_figure.update_layout(
                height=350,
                margin={
                    "l": 30,
                    "r": 30,
                    "t": 70,
                    "b": 20,
                },
            )

            st.plotly_chart(
                gauge_figure,
                use_container_width=True,
            )

            st.subheader("Clinical Summary")

            st.info(
                clinical_summary(
                    patient_frame,
                    predicted_probability,
                    optimal_threshold,
                )
            )

            st.caption(
                "The classification uses the optimised "
                f"threshold of {optimal_threshold:.1%}."
            )

            with st.expander("View processed model inputs"):
                patient_display = patient_frame.T.copy()
                patient_display.columns = ["Model value"]

                st.dataframe(
                    patient_display,
                    use_container_width=True,
                )

        except Exception as error:
            st.exception(error)


# ==========================================================
# TAB 3 — DISEASE TRAJECTORIES
# ==========================================================

with tab_events:
    st.header("Disease Trajectories After Diabetes")

    st.write(
        "Explore aggregated condition sequences recorded after "
        "the diabetes index date. No patient identifiers are shown."
    )

    if trajectory_summary.empty:
        st.warning("No trajectory results are available.")
    else:
        minimum_count = st.slider(
            "Minimum number of patients",
            min_value=20,
            max_value=500,
            value=20,
            step=10,
        )

        maximum_conditions = int(
            trajectory_summary["number_of_conditions"].max()
        )

        condition_count_range = st.slider(
            "Number of conditions in trajectory",
            min_value=1,
            max_value=max(1, maximum_conditions),
            value=(1, max(1, maximum_conditions)),
        )

        top_n = st.slider(
            "Number of trajectories to display",
            min_value=5,
            max_value=50,
            value=15,
            step=5,
        )

        filtered_trajectories = (
            trajectory_summary[
                (
                    trajectory_summary["Patients"]
                    >= minimum_count
                )
                & (
                    trajectory_summary[
                        "number_of_conditions"
                    ].between(
                        condition_count_range[0],
                        condition_count_range[1],
                    )
                )
            ]
            .head(top_n)
            .copy()
        )

        metric_1, metric_2, metric_3 = st.columns(3)

        metric_1.metric(
            "Displayed trajectories",
            f"{len(filtered_trajectories):,}",
        )

        metric_2.metric(
            "Conditions analysed",
            f"{condition_onset_summary.shape[0]:,}",
        )

        metric_3.metric(
            "Transitions available",
            f"{len(transition_summary):,}",
        )

        st.subheader("Most Common Complete Trajectories")

        if filtered_trajectories.empty:
            st.info(
                "No trajectories meet the selected filters."
            )
        else:
            chart_data = filtered_trajectories.sort_values(
                "Patients",
                ascending=True,
            )

            hover_data = {
                "number_of_conditions": True,
                "Median_time_to_first_years": ":.2f",
                "Median_time_to_last_years": ":.2f",
            }

            if "number_of_steps" in chart_data.columns:
                hover_data["number_of_steps"] = True

            trajectory_figure = px.bar(
                chart_data,
                x="Patients",
                y="trajectory",
                orientation="h",
                hover_data=hover_data,
            )

            trajectory_figure.update_layout(
                template="plotly_white",
                height=max(450, len(chart_data) * 34),
                xaxis_title="Patients",
                yaxis_title="Trajectory",
            )

            st.plotly_chart(
                trajectory_figure,
                use_container_width=True,
            )

            st.dataframe(
                filtered_trajectories,
                use_container_width=True,
                hide_index=True,
            )

        st.subheader("Most Common Condition Transitions")

        transition_display = (
            transition_summary[
                transition_summary["Transitions"]
                >= minimum_count
            ]
            .head(top_n)
            .copy()
        )

        if not transition_display.empty:
            transition_display["Transition"] = (
                transition_display["From"]
                + " → "
                + transition_display["To"]
            )

            transition_chart_data = (
                transition_display.sort_values(
                    "Transitions",
                    ascending=True,
                )
            )

            transition_hover = {}
            if "Median_years_between" in transition_chart_data.columns:
                transition_hover["Median_years_between"] = ":.2f"
            if (
                "Median_years_after_diabetes"
                in transition_chart_data.columns
            ):
                transition_hover[
                    "Median_years_after_diabetes"
                ] = ":.2f"

            transition_figure = px.bar(
                transition_chart_data,
                x="Transitions",
                y="Transition",
                orientation="h",
                hover_data=transition_hover,
            )

            transition_figure.update_layout(
                template="plotly_white",
                height=max(
                    450,
                    len(transition_chart_data) * 34,
                ),
                xaxis_title="Transitions",
                yaxis_title="Condition transition",
            )

            st.plotly_chart(
                transition_figure,
                use_container_width=True,
            )
        else:
            st.info(
                "No transitions meet the selected minimum count."
            )

        st.subheader("New Diagnoses by Follow-up Year")

        if yearly_onset_counts.empty:
            st.info("No yearly onset counts are available.")
        else:
            yearly_chart = px.line(
                yearly_onset_counts,
                x="year_interval_label",
                y="New_diagnoses",
                color="condition",
                markers=True,
                labels={
                    "year_interval_label": "Years after diabetes",
                    "New_diagnoses": "New diagnoses",
                },
            )

            yearly_chart.update_layout(
                template="plotly_white"
            )

            st.plotly_chart(
                yearly_chart,
                use_container_width=True,
            )

        st.subheader("Condition Onset Summary")

        st.dataframe(
            condition_onset_summary,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download aggregate trajectory results",
            data=filtered_trajectories.to_csv(index=False),
            file_name="aggregated_disease_trajectories.csv",
            mime="text/csv",
        )


# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:
    st.header("Model information")
    st.write(f"Features: **{len(model_features)}**")
    st.write(
        f"Decision threshold: **{optimal_threshold:.1%}**"
    )

    if hasattr(model, "n_estimators_"):
        st.write(f"Trees: **{model.n_estimators_:,}**")

    st.divider()

    st.caption(
        "Do not enter or display patient-identifying "
        "information in this application."
    )
    
    
importance = pd.DataFrame({
    "Feature": model.feature_name_,
    "Gain": model.booster_.feature_importance(
        importance_type="gain"
    )
})

importance = importance.sort_values(
    "Gain",
    ascending=False
)

print(importance)