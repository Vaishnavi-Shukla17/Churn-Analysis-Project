# app.py
import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, roc_auc_score, confusion_matrix, precision_recall_curve
)
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# --------------------------
# Page config
# --------------------------
st.set_page_config(
    page_title="Customer Churn Prediction Platform",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={'About': "This is my awesome web app"}
)

# --------------------------
# Minimal CSS
# --------------------------
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 1.25rem;
}
.metric-card {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 0.5rem;
    color: white;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# --------------------------
# Sidebar navigation
# --------------------------
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Choose a page:",
    ["Overview", "Data Analysis", "Monitoring", "Model Performance", "Churn Prediction", "About This"]
)

# --------------------------
# Caching
# --------------------------
@st.cache_data
def load_and_process_data(uploaded_file):
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    # Generic: try to convert object columns to numeric where possible
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors="ignore")

    return df


@st.cache_resource
def train_model(X_train, y_train):
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    return model


def preprocess_for_inference(df_train, inputs):
    sample = {}

    for col in df_train.columns:
        if col in inputs:
            sample[col] = inputs[col]
        else:
            if df_train[col].dtype in ["int64", "float64"]:
                sample[col] = df_train[col].median()
            else:
                sample[col] = df_train[col].mode()[0]

    sample_df = pd.DataFrame([sample])

    # Apply same encoding
    sample_encoded = pd.get_dummies(sample_df)

    # Align with training columns
    feature_cols = st.session_state.feature_columns

    sample_encoded = sample_encoded.reindex(columns=feature_cols, fill_value=0)

    return sample_encoded

# --------------------------
# Pages
# --------------------------
def show_overview():
    st.markdown('<h1 class="main-header">Customer Churn Prediction Platform</h1>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>Purpose</h3>
            <p>Identify at-risk customers before they churn</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>Technology</h3>
            <p>XGBoost + SHAP + Streamlit</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>Impact</h3>
            <p>Save revenue through targeted retention</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Platform Features")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        ### Analytics & Insights
        - Interactive data exploration
        - Customer segmentation analysis
        - Churn pattern identification
        - Revenue impact calculation
        """)
    with c2:
        st.markdown("""
        ### AI-Powered Predictions
        - XGBoost machine learning model
        - SHAP explainability analysis
        - Single and batch predictions
        - Risk scoring and prioritization
        """)

    st.markdown("---")
    st.subheader("Get Started")
    st.info("Upload your customer dataset in the Data Analysis page to begin.")


def show_data_analysis():
    st.title("Data Analysis & Exploration")

    uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "xlsx"])
    if uploaded_file is None:
        st.info("Upload a dataset to begin.")
        return

    df = load_and_process_data(uploaded_file)

        # Drop useless ID column
    if 'customerID' in df.columns:
        df.drop(columns=['customerID'], inplace=True)

    # Fix TotalCharges column (very important)
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

    # --------------------------
    # Target Column Selection
    # --------------------------
    st.subheader("Select Target Column")

    target_column = st.selectbox(
        "Choose the column you want to predict",
        df.columns
    )

    # Store in session
    st.session_state.target_column = target_column

    # Make target binary in a generic way
    if df[target_column].dtype == object:
        unique_vals = df[target_column].dropna().unique().tolist()
        # Common Yes/No case
        if set(unique_vals) == set(["Yes", "No"]) or set(unique_vals) == set(["No", "Yes"]):
            df[target_column] = df[target_column].map({"Yes": 1, "No": 0})
        else:
            counts = df[target_column].value_counts()
            if len(counts) == 2:
                positive_class = counts.index[-1]  # treat minority as churn by default
                df[target_column] = (df[target_column] == positive_class).astype(int)

    # --------------------------
    # Automatic Data Cleaning
    # --------------------------
    st.subheader("Data Cleaning Process")

    # 1. Remove duplicates
    initial_rows = len(df)
    df = df.drop_duplicates()
    removed_rows = initial_rows - len(df)

    # 2. Handle missing values
    missing_before = df.isnull().sum().sum()

    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)

    missing_after = df.isnull().sum().sum()

    # Update session state after cleaning
    st.session_state.df = df

    # Show cleaning summary
    st.write(f"✅ Removed duplicate rows: {removed_rows}")
    st.write(f"✅ Missing values before: {missing_before}")
    st.write(f"✅ Missing values after: {missing_after}")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    st.session_state.X = X
    st.session_state.y = y

    # --------------------------
    # Automatic Feature Detection
    # --------------------------
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    st.session_state.numerical_cols = numerical_cols
    st.session_state.categorical_cols = categorical_cols

    # Show to user
    st.subheader("Detected Feature Types")

    c1, c2 = st.columns(2)

    with c1:
        st.write("Numerical Features:")
        st.write(numerical_cols)

    with c2:
        st.write("Categorical Features:")
        st.write(categorical_cols)

    st.session_state.df = df.copy()

    st.subheader("Dataset Overview")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Rows", len(df))
    with c2:
        st.metric("Features", len(df.columns))
    with c3:
        missing = int(df.isnull().sum().sum())
        st.metric("Missing Values", f"{missing}")
    with c4:
        if target_column in df.columns and df[target_column].notna().any():

            # Try converting safely
            try:
                churn_rate = pd.to_numeric(df[target_column], errors='coerce').mean() * 100

                if not np.isnan(churn_rate):
                    st.metric("Positive Rate", f"{churn_rate:.1f}%")
                else:
                    st.write("Target Distribution:")
                    st.write(df[target_column].value_counts())
                    
            except:
                st.write("Target Distribution:")
                st.write(df[target_column].value_counts())
                
    st.subheader("Data Preview")
    st.dataframe(df.head(10), use_container_width=True)

    # Visuals in the app (still special-casing 'Churn' if present)
    if 'Churn' in df.columns:
        st.subheader("Interactive Visualizations")
        tab1, tab2 = st.tabs(["Churn Distribution", "Financial Analysis"])

        with tab1:
            c1, c2 = st.columns(2)
            with c1:
                churn_counts = df['Churn'].value_counts()
                if len(churn_counts) > 0:
                    fig_pie = px.pie(
                        values=churn_counts.values,
                        names=['Retained' if k == 0 else 'Churned' for k in churn_counts.index],
                        title="Customer Churn Distribution"
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
            with c2:
                if 'tenure' in df.columns:
                    fig_tenure = px.histogram(
                        df.dropna(subset=['tenure']),
                        x='tenure', color='Churn',
                        title="Churn Distribution by Tenure", barmode='overlay'
                    )
                    st.plotly_chart(fig_tenure, use_container_width=True)

        with tab2:
            if 'MonthlyCharges' in df.columns:
                c1, c2 = st.columns(2)
                with c1:
                    fig_charges = px.box(
                        df.dropna(subset=['MonthlyCharges', 'Churn']),
                        x='Churn', y='MonthlyCharges',
                        title="Monthly Charges vs Churn"
                    )
                    st.plotly_chart(fig_charges, use_container_width=True)
                with c2:
                    if 'TotalCharges' in df.columns:
                        fig_total = px.scatter(
                            df.dropna(subset=['MonthlyCharges', 'TotalCharges', 'Churn']),
                            x='MonthlyCharges', y='TotalCharges',
                            color='Churn', title="Charges Relationship"
                        )
                        st.plotly_chart(fig_total, use_container_width=True)


def show_prediction():
    st.title("Churn Prediction Engine")

    if 'df' not in st.session_state:
        st.warning("Please upload data in the Data Analysis page first.")
        return

    df = st.session_state.df
    target_column = st.session_state.target_column

    # Split X and y
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Apply encoding
    X_encoded = pd.get_dummies(X)

    # Store feature columns
    st.session_state.feature_columns = X_encoded.columns

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train model
    model = train_model(X_train, y_train)

    # Store everything
    st.session_state.model = model
    st.session_state.X_train = X_train
    st.session_state.X_test = X_test
    st.session_state.y_test = y_test

    # Predictions
    y_pred = model.predict(X_test)

    from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

    # Metrics
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    # Display metrics
    st.subheader("Model Performance")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Accuracy", f"{acc:.2%}")
    with c2:
        st.metric("Precision", f"{precision:.2%}")
    with c3:
        st.metric("Recall", f"{recall:.2%}")
    with c4:
        st.metric("F1 Score", f"{f1:.2%}")
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    st.pyplot(fig)

    # ROC curve
    st.subheader("ROC Curve")

    y_prob = model.predict_proba(X_test)[:, 1]

    from sklearn.metrics import roc_curve, auc

    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    ax.plot([0, 1], [0, 1], linestyle='--')

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()

    st.pyplot(fig)


    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    st.subheader("Feature Importance")
    st.dataframe(feature_importance.head(10))
    fig, ax = plt.subplots()
    ax.barh(feature_importance['Feature'][:10], feature_importance['Importance'][:10])
    ax.invert_yaxis()
    ax.set_title("Top 10 Important Features")
    st.pyplot(fig)

    st.subheader("Model Explainability (SHAP)")

    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)

    # Take sample data (small for speed)
    sample_data = X_test.iloc[:50]

    # Calculate SHAP values
    shap_values = explainer.shap_values(sample_data)

    # Summary plot
    st.write("SHAP Summary Plot (Feature Impact)")

    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, sample_data, show=False)
    st.pyplot(fig)

    show_single_prediction(df)

def show_single_prediction(df):
    st.subheader("Single Record Prediction")

    if 'model' not in st.session_state:
        st.warning("Please train the model first.")
        return

    model = st.session_state.model
    X = st.session_state.X
    target_column = st.session_state.target_column

    inputs = {}

    with st.form("dynamic_form"):

        columns = X.columns.tolist()

        for col in columns:

            if X[col].dtype in ["int64", "float64"]:
                inputs[col] = st.number_input(
                    f"{col}",
                    value=float(X[col].median())
                )

            else:
                unique_vals = X[col].dropna().unique().tolist()
                inputs[col] = st.selectbox(
                    f"{col}",
                    unique_vals
                )

        submitted = st.form_submit_button("Predict")

        if submitted:
            # Convert to DataFrame
            sample_df = pd.DataFrame([inputs])

            # One-hot encode
            sample_encoded = pd.get_dummies(sample_df)

            # Align with training columns
            missing_cols = set(st.session_state.feature_columns) - set(sample_encoded.columns)
            for col in missing_cols:
                sample_encoded[col] = 0

            sample_encoded = sample_encoded[st.session_state.feature_columns]

            # Prediction
            prediction_prob = float(model.predict_proba(sample_encoded)[:, 1])

            # Output
            c1, c2 = st.columns(2)

            with c1:
                st.metric("Prediction Probability", f"{prediction_prob:.2%}")

            with c2:
                risk = "High" if prediction_prob > 0.7 else ("Medium" if prediction_prob > 0.4 else "Low")
                st.metric("Risk Level", risk)

            # Recommendations
            st.subheader("Recommended Actions")

            if prediction_prob > 0.7:
                st.error("High Risk - Immediate action needed")
                st.write("- Offer discount")
                st.write("- Contact customer")
            elif prediction_prob > 0.4:
                st.warning("Medium Risk - Monitor user")
                st.write("- Send engagement offers")
            else:
                st.success("Low Risk - No immediate action")


            st.subheader("Why this prediction? (SHAP)")

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(sample_encoded)

            fig, ax = plt.subplots()
            shap.waterfall_plot(shap.Explanation(
                values=shap_values[0],
                base_values=explainer.expected_value,
                data=sample_encoded.iloc[0]
            ), show=False)

            st.pyplot(fig)
def show_model_performance():
    st.title("Model Performance Dashboard")

    if 'df' not in st.session_state:
        st.warning("Please upload data first.")
        return

    # For now still simulated; later we will connect to real y_test/y_pred
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Accuracy", "87.3%", "2.1%")
    with c2:
        st.metric("Precision", "84.7%", "-0.5%")
    with c3:
        st.metric("Recall", "82.1%", "1.3%")
    with c4:
        st.metric("AUC-ROC", "0.91", "0.02")

    st.subheader("Model Performance Trends")
    dates = pd.date_range(start='2023-01-01', periods=12, freq='M')
    accuracy_trend = np.random.normal(0.87, 0.02, 12)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=accuracy_trend, mode='lines+markers', name='Accuracy'))
    fig.update_layout(title="Model Accuracy Over Time", xaxis_title="Month", yaxis_title="Accuracy")
    st.plotly_chart(fig, use_container_width=True)


def show_monitoring():
    st.title("Model Monitoring & Health")

    if 'df' not in st.session_state:
        st.info("Upload data to see monitoring metrics based on your dataset.")
        return

    df = st.session_state.df

    c1, c2, c3 = st.columns(3)

    with c1:
        completeness = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        st.metric("Data Health", f"{completeness:.1f}%",
                  f"{completeness - 95:.1f}%" if completeness < 95 else "+Good")

    with c2:
        estimated_time = len(df) * 0.1  # 0.1ms per record
        st.metric("Est. Processing Time", f"{estimated_time:.0f}ms",
                  "-Fast" if estimated_time < 100 else "+Slow")
    with c3:
        target_column = st.session_state.get("target_column")
        if target_column and target_column in df.columns:
            # If target is still string → show distribution instead of sum
            if df[target_column].dtype == object:
                value_counts = df[target_column].value_counts().to_dict()
                st.metric("Target Distribution", str(value_counts))

            else:
                churn_cases = int(df[target_column].sum())
                st.metric("Positive Cases", f"{churn_cases}",
                          f"+{churn_cases}" if churn_cases > 0 else "None")

        else:
            st.metric("Total Records", f"{len(df):,}", f"+{len(df)}")
    st.subheader("Data Drift Detection")
    st.info("Monitoring feature distributions for changes that might affect model performance")

    features = [c for c in ['MonthlyCharges', 'tenure', 'TotalCharges'] if c in df.columns]
    if not features:
        st.info("No standard features found; using first 3 numeric columns as drift demo.")
        features = df.select_dtypes(include=['int64', 'float64']).columns[:3].tolist()

    drift_scores = np.random.random(len(features))
    drift_df = pd.DataFrame({'Feature': features, 'Drift Score': drift_scores})
    fig_drift = px.bar(drift_df, x='Feature', y='Drift Score', title="Feature Drift Scores (Last 30 Days)")
    st.plotly_chart(fig_drift, use_container_width=True)


def show_documentation():
    st.title("About This")

    st.markdown("""
## Project Overview
This Customer Churn Prediction Platform combines:
- Machine Learning: XGBoost classifier for high-accuracy predictions
- Explainable AI: SHAP values for model interpretability (add later)
- Interactive Analytics: Real-time data exploration and visualization
- Business Intelligence: Actionable insights for retention strategies

## Technical Stack
- Frontend: Streamlit with Plotly for interactive visualizations
- ML Model: XGBoost (upgradable with hyperparameter tuning)
- Data Processing: Pandas and NumPy
- Visualization: Matplotlib, Seaborn, Plotly

## Key Features
- Data Analysis: Data quality checks and interactive visuals
- Prediction Engine: Single-customer scoring (batch optional later)
- Model Performance: Metrics and trend visualization
- Monitoring: Drift indicators (simulated)

## Notes
- Works with a configurable target column; binary mapping applied automatically when possible.
- Numeric columns are coerced automatically when possible.
- For production, consider time-based splits, SHAP explainability, and threshold/ROI policy.
""")


# --------------------------
# Router
# --------------------------
def main():
    try:
        if page == "Overview":
            show_overview()
        elif page == "Data Analysis":
            show_data_analysis()
        elif page == "Monitoring":
            show_monitoring()
        elif page == "Model Performance":
            show_model_performance()
        elif page == "Churn Prediction":
            show_prediction()
        elif page == "About This":
            show_documentation()
        else:
            st.write("Unknown page.")
    except Exception as e:
        st.error("An error occurred while rendering this page.")
        st.exception(e)


if __name__ == "__main__":
    main()
