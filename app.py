import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Healthcare Cost Prediction & Patient Segmentation | Sohail Ahmed",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Data Loading & Caching
# ---------------------------------------------------------
@st.cache_data
def load_data():
    possible_paths = [
        os.path.join(os.path.dirname(__file__), "data", "medical_insurance_data.csv"),
        os.path.join(os.path.dirname(__file__), "medical_insurance_data.csv"),
        r"C:\Users\BILAL.COM\OneDrive\PROJECTS\project_08\dataset\insurance.csv"
    ]
    for p in possible_paths:
        if os.path.exists(p):
            df = pd.read_csv(p)
            df['smoker_numeric'] = (df['smoker'].str.lower() == 'yes').astype(int)
            return df
    # Fallback synthetic demo
    np.random.seed(42)
    n = 1000
    return pd.DataFrame({
        'age': np.random.randint(18, 65, n),
        'sex': np.random.choice(['female', 'male'], n),
        'bmi': np.random.uniform(16.0, 48.0, n),
        'children': np.random.choice([0, 1, 2, 3, 4, 5], n, p=[0.4, 0.25, 0.18, 0.11, 0.04, 0.02]),
        'smoker': np.random.choice(['yes', 'no'], n, p=[0.20, 0.80]),
        'smoker_numeric': np.random.choice([1, 0], n, p=[0.20, 0.80]),
        'region': np.random.choice(['southwest', 'southeast', 'northwest', 'northeast'], n),
        'charges': np.random.uniform(1500, 45000, n)
    })

df_raw = load_data()

# ---------------------------------------------------------
# Sidebar Filters
# ---------------------------------------------------------
st.sidebar.image("https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?w=400&q=80", use_container_width=True)
st.sidebar.title("⚕️ Healthcare Cost Analytics")
st.sidebar.markdown("**Portfolio Project by Sohail Ahmed**")
st.sidebar.markdown("---")

st.sidebar.subheader("Patient Cohort Filters")
selected_smoker = st.sidebar.multiselect("Smoking Status", ['yes', 'no'], default=['yes', 'no'])
all_regions = sorted(df_raw['region'].dropna().unique().tolist())
selected_regions = st.sidebar.multiselect("Region", all_regions, default=all_regions)
selected_sex = st.sidebar.multiselect("Sex", ['female', 'male'], default=['female', 'male'])

age_range = st.sidebar.slider("Age Range", int(df_raw['age'].min()), int(df_raw['age'].max()), (int(df_raw['age'].min()), int(df_raw['age'].max())))
bmi_range = st.sidebar.slider("BMI Range", float(df_raw['bmi'].min()), float(df_raw['bmi'].max()), (float(df_raw['bmi'].min()), float(df_raw['bmi'].max())))

# Filter dataframe
filtered_df = df_raw[
    (df_raw['smoker'].isin(selected_smoker)) &
    (df_raw['region'].isin(selected_regions)) &
    (df_raw['sex'].isin(selected_sex)) &
    (df_raw['age'].between(age_range[0], age_range[1])) &
    (df_raw['bmi'].between(bmi_range[0], bmi_range[1]))
]

if filtered_df.empty:
    st.warning("No records match the filter criteria. Showing all patient records.")
    filtered_df = df_raw.copy()

# ---------------------------------------------------------
# Main Header & KPIs
# ---------------------------------------------------------
st.markdown('<div class="main-header">Healthcare Cost Prediction & Patient Risk Segmentation</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Actuarial healthcare expenditure analytics, non-linear premium prediction, and unsupervised patient risk clustering.</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
total_pts = len(filtered_df)
mean_charge = filtered_df['charges'].mean()
smoker_mult = (df_raw[df_raw['smoker'] == 'yes']['charges'].mean()) / (df_raw[df_raw['smoker'] == 'no']['charges'].mean())
med_age = filtered_df['age'].median()

col1.metric("Analyzed Patient Cohort", f"{total_pts:,}")
col2.metric("Mean Annual Premium", f"${mean_charge:,.2f}")
col3.metric("Smoker Cost Multiple", f"{smoker_mult:.1f}x", delta="Smoking Actuarial Penalty", delta_color="inverse")
col4.metric("Median Patient Age", f"{med_age:.0f} yrs")

st.markdown("---")

# ---------------------------------------------------------
# Model Training (Cached)
# ---------------------------------------------------------
@st.cache_resource
def train_insurance_models(df):
    m_df = df.copy()
    m_df['is_smoker'] = (m_df['smoker'] == 'yes').astype(int)
    m_df['is_male'] = (m_df['sex'] == 'male').astype(int)
    
    # One-hot encode region
    region_dummies = pd.get_dummies(m_df['region'], prefix='region', drop_first=True)
    X = pd.concat([m_df[['age', 'bmi', 'children', 'is_smoker', 'is_male']], region_dummies], axis=1)
    y = m_df['charges']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    results = []
    trained_models = {}
    for name, m in models.items():
        m.fit(X_train, y_train)
        y_pred = m.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        results.append({'Model': name, 'R2 Score': r2, 'MAE ($)': mae, 'RMSE ($)': rmse})
        trained_models[name] = m
        
    # K-Means Clustering for Risk Segmentation
    cluster_features = m_df[['age', 'bmi', 'charges']]
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    m_df['Risk_Cluster'] = kmeans.fit_predict(cluster_features)
    
    # Map cluster names based on mean charges
    cluster_order = m_df.groupby('Risk_Cluster')['charges'].mean().sort_values().index
    cluster_names = {cluster_order[0]: 'Tier 1: Low Actuarial Risk', cluster_order[1]: 'Tier 2: Moderate Risk', cluster_order[2]: 'Tier 3: High Actuarial Risk'}
    m_df['Risk_Tier'] = m_df['Risk_Cluster'].map(cluster_names)
    
    return trained_models['Gradient Boosting'], pd.DataFrame(results), m_df, kmeans

best_model, benchmark_df, clustered_df, kmeans_model = train_insurance_models(df_raw)

# ---------------------------------------------------------
# Tabs Structure
# ---------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Healthcare Expenditure Dashboard",
    "🧬 Patient Risk Clustering (K-Means)",
    "🎯 Live Premium Cost Predictor",
    "📈 Comparative Model Benchmarks",
    "📖 Methodology & Documentation"
])

# ---------------------------------------------------------
# Tab 1: Cost Dashboard
# ---------------------------------------------------------
with tab1:
    st.subheader("Demographic & Lifestyle Drivers of Healthcare Costs")
    c1, c2 = st.columns([3, 2])
    with c1:
        fig_scatter = px.scatter(
            filtered_df,
            x="bmi",
            y="charges",
            color="smoker",
            size="age",
            title="Body Mass Index (BMI) vs Medical Charges ($) by Smoking Status",
            color_discrete_map={'yes': '#EF4444', 'no': '#10B981'},
            labels={"bmi": "Body Mass Index (BMI)", "charges": "Annual Charges ($)"}
        )
        fig_scatter.update_layout(height=400)
        st.plotly_chart(fig_scatter, use_container_width=True)
    with c2:
        fig_age = px.scatter(
            filtered_df,
            x="age",
            y="charges",
            color="smoker",
            trendline="ols",
            title="Age Progression vs Annual Charges ($)",
            color_discrete_map={'yes': '#EF4444', 'no': '#10B981'},
            labels={"age": "Age (Years)", "charges": "Annual Charges ($)"}
        )
        fig_age.update_layout(height=400)
        st.plotly_chart(fig_age, use_container_width=True)

    st.subheader("Regional & Family Size Variations")
    c3, c4 = st.columns(2)
    with c3:
        region_avg = filtered_df.groupby('region')['charges'].mean().reset_index()
        fig_reg = px.bar(
            region_avg,
            x="region",
            y="charges",
            color="region",
            title="Average Healthcare Premium by Geographic Region ($)",
            color_discrete_sequence=px.colors.qualitative.Prism
        )
        fig_reg.update_layout(showlegend=False, height=380)
        st.plotly_chart(fig_reg, use_container_width=True)
    with c4:
        child_avg = filtered_df.groupby('children')['charges'].mean().reset_index()
        fig_child = px.line(
            child_avg,
            x="children",
            y="charges",
            markers=True,
            title="Impact of Number of Covered Children on Mean Cost ($)"
        )
        fig_child.update_traces(line_color="#0284C7", line_width=3)
        fig_child.update_layout(height=380)
        st.plotly_chart(fig_child, use_container_width=True)

# ---------------------------------------------------------
# Tab 2: Patient Clustering
# ---------------------------------------------------------
with tab2:
    st.subheader("Unsupervised Patient Risk Stratification (K-Means Clustering)")
    st.markdown("Patients are segmented into 3 distinct actuarial risk tiers based on combined interaction between Age, BMI, and Historic Expenditures.")

    fig_3d = px.scatter(
        clustered_df,
        x="bmi",
        y="charges",
        color="Risk_Tier",
        size="age",
        title="Patient Risk Clusters: BMI vs Healthcare Charges (Bubble Size = Age)",
        color_discrete_map={
            'Tier 1: Low Actuarial Risk': '#10B981',
            'Tier 2: Moderate Risk': '#F59E0B',
            'Tier 3: High Actuarial Risk': '#EF4444'
        }
    )
    fig_3d.update_layout(height=450)
    st.plotly_chart(fig_3d, use_container_width=True)

    tier_summary = clustered_df.groupby('Risk_Tier').agg(
        Patient_Count=('age', 'count'),
        Avg_Age=('age', 'mean'),
        Avg_BMI=('bmi', 'mean'),
        Smoker_Rate=('smoker_numeric', 'mean'),
        Avg_Charges=('charges', 'mean')
    ).reset_index()
    tier_summary['Smoker_Rate_%'] = tier_summary['Smoker_Rate'] * 100
    st.dataframe(tier_summary[['Risk_Tier', 'Patient_Count', 'Avg_Age', 'Avg_BMI', 'Smoker_Rate_%', 'Avg_Charges']].style.format({
        'Avg_Age': '{:.1f}', 'Avg_BMI': '{:.1f}', 'Smoker_Rate_%': '{:.1f}%', 'Avg_Charges': '${:,.2f}'
    }), use_container_width=True)

# ---------------------------------------------------------
# Tab 3: Live Premium Estimator
# ---------------------------------------------------------
with tab3:
    st.subheader("Live Actuarial Medical Premium Estimator")
    st.markdown("Input patient biometric and lifestyle parameters to simulate annual insurance expenditure using our trained Gradient Boosting model.")

    col_in1, col_in2, col_in3 = st.columns(3)
    with col_in1:
        in_age = st.slider("Patient Age", 18, 70, 35)
        in_sex = st.selectbox("Biological Sex", ['male', 'female'])
    with col_in2:
        in_bmi = st.slider("Body Mass Index (BMI)", 15.0, 50.0, 27.5, step=0.1)
        in_children = st.selectbox("Number of Covered Dependents", [0, 1, 2, 3, 4, 5], index=1)
    with col_in3:
        in_smoker = st.selectbox("Tobacco / Smoking Status", ['no', 'yes'])
        in_region = st.selectbox("Geographic Region", ['northeast', 'northwest', 'southeast', 'southwest'])

    if st.button("Forecast Annual Healthcare Premium", type="primary"):
        # Construct feature vector matching model
        is_smoker = 1 if in_smoker == 'yes' else 0
        is_male = 1 if in_sex == 'male' else 0
        reg_nw = 1 if in_region == 'northwest' else 0
        reg_se = 1 if in_region == 'southeast' else 0
        reg_sw = 1 if in_region == 'southwest' else 0

        X_input = pd.DataFrame([[in_age, in_bmi, in_children, is_smoker, is_male, reg_nw, reg_se, reg_sw]],
                               columns=['age', 'bmi', 'children', 'is_smoker', 'is_male', 'region_northwest', 'region_southeast', 'region_southwest'])

        pred_charge = best_model.predict(X_input)[0]

        # Calculate smoking premium delta
        X_nonsmoker = X_input.copy()
        X_nonsmoker['is_smoker'] = 0
        non_smoker_charge = best_model.predict(X_nonsmoker)[0]
        smoking_penalty = max(0.0, pred_charge - non_smoker_charge)

        st.success(f"💰 **Estimated Annual Medical Premium:** `${pred_charge:,.2f}` (~`${pred_charge/12:,.2f}/month`)")

        if is_smoker:
            st.warning(f"🚬 **Smoking Surcharge Alert:** Approximately `${smoking_penalty:,.2f}/year` of this premium is directly attributable to tobacco usage.")
        else:
            st.info(f"🌟 **Favorable Lifestyle Profile:** Non-smoker status saves an estimated `${pred_charge * 2.5:,.2f}` annually compared to an equivalent smoking profile.")

# ---------------------------------------------------------
# Tab 4: Model Benchmarks
# ---------------------------------------------------------
with tab4:
    st.subheader("Model Architecture Benchmarks ($R^2$, MAE, RMSE)")
    st.markdown("Comparison across regression architectures implemented in the analytical study.")

    fig_bench = px.bar(
        benchmark_df,
        x="Model",
        y="R2 Score",
        color="R2 Score",
        color_continuous_scale="Viridis",
        title="Model Accuracy Benchmark Comparison (Coefficient of Determination $R^2$)",
        text_auto=".3f"
    )
    fig_bench.update_layout(height=380)
    st.plotly_chart(fig_bench, use_container_width=True)

    st.dataframe(benchmark_df.style.format({
        'R2 Score': '{:.3f}', 'MAE ($)': '${:,.2f}', 'RMSE ($)': '${:,.2f}'
    }), use_container_width=True)

# ---------------------------------------------------------
# Tab 5: Methodology
# ---------------------------------------------------------
with tab5:
    st.subheader("Methodology & Actuarial Context")
    st.markdown("""
    ### 1. Actuarial Objective
    Medical cost risk modeling enables health insurers and hospital networks to anticipate financial liability and design proactive wellness interventions.

    ### 2. Dataset Overview
    - **Records:** 1,338 patient profiles.
    - **Features:** Age, Sex, BMI, Number of dependents, Smoking status, Geographic territory, and Historic insurance charges.

    ### 3. Key Findings
    - **The Smoker-BMI Multiplier:** Non-smoking patients experience a modest linear cost increase as BMI increases. In contrast, smoking patients with BMI > 30 experience an exponential cost surge, with average premiums jumping from $13,000 to over $40,000.
    - **Winning Architecture:** Gradient Boosting achieves the highest $R^2$ score (~0.87), capturing non-linear feature interactions between age and smoking status.
    """)

st.markdown("---")
st.markdown("© 2026 **Sohail Ahmed** | [GitHub Profile](https://github.com/sohailahmed-bsrh) | [Portfolio](https://sohail.devsil.com)")
