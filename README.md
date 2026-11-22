# Medical Insurance Cost Prediction - Actuarial Regression Engine

<p align="center">
  <a href="https://sohail.devsil.com"><strong>Portfolio</strong></a> | 
  <a href="https://github.com/sohailahmed-bsrh"><strong>GitHub Profile</strong></a> | 
  <a href="https://github.com/sohailahmed-bsrh/medical-insurance-cost-prediction"><strong>Source Repository</strong></a>
</p>

---

## Executive Summary & Problem Statement

Health insurance underwriting relies on estimating individual expected medical expenditures. Flawed actuarial estimation leads to adverse selection: underpricing premiums triggers severe underwriting deficits, while overpricing premiums drives away healthy policyholders. Non-linear interactions between demographic attributes (e.g., age, BMI, smoking status) make traditional linear models insufficient.

This repository implements an **Actuarial Healthcare Cost Prediction Engine**. Using non-linear polynomial transformations, regularized regression techniques (Ridge, Lasso), and ensemble gradient boosting (XGBoost, Random Forest), it accurately projects annual healthcare expenses while isolating cost drivers.

<div align="center">
  <img src="https://raw.githubusercontent.com/sohailahmed-bsrh/myportfolio/main/assets/images/thumbnail/healthcare-cost-prediction.png" width="650" alt="Healthcare Cost Prediction Engine"/>
  <p><em>Actuarial risk regression engine modeling non-linear demographic and clinical cost drivers.</em></p>
</div>

---

## Real-World Industry Applications

- **Actuarial Premium Underwriting**: Computes risk-adjusted policy pricing based on medical risk profiles.
- **Corporate Health Benefit Planning**: Estimates future healthcare liability budgets for self-insured corporate employers.
- **Preventative Health Incentives**: Quantifies cost savings resulting from corporate wellness programs (e.g., smoking cessation, BMI reduction).
- **Healthcare Risk Stratification**: Identifies high-risk patient cohorts for proactive chronic disease case management.

---

## Technical Architecture & Engineering Walkthrough

```
                      [Actuarial Policyholder Dataset]
                                     │
                                     ▼
                   [Exploratory Data & Outlier Analysis]
                                     │
                                     ▼
                 [Feature Engineering & Interaction Modeling]
                   (BMI x Smoker, Age Polynomial Powers)
                                     │
         ┌───────────────────────────┼───────────────────────────┐
         ▼                           ▼                           ▼
  [Ridge / Lasso]             [Random Forest]               [XGBoost]
(L1/L2 Regularization)    (Non-Linear Tree Ensemble)   (Gradient Boosted Regressor)
         │                           │                           │
         └───────────────────────────┼───────────────────────────┘
                                     │
                                     ▼
                  [Residual Diagnostics & Homoscedasticity]
                                     │
                                     ▼
               [Evaluation Metrics: R-Squared, RMSE, MAE]
```

### Step 1: Exploratory Analysis & Skewness Correction
Examines medical charge distributions (`dataset/insurance.ipynb`), identifying significant right-skewness. Evaluates log transformations `log(charges)` to stabilize variance and satisfy homoscedasticity assumptions.

### Step 2: Non-Linear Interaction Engineering
Isolates critical interaction terms: while high BMI alone causes moderate cost inflation, the composite interaction `BMI x Smoker` triggers exponential cost escalation. Generates second-order polynomial features to capture compound clinical risks.

### Step 3: Regularized Linear Baselines (Ridge & Lasso)
Fits L1 (Lasso) and L2 (Ridge) regularized regression pipelines to penalize collinear features and prevent coefficient overfitting.

### Step 4: Non-Linear Ensemble Modeling
Trains Random Forest Regressors and Gradient Boosted Trees (XGBoost). Employs cross-validation with grid search to tune tree depth, learning rate, and minimum samples per leaf.

### Step 5: Residual Diagnostics & Performance Auditing
Evaluates model accuracy using $R^2$ Score, Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE). Generates residual quantile plots (Q-Q plots) to verify error normality.

---

## Technology Stack & Implementation Specifications

| Component | Technology | Version | Architectural Function |
| :--- | :--- | :--- | :--- |
| **Machine Learning** | Scikit-Learn | 1.3+ | Pipeline modeling, Ridge, Lasso, Random Forest, metrics |
| **Gradient Boosting** | XGBoost | 2.0+ | Non-linear regression tree boosting |
| **Data Manipulation** | Pandas / NumPy | 2.x | Tabular parsing, log transformations, dummy encoding |
| **Statistical Plotting** | Seaborn / Matplotlib | 3.8+ | Correlation heatmaps, residual distribution plots |

---

## Installation & Execution Guide

```bash
git clone https://github.com/sohailahmed-bsrh/medical-insurance-cost-prediction.git
cd medical-insurance-cost-prediction
python -m venv venv
# Windows: venv\Scriptsctivate | Unix: source venv/bin/activate
pip install -r requirements.txt
jupyter notebook dataset/insurance.ipynb
```

---

## Author & Contact Information

<table width="100%">
  <tr>
    <td width="150" align="center" valign="middle">
      <img src="https://avatars.githubusercontent.com/u/231165170?v=4" width="120" style="border-radius: 50%;" alt="Sohail Ahmed"/>
    </td>
    <td>
      <h3>Sohail Ahmed</h3>
      <p><strong>Senior Software Engineer & AI Systems Architect</strong></p>
      <p>Specializing in Full-Stack AI Application Development, Computer Vision, Deep Learning Pipelines, and Scalable Automation Architectures.</p>
      <p>
        <a href="https://sohail.devsil.com"><strong>Portfolio Website</strong></a> | 
        <a href="https://github.com/sohailahmed-bsrh"><strong>GitHub Profile</strong></a> | 
        <a href="https://www.linkedin.com/in/sohail-ahmed-546314316/"><strong>LinkedIn</strong></a> | 
        <a href="mailto:se906403@gmail.com"><strong>Email Contact</strong></a>
      </p>
    </td>
  </tr>
</table>
