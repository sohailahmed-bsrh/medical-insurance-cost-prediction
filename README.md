# ⚕️ Healthcare Cost Prediction & Patient Risk Segmentation

An actuarial healthcare expenditure and predictive risk stratification web application built with Python and Streamlit. This project analyzes 1,338 patient records to evaluate lifestyle cost multipliers (smoking, BMI, age), benchmark multiple machine learning regression algorithms, and cluster patients into actuarial risk tiers using K-Means clustering.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🌟 Key Features

- **Expenditure Analytics Dashboard:** Tracks cohort size, mean annual premiums, median age, and lifestyle risk factors.
- **The Smoker-BMI Multiplier:** Interactive visualizations illustrating how tobacco usage exponentially multiplies healthcare liabilities in patients with BMI > 30.
- **Unsupervised Risk Segmentation (K-Means):** Groups patients into 3 distinct actuarial tiers (Low Risk, Moderate Risk, High Actuarial Risk) using combined Age, BMI, and Expenditure interactions.
- **Live Actuarial Premium Estimator:** Real-time Gradient Boosting regression simulator forecasting annual premiums and calculating exact dollar surcharges attributable to tobacco use.
- **Model Benchmark Suite:** Rigorous cross-model evaluation comparing Linear Regression, Ridge, Lasso, Random Forest, and Gradient Boosting ($R^2$, MAE, RMSE).

---

## 📊 Analytical Insights

1. **Tobacco Penalty:** Smokers pay an average premium nearly **4x higher** than comparable non-smokers.
2. **Non-Linear Compounding:** Gradient Boosting achieves the superior $R^2 \approx 0.87$, successfully capturing the non-linear interaction between elevated BMI and smoking status that standard linear models miss.

---

## 💻 Local Setup & Execution

1. **Clone the repository:**
   ```bash
   git clone https://github.com/sohailahmed-bsrh/medical-insurance-cost-prediction.git
   cd medical-insurance-cost-prediction
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

---

## 🚀 Deployment on Streamlit Community Cloud

1. Fork or push this repository to your GitHub account (`https://github.com/sohailahmed-bsrh/medical-insurance-cost-prediction`).
2. Visit [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
3. Click **New app**.
4. Select repository: `sohailahmed-bsrh/medical-insurance-cost-prediction`
5. Branch: `main`
6. Main file path: `app.py`
7. Click **Deploy!**

---

## 👤 Author

**Sohail Ahmed**
- **GitHub:** [@sohailahmed-bsrh](https://github.com/sohailahmed-bsrh)
- **Portfolio:** [https://sohail.devsil.com](https://sohail.devsil.com)
