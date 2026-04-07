# 🏥 Hospital Readmission Prediction System

## 📌 Project Overview

This project simulates a **production-grade machine learning system** designed to predict **30-day hospital readmissions**. The goal is not just to build a model, but to design an **end-to-end system** that integrates data ingestion, validation, training, deployment, and monitoring — just like in real healthcare organizations.

---

## 💼 The Business Problem

In real-world healthcare settings:

> **Hospital CFO:**  
> "We lost $4.2 million in CMS penalties last year because our 30-day readmission rate is 18%. The national average is 15%."

> **VP of Analytics:**  
> "We need a model that flags high-risk patients BEFORE discharge so the care team can intervene."

### 🎯 Objective
Build a system that:
- Predicts patient readmission risk **before discharge**
- Enables clinicians to take preventive action
- Reduces costly penalties from CMS

---

## 🧠 What is a 30-Day Readmission?

A **30-day readmission** occurs when:
- A patient is discharged from the hospital
- The same patient is readmitted within **30 days**

### ⚠️ Why it matters
- The **Centers for Medicare & Medicaid Services (CMS)** penalize hospitals with high readmission rates
- Penalties can reach **up to 3% of total Medicare reimbursements**
- This translates to **millions of dollars annually**

---

## 🎯 Target Variable

```text
readmitted_30d:
    1 → Patient readmitted within 30 days
    0 → Patient NOT readmitted

## 🛠️ Tools & Technologies

| Tool | Purpose |
|------|---------|
| PostgreSQL | Patient data storage |
| pandas | Data manipulation and feature engineering |
| Great Expectations | Data quality validation |
| Apache Airflow | Pipeline orchestration |
| PyTorch | Model training |
| MLflow | Experiment tracking and model registry |
| FastAPI | Model serving via REST API |
| Docker | Containerization |
| Streamlit | Monitoring dashboard |
| GitHub Actions | CI/CD automation |

## 🚀 Setup Instructions

1. Clone the repository:
```bash
   git clone https://github.com/davforson/Hospital-Readmission-Risk-Prediction-System.git
   cd Hospital-Readmission-Risk-Prediction-System
```

2. Create and activate virtual environment:
```bash
   python -m venv venv
   source venv/bin/activate  # macOS/Linux
```

3. Install dependencies:
```bash
   pip install -r requirements.txt
```

4. Set up environment variables:
```bash
   cp .env.example .env
   # Edit .env with your actual credentials
```
