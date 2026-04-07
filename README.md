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
