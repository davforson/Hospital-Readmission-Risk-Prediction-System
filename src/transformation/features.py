import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def build_utilization_features(admissions: pd.DataFrame) -> pd.DataFrame:
    """Build features about how often each patient uses the hospital.

    For each admission, we look BACKWARDS in time to count prior visits.
    This avoids train/test leakage — we only use past data.
    """
    df = admissions.copy()
    df["admission_date"] = pd.to_datetime(df["admission_date"])
    df["discharge_date"] = pd.to_datetime(df["discharge_date"])

    # Sort by patient and date so we can look backwards
    df = df.sort_values(["patient_id", "admission_date"]).reset_index(drop=True)

    prior_admissions_6mo = []
    prior_admissions_12mo = []
    days_since_last = []
    total_previous_los = []

    for idx, row in df.iterrows():
        patient_history = df[
            (df["patient_id"] == row["patient_id"]) &
            (df["admission_date"] < row["admission_date"])
        ]

        # Admission reference dates in the last 6 and 12 months
        six_months_ago = row["admission_date"] - pd.Timedelta(days=180)
        twelve_months_ago = row["admission_date"] - pd.Timedelta(days=365)

        # Patients data in the last 6 and 12 months
        prior_6mo = patient_history[
                patient_history["admission_date"] >= six_months_ago
            ]
        prior_12mo = patient_history[
            patient_history["admission_date"] >= twelve_months_ago
        ]


        prior_admissions_6mo.append(len(prior_6mo))
        prior_admissions_12mo.append(len(prior_12mo))

        if len(patient_history) > 0:
            last_discharge = patient_history["discharge_date"].max()
            days_gap = (row["admission_date"] - last_discharge).days
            days_since_last.append(max(0, days_gap))
        else:
            days_since_last.append(0)

        # Total length of stay across all prior admissions
        total_previous_los.append(patient_history["length_of_stay"].sum())


    df["num_prior_admissions_6mo"] = prior_admissions_6mo
    df["num_prior_admissions_12mo"] = prior_admissions_12mo
    df["days_since_last_admission"] = days_since_last
    df["total_previous_los"] = total_previous_los

    logger.info(f"Built utilization features for {len(df)} admissions")
    return df

def build_lab_features(lab_results: pd.DataFrame, admissions: pd.DataFrame) -> pd.DataFrame: 
    """Build features for using lab results

    - For each admission, filter lab results where:
    patient_id matches AND test_date <= discharge_date
    - Calculate per admission:
        avg_glucose          → mean of glucose test values
        avg_creatinine       → mean of creatinine test values
        avg_hemoglobin       → mean of hemoglobin test values
        avg_white_blood_cell → mean of white blood cell test values
        num_lab_tests        → total number of lab tests
    - Merge these features back onto the admissions dataframe
    - Fill nulls with 0 (patient had no labs of that type)
    """
    # Step 1: Merge labs with admissions to get discharge_date alongside each lab
    labs_with_dates = pd.merge(
        lab_results,
        admissions[["admission_id", "patient_id", "discharge_date"]],
        on="patient_id",
        how="inner"
    )

    # Step 2: Filter — only keep labs from BEFORE discharge (prevent leakage)
    labs_with_dates = labs_with_dates[
        labs_with_dates["test_date"] <= labs_with_dates["discharge_date"]
    ]

    # Step 3: Pivot — get average of each test type PER ADMISSION
    lab_avg = labs_with_dates.groupby(
        ["admission_id", "test_name"]
    )["test_value"].mean().reset_index()

    lab_pivot = lab_avg.pivot_table(
        index="admission_id",
        columns="test_name",
        values="test_value"
    ).reset_index()

    lab_pivot.columns = [
        f"avg_{col}" if col != "admission_id" else col
        for col in lab_pivot.columns
    ]

    # Step 4: Count total lab tests per admission
    lab_counts = labs_with_dates.groupby("admission_id").size().reset_index(name="num_lab_tests")
    lab_pivot = pd.merge(lab_pivot, lab_counts, on="admission_id", how="left")

    # Step 5: Merge back onto admissions
    df = pd.merge(admissions, lab_pivot, on="admission_id", how="left")

    # Step 6: Fill nulls with 0 (patient had no labs of that type)
    lab_feature_cols = [col for col in df.columns if col.startswith("avg_") or col == "num_lab_tests"]
    df[lab_feature_cols] = df[lab_feature_cols].fillna(0)

    logger.info(f"Built lab features for {len(df)} admissions")
    return df


def build_demographic_features(patients: pd.DataFrame, admissions: pd.DataFrame) -> pd.DataFrame:
    """
    - Calculate age_at_admission:
    (admission_date - date_of_birth).days / 365.25, rounded to integer
    - One-hot encode: gender, insurance_type
        Use pd.get_dummies with a prefix
    - Merge demographic features onto admissions by patient_id
    - Drop the original string columns after encoding
    """
    demo_df = pd.merge(
        patients,
        admissions[['patient_id','admission_id','admission_date']],
        on = "patient_id",
        how = "left"
    )

    demo_df["age_at_admission"] = ((demo_df["admission_date"] - demo_df["date_of_birth"]).dt.days/365.25).astype(int)

    demo_df = pd.get_dummies(data= demo_df, 
                              columns=['gender','insurance_type'], 
                              prefix=['gender','insurance'],
                              drop_first = True) # To prevent multicollinearity or dummy variable trap
    
    feature_cols = ['admission_id','age_at_admission'] + [
        col for col in demo_df.columns if col.startswith('gender_') or col.startswith('insurance_')
    ]

    df = pd.merge(admissions, demo_df[feature_cols], on="admission_id", how="left")

    logger.info(f"Built demographic features for {len(df)} admissions")
    return df 

def build_all_features(patients: pd.DataFrame, admissions: pd.DataFrame, lab_results: pd.DataFrame) -> pd.DataFrame:
    """ Build all features

    - Call build_utilization_features(admissions) → get admissions with utilization features
    - Call build_lab_features(lab_results, admissions_with_util) → add lab features
    - Call build_demographic_features(patients, admissions_with_lab) → add demographic features
    - One-hot encode: admission_type, discharge_disposition
    - Drop non-feature columns: patient_id, first_name, last_name,
        admission_id, admission_date, discharge_date,
        primary_diagnosis_desc, primary_care_physician,
        date_of_birth, zip_code, race
    - Keep: readmitted_30d (target variable) and primary_diagnosis_code
    - Save to data/processed/features.parquet
    - Return the final dataframe
    - Log the final shape: "{rows} rows, {columns} features"
    
    """
    df = build_utilization_features(admissions)
    df = build_lab_features(lab_results, df)
    df = build_demographic_features(patients, df)

    
    df = pd.get_dummies(data= df,
                        columns = ['admission_type', 'discharge_disposition'],
                        prefix = ['admission', 'discharge'],
                        drop_first=True)
    
    cols_to_drop = ["patient_id", "first_name", "last_name",
        "admission_id", "admission_date", "discharge_date",
        "primary_diagnosis_desc", "primary_care_physician",
        "date_of_birth", "zip_code", "race"]
    
    df = df.drop(
        columns = [col for col in cols_to_drop if col in df.columns]
    )

    df.to_parquet("data/processed/features.parquet", index = False)

    logger.info(f"All features data has {len(df)} rows and {df.shape[1]} features.")
    return df


    








