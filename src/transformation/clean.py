import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def clean_patients(df: pd.DataFrame) -> pd.DataFrame:
    """Clean patient demographics.

    Every decision is justified:
    - Duplicates: keep first (shouldn't exist, but defensive)
    - date_of_birth: drop rows with null (can't calculate age without it)
    - race: fill 'unknown' (optional demographic, not a model feature)
    - zip_code: fill 'unknown' (same reasoning)
    """
    df = df.copy() 
    before = len(df)

    # Remove duplicates on the primary key
    df = df.drop_duplicates(subset=['patient_id'], keep="first")
    logger.info(f"Removed {before - len(df)} duplicate patients")

    # Date of birth: drop nulls(need this for age calculation)
    null_dob = df["date_of_birth"].isna().sum()
    if null_dob > 0:
        df = df.dropna(subset=["date_of_birth"])
        logger.info(f"Dropped {null_dob} patients will null date_of_birth")

    # Categorical columns: fill with unknown
    for col in ["race","zip_code"]:
        null_count = df[col].isna().sum()
        if null_count > 0:
            df[col] = df[col].fillna("unknown")
            logger.info("Filled {null_count} rows in {col} with 'unknown'.")

    # Standardize string columns
    for col in ["gender", "race", "insurance_type"]:
        df[col] = df[col].str.strip()

    logger.info(f"Clean patients: {len(df)} rows.")
    return df

def clean_admissions(df: pd.DataFrame) -> pd.DataFrame:
    """ Clean admissions data
    
    - Copy the dataframe (never mutate input)
    - Remove duplicates on admission_id
    - Convert admission_date and discharge_date to datetime
    - Drop rows where admission_date or discharge_date is null
    (we need both to calculate length of stay)
    - Fix any rows where length_of_stay < 1 by recalculating:
    length_of_stay = (discharge_date - admission_date).days
    Then set any that are still < 1 to 1 (minimum 1 day stay)
    - Standardize string columns: admission_type, discharge_disposition
    (strip whitespace)
    """
    df = df.copy()
    before = len(df)

    # Remove duplicate admission_ids
    df = df.drop_duplicates(subset=['admission_id'], keep="first")
    logger.info(f"Removed {before - len(df)} duplicate admission_ids.")

    # Convert admission and discharge date to datetime
    df['admission_date'] = pd.to_datetime(df['admission_date'])
    df['discharge_date'] = pd.to_datetime(df['discharge_date'])

    null_dates = df[["admission_date", "discharge_date"]].isna().any(axis=1).sum()
    if null_dates > 0:
        df= df.dropna(subset=['admission_date', 'discharge_date'])
        logger.info(f"Dropped {null_dates} rows for admission and discharge dates with null values.")

    df["length_of_stay"] = (df["discharge_date"] - df["admission_date"]).dt.days
    df.loc[df["length_of_stay"] < 1, "length_of_stay"] = 1

    for col in ["admission_type", "discharge_disposition"]:
        df[col] = df[col].str.strip()

    logger.info(f"Clean admissions: {len(df)} rows.")
    return df 


def clean_lab_results(df: pd.DataFrame) -> pd.DataFrame: 
    """Clean lab results

    - Copy the dataframe
    - Drop rows where patient_id is null (can't link to a patient)
    - Drop rows where test_value is null (a result without a value is useless)
    - Clip negative test_values to 0 (we found 133 negatives in Section 3)
    - Convert test_date to datetime
    - Standardize test_name to lowercase and strip whitespace
    
    """
    df = df.copy()
    before = len(df)

    null_patients = df["patient_id"].isna().sum()
    if null_patients > 0: 
        df = df.dropna(subset=['patient_id'])
        logger.info(f"Dropped {null_patients} rows with null patient_id")

   
    null_values = df["test_value"].isna().sum()
    if null_values > 0:
        df = df.dropna(subset=["test_value"])
        logger.info(f"Dropped {null_values} rows with null test_value")

    df["test_value"] = df["test_value"].clip(lower=0)

    df["test_date"] = pd.to_datetime(df["test_date"])

    df["test_name"] = df["test_name"].str.lower().str.strip()

    logger.info(f"Cleaned lab results: {len(df)} rows.")
    return df