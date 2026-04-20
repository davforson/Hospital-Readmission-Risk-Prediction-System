import great_expectations as gx
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def validate_patients(df: pd.DataFrame) -> dict:
    """Validate patient data against business rules.

    Rules:
    - patient_id: unique, never null (primary key integrity)
    - gender: only known values (data consistency)
    - insurance_type: only known values (critical for billing logic)
    - date_of_birth: never null, in the past (no future dates)
    - Other demographic fields: allow up to 5% nulls (realistic)

    Returns dict with 'success' (bool) and 'results' (details).
    """
    # Create a gx context
    context = gx.get_context()

    # Connect our data: both source and asset
    data_source = context.data_sources.add_pandas("patients_source")
    data_asset = data_source.add_dataframe_asset("patients_asset")

    # Create a batch_definition
    batch_def = data_asset.add_batch_definition_whole_dataframe("patients_batch")

    # Create a suite of expectation
    suite = context.suites.add(gx.ExpectationSuite(name="patient_validation_suite"))

    # Add expectations to the suite
    # Primary key must never be Null and should be unique
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeUnique(column="patient_id"))
    suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="patient_id"))

    # Gender should have only known values: Male, Female and Other
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="gender", value_set=["Male", "Female", "Other"]
        )
    )

    # Insurance_type: only known values: Medicare, Medicaid, Private, Self-Pay, VA
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="insurance_type", value_set=["Medicare", "Medicaid", "Private", "Self-Pay", "VA"]
        )
    )

    # Date_of_birth: never null, in the past (no future dates)
    suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="date_of_birth"))

    suite.add_expectation(
        gx.expectations.ExpectColumnMaxToBeBetween(
            column="date_of_birth", max_value=pd.Timestamp.now().isoformat()
        )
    )

    # Demographic details: zip_code - allow up to 5% null
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToNotBeNull(column="zip_code", mostly=0.95)
    )

    # Demographic details: race - allow up to 5% null
    suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="race", mostly=0.95))

    # Create a validation definition
    validation_def = context.validation_definitions.add(
        gx.ValidationDefinition(name="Validate_patients", data=batch_def, suite=suite)
    )

    # Run the validation defintion to get results
    result = validation_def.run(batch_parameters={"dataframe": df})

    # Check for the success of the results/ Log results
    if result.success:
        logger.info("Patient data validation passed")
    else:
        logger.error("Patient data validaton failed")
        for exp_result in result.results:
            if not exp_result.success:
                logger.error(f"FAILED: {exp_result.expectation_config}")

    return {"success": result.success, "results": result.results}


def validate_admissions(df: pd.DataFrame) -> dict:
    """Validate the results for admissions
    admission_id    → unique, not null
    patient_id      → not null
    admission_date  → not null
    discharge_date  → not null
    admission_type  → only: Emergency, Urgent, Elective, Trauma
    discharge_disposition → only: Home, SNF, Rehab, Home Health, AMA, Expired
    length_of_stay  → not null, minimum value 1 (nobody stays 0 days)
    readmitted_30d  → not null (this is our target variable — must be clean)

    """
    context = gx.get_context()

    data_source = context.data_sources.add_pandas("admissions_source")
    data_asset = data_source.add_dataframe_asset("admissions_asset")
    batch_def = data_asset.add_batch_definition_whole_dataframe("admissions_batch")

    suite = context.suites.add(gx.ExpectationSuite(name="admissions_validation_suite"))

    # admission_id    → unique, not null
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeUnique(column="admission_id"))

    suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="admission_id"))

    # patient_id      → not null
    suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="patient_id"))

    # admission_date  → not null
    suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="admission_date"))

    # discharge_date  → not null
    suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="discharge_date"))

    # admission_type  → only: Emergency, Urgent, Elective, Trauma
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="admission_type", value_set=["Emergency", "Urgent", "Elective", "Trauma"]
        )
    )

    # discharge_disposition → only: Home, SNF, Rehab, Home Health, AMA, Expired
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="discharge_disposition",
            value_set=["Home", "SNF", "Rehab", "Home Health", "AMA", "Expired"],
        )
    )

    # length_of_stay  → not null, minimum value 1 (nobody stays 0 days)
    suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="length_of_stay"))

    suite.add_expectation(
        gx.expectations.ExpectColumnMinToBeBetween(column="length_of_stay", min_value=1)
    )

    # readmitted_30d  → not null (this is our target variable — must be clean)
    suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="readmitted_30d"))

    validation_def = context.validation_definitions.add(
        gx.ValidationDefinition(name="validate_admissions", data=batch_def, suite=suite)
    )

    result = validation_def.run(batch_parameters={"dataframe": df})

    if result.success:
        logger.info("Validate all admission succesfully")
    else:
        logger.error("FAILED: Could not validate admissions")
        for exp_result in result.results:
            if not exp_result.success:
                logger.error(f" FAILED: {exp_result.expectation_config}")

    return {"success": result.success, "results": result.results}


def validate_lab_results(df: pd.DataFrame) -> dict:
    """Validate lab results

    patient_id      → not null
    test_name       → not null, only: glucose, creatinine, hemoglobin,
                    white_blood_cell, sodium, potassium, bun, hba1c
    test_value      → not null, minimum value 0 (no negative lab values)
    test_date       → not null
    test_unit       → allow up to 5% nulls (sometimes units aren't recorded)
    """
    context = gx.get_context()

    data_source = context.data_sources.add_pandas("lab_results_source")
    data_asset = data_source.add_dataframe_asset("lab_results_asset")
    batch_def = data_asset.add_batch_definition_whole_dataframe("lab_results_batch")

    suite = context.suites.add(gx.ExpectationSuite(name="Lab_results_validation_suite"))

    suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="patient_id"))

    suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="test_name"))

    # glucose, creatinine, hemoglobin,white_blood_cell, sodium, potassium, bun, hba1c
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="test_name",
            value_set=[
                "glucose",
                "creatinine",
                "hemoglobin",
                "white_blood_cell",
                "sodium",
                "potassium",
                "bun",
                "hba1c",
            ],
        )
    )

    # test_value      → not null, minimum value 0 (no negative lab values)
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(column="test_value", min_value=0, mostly=0.99)
    )

    # test_date       → not null
    suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="test_value"))

    # test_date       → not null
    suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="test_date"))

    # test_unit       → allow up to 5% nulls (sometimes units aren't recorded)
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToNotBeNull(column="test_unit", mostly=0.95)
    )

    validation_def = context.validation_definitions.add(
        gx.ValidationDefinition(name="validate_lab_results", data=batch_def, suite=suite)
    )

    result = validation_def.run(batch_parameters={"dataframe": df})

    if result.success:
        logger.info("Successfully validate all lab results")
    else:
        logger.error("Failed to validate all lab results")
        for exp_result in result.results:
            if not exp_result.success:
                logger.error(f"FAILED: {exp_result.expectation_config}")

    return {"success": result.success, "results": result.results}
