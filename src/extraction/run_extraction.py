from src.extraction.api_extractor import APIExtractor
from src.extraction.csv_extractor import CSVExtractor
from src.extraction.db_extractor import DatabaseExtractor
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_full_extraction():
    "Run all extractions to get data"
    api_extractor = APIExtractor()
    csv_extractor = CSVExtractor()
    database_extractor = DatabaseExtractor()

    patients_records = database_extractor.extract_patients()
    admissions_data = database_extractor.extract_admissions()
    lab_results = csv_extractor.extract_lab_results()
    medication_records = api_extractor.extract_medications()

    patients_records.to_parquet("data/raw/patients.parquet", index=False)
    admissions_data.to_parquet("data/raw/admissions.parquet", index=False)
    lab_results.to_parquet("data/raw/lab_results.parquet", index=False)
    medication_records.to_parquet("data/raw/medication.parquet", index=False)

    logger.info("Successfully extracted all data to raw directory")


if __name__ == "__main__":
    run_full_extraction()
