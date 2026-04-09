from src.extraction.api_extractor import APIExtractor
from src.extraction.csv_extractor import CSVExtractor
from src.extraction.db_extractor import DatabaseExtractor
import logging

logger = logging.getLogger(__name__)

api_extractor = APIExtractor()
csv_extractor = CSVExtractor()
database_extractor = DatabaseExtractor()

patients_records = database_extractor.extract_patients()
admissions_data = database_extractor.extract_admissions()
lab_results = csv_extractor.extract_lab_results()
medication_records = api_extractor.extract_medications()

patients_records.to_parquet("/data/raw/patients_records.parquet")
admissions_data.to_parquet("/data/raw/admissions_data.parquet")
lab_results.to_parquet("/data/raw/lab_results.parquet")
medication_records.to_parquet("/data/raw/medication_records.parquet")

logger.info("Successfully extracted all data to raw directory")


