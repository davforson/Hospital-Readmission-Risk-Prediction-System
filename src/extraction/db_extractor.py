from sqlalchemy import create_engine, text
import pandas as pd
import logging
from dotenv import load_dotenv
import os

load_dotenv()
logger = logging.getLogger(__name__)


class DatabaseExtractor():
    """
    Conect to the database using a connection uri, connection pool and chunk reading
    """
    def __init__(self):
        self.engine = create_engine(
            f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}"
            f"@{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DB')}"
            ,
            pool_size=5,
            max_overflow=10,
            pool_timeout=30,
            pool_recycle=1800
        )


    def extract_patients(self, chunksize: int = 10000) -> pd.DataFrame:
        """
        Extract patient data from database in chunks of 10,000
        """
        query = text("""
            SELECT 
                patient_id,
                first_name,
                last_name,
                date_of_birth,
                gender,
                race,
                zip_code,
                insurance_type,
                primary_care_physician
              FROM patients 
        """)


        chunks = []
        with self.engine.connect() as conn:
            for chunk in pd.read_sql(query, conn, chunksize=chunksize):
                chunks.append(chunk)
                logger.info(f"Extracted {len(chunk)} patient records")

        
        df = pd.concat(chunks, ignore_index = True)
        logger.info(f"Total patient records extracted: {len(df)}")
        return df
    
    def extract_admissions(self, chunksize: int = 10000) -> pd.DataFrame:
        """
        Extracts admission info from database in chunks of 10000
        """
        query = text("""
                     SELECT 
                       admission_id, 
                       patient_id, 
                       admission_date, 
                       discharge_date, 
                       admission_type, 
                       discharge_disposition, 
                       primary_diagnosis_code, 
                       primary_diagnosis_desc, 
                       number_of_procedures, 
                       number_of_diagnoses, 
                       length_of_stay, 
                       readmitted_30d
                    FROM admissions
                     """)
        
        chunks = []
        with self.engine.connect() as conn:
            for chunk in pd.read_sql(query, conn, chunksize=chunksize):
                chunks.append(chunk)
                logger.info(f"Extracted {len(chunk)} admission records")

        
        df = pd.concat(chunks, ignore_index = True)
        logger.info(f"Total admisssion records extracted: {len(df)}")
        return df
