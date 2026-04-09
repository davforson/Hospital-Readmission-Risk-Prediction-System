import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class CSVExtractor():
    """
    Extract csv files from sources systems.
    """
    def __init__(self, data: str = "data"):
        self.data = Path(data)

    def extract_lab_results(self) -> pd.DataFrame :
        """
        Extract lab results from source system
        """
        filepath = self.data / "lab_results.csv"

        if not filepath.exists():
            raise FileNotFoundError(f"Lab results not found at {filepath}")
        
        df = pd.read_csv(
            filepath,
            encoding = 'utf-8-sig',
            on_bad_lines= 'warn',
            dtype = {
                'patient_id': str,
                'test_name': str
            },
            parse_dates= ['test_date']
        )

        logger.info(f"Total lab records: {len(df)}")
        return df