import requests
from dotenv import load_dotenv
import os
import logging
from time import sleep
import pandas as pd

load_dotenv()
logger = logging.getLogger(__name__)

class APIExtractor():
    """
    Extract data from a website using REST API.
    """
    def __init__(self):
        self.base_url = os.getenv("MEDS_API_URL")
        self.headers = {
            "Authorization": f"Bearer {os.getenv('MEDS_API_KEY')}",
            "Content-Type": "application/json"
        }
        self.max_retries = 3
        self.retry_delay = 2

    def _make_request(self, endpoint: str, params: dict) -> dict:
        """
        Make requesting addressing rate limitting and exponential backoff
        """
        for attempt in range(self.max_retries):
            try:
                response = requests.get(
                    f"{self.base_url}/{endpoint}/",
                    headers = self.headers,
                    params = params,
                    timeout = 30
                )

                # Rate limitting
                if response.status_code == 429: #making too many requests
                    wait = int(response.headers.get('Retry-after', 60))
                    logger.warning(f"HTTP issures. Retrying after {wait} seconds")
                    sleep(wait)
                    continue

                # raise any HTTP expception
                response.raise_for_status()
                return response.json()
            
            # Exponential Backoff
            except requests.exceptions.RequestException as e:
                wait = self.retry_delay * (2 ** attempt)
                logger.error(f"Network/server issues: {e}. Attempt {attempt + 1}. Retrying after {wait} seconds")
                sleep(wait)

        # Final exception if nothing work
        raise Exception(f"Endpoint {endpoint} could not be reached after {self.max_retries} attempts")
    

    def extract_medications(self)-> pd.DataFrame: 
        """
        Extracting medication (using pagination logic)
        """
        all_records = []
        page = 1
        per_page = 100

        while True: 
            response = self._make_request("medications", {
                'page': page,
                'per_page': per_page
            })

            records = response.get('results', [])
            if not records:
                break

            all_records.extend(records)
            logger.info(f"Extracted {len(records)} medication records")

            if page >= response.get("total_pages", 1):
                break
            page += 1
            sleep(0.5)

        df = pd.DataFrame(all_records)
        logger.info(f"Extracted {len(df)} medication records")
        return df 


