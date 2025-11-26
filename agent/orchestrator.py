import os
import time
import logging
import requests

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Get CORE_API_URL from environment variables, with a default value
CORE_API_URL = os.getenv('CORE_API_URL', 'http://apu_core:5002')
HEALTH_CHECK_ENDPOINT = f"{CORE_API_URL}/api/health"
CHECK_INTERVAL_SECONDS = 30

def check_core_health():
    """
    Checks the health of the core service by making a GET request to its health endpoint.
    """
    try:
        response = requests.get(HEALTH_CHECK_ENDPOINT, timeout=10)
        if response.status_code == 200:
            logging.info(f"Core service is healthy. Response: {response.json()}")
        else:
            logging.warning(f"Core service returned non-200 status: {response.status_code}")
    except requests.RequestException as e:
        logging.error(f"Failed to connect to core service: {e}")

def main():
    """
    Main loop to periodically check the core service's health.
    """
    logging.info(f"Agent orchestrator started. Monitoring core at {HEALTH_CHECK_ENDPOINT}")
    while True:
        check_core_health()
        time.sleep(CHECK_INTERVAL_SECONDS)

if __name__ == "__main__":
    main()
