import logging

def setup_logging():
    logging.basicConfig(
        filename='log_history.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
