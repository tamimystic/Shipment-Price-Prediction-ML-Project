from shipment.logger import logging
from shipment.exception import CustomException
import sys

try:
    t=1/0
except Exception as e:
    logging.info(CustomException(e))
    raise CustomException(e)