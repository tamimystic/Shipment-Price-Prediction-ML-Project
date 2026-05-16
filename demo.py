from shipment.logger import logging
from shipment.exception import CustomException
from shipment.utils.main_utils import MainUtils

try:
    ob = MainUtils()

    data = ob.read_yaml_file("config/schema.yaml")

    print(data)
except Exception as e:
    logging.info(CustomException(e))
    raise CustomException(e)