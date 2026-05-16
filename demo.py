from shipment.logger import logging
from shipment.exception import CustomException
from shipment.utils.main_utils import MainUtils
from shipment.constants import DB_URL
from shipment.configuration.mongo_operations import MongoDBOperation

try:
    obj = MongoDBOperation()
    df=obj.get_collection_as_dataframe(db_name="ShipmentDatabase",collection_name="ShipmentCollection")
    print(df)
except Exception as e:
    logging.info(CustomException(e))
    raise CustomException(e)