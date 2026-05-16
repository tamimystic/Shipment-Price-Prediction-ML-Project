import os
from os import environ
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

MODEL_CONFIG_FILE= "config/model.yaml"
MODEL_FILE_PATH = "config/schema.yaml"

DB_URL = environ["CONNECTION_URI"]