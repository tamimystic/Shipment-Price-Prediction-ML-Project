from dataclasses import dataclass

# Data Ingestion from MongoDB related ArtIfact Directory
@dataclass
class DataIngestionArtifacts:
    train_data_file_path: str
    test_data_file_path: str

# Data Validation related ArtIfact Directory
@dataclass
class DataValidationArtifacts:
    data_drift_file_path: str
    validation_status: bool

# Data Transformation related ArtIfact Directory
@dataclass
class DataTransformationArtifacts:
    transformed_object_file_path: str
    transformed_train_file_path: str
    transformed_test_file_path: str