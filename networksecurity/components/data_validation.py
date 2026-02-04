from networksecurity.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact
from networksecurity.entity.config_entity import DataValidationConfig
from networksecurity.logging.logger import logging
from networksecurity.exception.exception import NetworkSecurityException
from scipy.stats import ks_2samp
from networksecurity.constant.training_pipeline import SCHEMA_FILE_PATH
import pandas as pd
import numpy as np 
import os
import sys
from networksecurity.utils.main_utils.utils import read_yaml_file,write_yaml_file

#  define a schema so that we can validate the number of columns in the dataset
# read the schema in the schema.yml and validate the data 

class DataValidation:
    def __init__(self,data_ingestion_artifact:DataIngestionArtifact,
                 data_validation_config:DataValidationConfig):
        try:
            self.data_ingestion_artifact=data_ingestion_artifact
            self.data_validation=data_validation_config
            self._schema_config=read_yaml_file(SCHEMA_FILE_PATH)
            
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    @staticmethod
    def read_data(file_path)->pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    def validate_number_of_columns(self,dataframe:pd.DataFrame)->bool:
        try:
            number_of_columns=len(self._schema_config)
            logging.info(f"Required number of columns:{number_of_columns}")
            logging.info(f"Dataframe has columns: {len(dataframe.columns)}")
            if len(dataframe.columns)== number_of_columns:
                return True
            else :
                return False
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    def validate_number_of_numerical_columns(self,dataframe:pd.DataFrame)->bool:
        try:
            numerical_columns = [key for key, value in self._schema_config.items() if isinstance(value, (int, np.int64))]
            number_of_numerical_columns=len(numerical_columns)
            logging.info(f"Required number of numerical columns:{number_of_numerical_columns}")
            logging.info(f"Dataframe has these number of numerical columns: {len(dataframe.columns)}")
            if len(dataframe.columns)== number_of_numerical_columns:
                return True
            else :
                return False
        except Exception as e:
            raise NetworkSecurityException(e,sys)
    
    def detect_dataset_drift(self,base_df,current_df,threshold=0.05)->bool:
        try:
            status=True
            report={}
            for column in base_df.columns:
                d1=base_df[column]
                d2=current_df[column]
                is_same_dist=ks_2samp(d1,d2)
                if threshold<=is_same_dist.pvalue:
                    is_found=False
                else:
                    is_found=True
                    status=False
                report.update({column:{
                    "p_value":float(is_same_dist.pvalue),
                    "drift_status":is_found
                }})
                
            drift_report_file_path= self.data_validation.drift_report_file_path
            #  Create directory
            dir_path=os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path,exist_ok=True)
            write_yaml_file(file_path=drift_report_file_path,content=report)
            
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    def initiate_data_validation(self)->DataValidationArtifact:
        try:
            train_file_path=self.data_ingestion_artifact.trained_file_path
            test_file_path=self.data_ingestion_artifact.test_file_path
            
            # read the data from train and test
            train_dataframe=DataValidation.read_data(train_file_path)
            test_dataframe=DataValidation.read_data(test_file_path)
            
            # validate number of columns
            # for training dataset
            # total columns
            status=self.validate_number_of_columns(dataframe=train_dataframe)
            if not status:
                error_message=f"Train message does not have all columns \n"
            # numerical columns 
            status=self.validate_number_of_numerical_columns(dataframe=train_dataframe)
            if not status:
                error_message=f"Train message does not have all the numerical columns \n"
            
            # for test dataset
            # total columns 
            status=self.validate_number_of_columns(dataframe=test_dataframe)
            if not status:
                error_message=f"Test message does not have all columns \n"
            # numerical columns
            status=self.validate_number_of_numerical_columns(dataframe=test_dataframe)
            if not status:
                error_message=f"Test message does not have all the numerical columns \n"
            
            # lets check data_drift
            status=self.detect_dataset_drift(base_df=train_dataframe,current_df=test_dataframe)
            dir_path=os.path.dirname(self.data_validation.valid_train_file_path)
            os.makedirs(dir_path,exist_ok=True)
            
            train_dataframe.to_csv(
                self.data_validation.valid_train_file_path,index=False,header=True
            )
                        
            test_dataframe.to_csv(
                self.data_validation.valid_test_file_path,index=False,header=True
            )
            
            data_validation_artifact=DataValidationArtifact(
                validation_status=status,
                valid_train_file_path=self.data_ingestion_artifact.trained_file_path,
                valid_test_file_path=self.data_ingestion_artifact.test_file_path,
                invalid_train_file_path=None,
                invalid_test_file_path=None,drift_report_file_path=self.data_validation.drift_report_file_path
            )
            return data_validation_artifact
        except Exception as e:
            raise NetworkSecurityException(e,sys)
