import os 
import sys
import json

from dotenv import load_dotenv
load_dotenv()


MONGO_DB_URL=os.getenv("MONGO_DB_URL")
print(MONGO_DB_URL)

# this ensures that mongobd gets to know that the request is being made from a certified source 
import certifi
# this line retrieves the path to the bundle of ca certificates and stores it in the variable ca 
# done to certify that the request is being made from a certified source 
ca=certifi.where()

import pandas as pd 
import numpy as np
import pymongo
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

class NetworkDataExtract():
    def __init__(self):
        try:
            pass
        except Exception as e:
            raise NetworkSecurityException(e,sys)
    
    def cv_to_json_converter(self,file_path):
        try:
            data=pd.read_csv(file_path)
            data.reset_index(drop=True,inplace=True)
            records=list(json.loads(data.T.to_json()).values())
            return records
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    def insert_data_mongodb(self,records,database,collection):
        try:
            self.database=database
            self.collection=collection
            self.records=records
            
            self.mongo_client=pymongo.MongoClient(MONGO_DB_URL)
            self.database=self.mongo_client[self.database]
            
            self.collection=self.database[self.collection]
            self.collection.insert_many(self.records)
            
            return self.records
        except Exception as e:
            raise NetworkSecurityException(e,sys)
            
if __name__=="__main__":
   FILE_PATH="Network_Data/phisingData.csv"
   DATABASE="SRIJANAI"
   Collection='NetworkData'
   networkobj=NetworkDataExtract()
   record=networkobj.cv_to_json_converter(file_path=FILE_PATH)
   print(record)
   no_of_records=networkobj.insert_data_mongodb(record,DATABASE,Collection)
   print(f"\n{no_of_records}")
   
   
    