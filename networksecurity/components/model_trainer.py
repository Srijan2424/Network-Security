import os,sys

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

from networksecurity.entity.artifact_entity import DataTransformationArtifact,ModelTrainerArtifact
from networksecurity.entity.config_entity import ModelTrinerConfig

from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.utils.main_utils.utils import save_object,load_object,evaluate_models
from networksecurity.utils.main_utils.utils import load_numpy_array_data
from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_score

# all the models being used
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier
)
import mlflow
from urllib.parse import urlparse

import dagshub
import dagshub
dagshub.init(repo_owner='Srijan2424', repo_name='Network-Security', mlflow=True)

os.environ["MLFLOW_TRACKING_URI"]="https://dagshub.com/Srijan2424/Network-Security.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"]="Srijan2424"
os.environ["MLFLOW_TRACKING_PASSWORD"]="f2226a1dc4f9d988420482e2cb758ceb1d2cfd10"




class ModelTrainer:
    def __init__(self,model_trainer_config:ModelTrinerConfig,data_transformation_artifact:DataTransformationArtifact):
        try:
            self.model_trainer_config=model_trainer_config
            self.data_transformation_config=data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e,sys)
    
    def track_mlflow(self,best_model,classificationmetric):
        with mlflow.start_run():
            mlflow.set_registry_uri("https://dagshub.com/Srijan2424/Network-Security.mlflow")
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            
            f1_score=classificationmetric.f1_score
            precision_score=classificationmetric.precision_score
            recall_score=classificationmetric.recall_score
            
            mlflow.log_metric("f1_score",f1_score)
            mlflow.log_metric("precision_score",precision_score)
            mlflow.log_metric("recall_score",recall_score)

            if tracking_url_type_store != "file":

                # Register the model
                # There are other ways to use the Model Registry, which depends on the use case,
                # please refer to the doc for more information:
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                model_name = type(best_model).__name__  # e.g., 'RandomForestClassifier'
                # or any other string you want, like "network-security-best-model"

                mlflow.sklearn.log_model(
                    sk_model=best_model,
                    artifact_path="model",
                    registered_model_name=model_name,   # <-- must be str, not the model object
                )
            else:
                mlflow.sklearn.log_model(best_model, "model")
        
    def train_model(self,X_train,y_train,X_test,y_test):
        # initialise all the models
        models = {
            "Random Forest":RandomForestClassifier(),
            "Decision Tree":DecisionTreeClassifier(),
            "Gradient Boosting":GradientBoostingClassifier(),
            "Logistic Regression":LogisticRegression(),
            "AdaBoost":AdaBoostClassifier()
        }
        
        # preparation of all the hyperparameters
        params={
            "Decision Tree": {
                'criterion':['gini', 'entropy', 'log_loss'],
                # 'splitter':['best','random'],
                # 'max_features':['sqrt','log2'],
            },
            "Random Forest":{
                # 'criterion':['gini', 'entropy', 'log_loss'],
                
                # 'max_features':['sqrt','log2',None],
                'n_estimators': [8,32,128,256]
            },
            "Gradient Boosting":{
                # 'loss':['log_loss', 'exponential'],
                'learning_rate':[.1,.01,.001],
                'subsample':[0.6,0.7,0.9],
                # 'criterion':['squared_error', 'friedman_mse'],
                # 'max_features':['auto','sqrt','log2'],
                'n_estimators': [8,16,128,256]
            },
            "Logistic Regression":{},
            "AdaBoost":{
                'learning_rate':[.1,.01,.001],
                'n_estimators': [8,64,128,256]
            }
            
        }
        # this will give the best possible model score for all the models 
        model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,params=params)
        
        # get the best model score from dict
        best_model_score=max(sorted(model_report.values()))
        
        best_model_name=list(model_report.keys())[
            list(model_report.values()).index(best_model_score)
        ]
        
        best_model=models[best_model_name]
        y_train_pred=best_model.predict(X_train)
        
        classification_train_metric=get_classification_score(y_true=y_train,y_pred=y_train_pred)
        
        # Track the mlflow
        # Tracks the life-cycle of the ml model 
        # For Train metrics
        self.track_mlflow(best_model,classification_train_metric)
        
        
        # for test data
        y_test_pred=best_model.predict(X_test)
        
        classification_test_metric=get_classification_score(y_true=y_test,y_pred=y_test_pred)   
        
        # Track the mlflow
        # Tracks the life-cycle of the ml model 
        # For Test metrics
        self.track_mlflow(best_model,classification_test_metric)        
        
        # making / loading the pickle file 
        preprocessor=  load_object(file_path=self.data_transformation_config.transformed_object_file_path)
        
        model_dir_path=os.path.dirname(self.model_trainer_config.trained_model_file_path)
        os.makedirs(model_dir_path,exist_ok=True) 
         
        # training the new enetred data by the user 
        Network_Model=NetworkModel(preprocessor=preprocessor,model=best_model)
        save_object(self.model_trainer_config.trained_model_file_path,obj=NetworkModel)
        
        
        # Model Trainer Artifact 
        model_trainer_artifact=ModelTrainerArtifact(trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                             train_metric_artifact=classification_train_metric,
                             test_metric_artifact=classification_test_metric
                             ) 
        
        logging.info(f"Model trainer artifact: {model_trainer_artifact}")
        return model_trainer_artifact
        
    def initiate_model_trainer(self)->ModelTrainerArtifact:
        try:
            train_file_path=self.data_transformation_config.transformed_train_file_path
            test_file_path=self.data_transformation_config.transformed_test_file_path
            
            # loading training array and testing array 
            train_arr=load_numpy_array_data(train_file_path)
            test_arr=load_numpy_array_data(test_file_path)
            
            X_train,y_train,X_test,y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            
            model=self.train_model(X_train,y_train,X_test,y_test)
            
        except Exception as e:
            raise NetworkSecurityException(e,sys)