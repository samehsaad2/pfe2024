from pathlib import Path
import pickle
import pandas as pd
import sys
from dataprocessing import DataCleaning
import mlflow
#from dotenv import load_dotenv
import os

#from dotenv import load_dotenv

#load_dotenv(".env")



#setup mlflow
os.environ['MLFLOW_TRACKING_USERNAME']= "rihemmanel54"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "3e93c19c879ea3562d4638daa2bab19d3eabb3c9"

#setup mlflow
mlflow.set_tracking_uri('https://dagshub.com/rihemmanel54/CarPricePrediction_mlFlow.mlflow')


#tests if the model works as expected

def test_model_use():

    df_mlflow = mlflow.search_runs(filter_string="metrics.Accuracy<1")
    run_id = df_mlflow.loc[df_mlflow['metrics.Accuracy'].idxmax()]['run_id']

    logged_model = f'runs:/{run_id}/ML_models'


    # Load model as a PyFuncModel.
    model = mlflow.pyfunc.load_model(logged_model)
    
    d= {'name':'Maruti Swift Dzire VDI','year':2014,'km_driven':145500,'fuel':'Diesel','seller_type':'Individual',
    'transmission':'Manual','owner':'First Owner','mileage':"23.4 kmpl",'engine':'1248 CC','max_power':'74 bhp',
    'torque':'190Nm@ 2000rpm','seats':5}
    df = pd.DataFrame(data=d,index=[0])
    dd = DataCleaning(df)
    predict_result = model.predict(dd)
    print(predict_result)
    assert predict_result[0] == 974525.859041541

test_model_use()
