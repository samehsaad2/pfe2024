
import pandas as pd
import numpy as np
from fastapi import FastAPI, File, UploadFile, BackgroundTasks
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, BaseConfig
import mlflow
import os
from dataprocessing import DataCleaning
from fastapi.responses import StreamingResponse

os.environ['MLFLOW_TRACKING_USERNAME'] = "rihemmanel54"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "3e93c19c879ea3562d4638daa2bab19d3eabb3c9"

# setup mlflow
mlflow.set_tracking_uri('https://dagshub.com/rihemmanel54/CarPricePrediction_mlFlow.mlflow')

app = FastAPI()
origins = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# let's call the model from the model registry (in the production stage)
df_mlflow = mlflow.search_runs(filter_string="metrics.Accuracy<1")
run_id = df_mlflow.loc[df_mlflow['metrics.Accuracy'].idxmax()]['run_id']
logged_model = f'runs:/{run_id}/ML_models'
model = mlflow.pyfunc.load_model(logged_model)

class Item(BaseModel):
    transmission: int
    fuel: int
    owner: int
    year: int
    km_driven: int
    engine: int
    max_power: int
    class Config(BaseConfig):
        protected_namespaces = ()
        json_schema_extra = {}

@app.get("/")
def read_root():
    return {"Hello": "to Car price prediction app"}

# this endpoint receives data in the form of csv (informations about one transaction)
ALLOWED_EXTENSIONS = {'csv'}

def is_valid_file_extension(filename: str) -> bool:
    return filename.lower().endswith('.csv')

@app.post("/upload")
async def return_predictions(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    df_copy = df.copy()
    data = DataCleaning(df)
    predictions = model.predict(data)
    data["predicted_price"] = predictions.tolist()
    return StreamingResponse(
        iter([data.to_csv(index=False)]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=data.csv"}
    )

# this endpoint receives data in the form of json (informations about one transaction)
@app.post("/predict")
def predict(item: Item):
    print(item)
    data = [item.transmission, item.fuel, item.owner, item.year, item.km_driven, item.engine, item.max_power]
    data = np.array(data).reshape(1, -1)
    predictions = model.predict(data.reshape(1, -1))
    print(predictions)
    return {"Predict": predictions[0]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3000)
