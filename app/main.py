from fastapi import FastAPI
#BaseModel: A class from the Pydantic library for creating data models with validation.
from pydantic import BaseModel
from app.model.model import predict_pipeline
from app.model.model import __version__ as model_version

app = FastAPI()



"""
to make sure that we pass the correct datatype to this api when we send a data,
so will define a "class TextIn" that inherits from BaseModel this class have only one field,
which is text and its type string.
when we send a data this not a string the FastAPI can detect this so will raise an error in the api.

"""
# this for the input data  
class TextIn(BaseModel):
    text:str
    

# create the endpoints

@app.get("/")
def home():
    return{"Health_check":"Ok","model_version":model_version}


# response_model=PredictionOut parameter in the decorator ensures that the response conforms to the PredictionOut
@app.post("/predict")
def predict(payload: TextIn):
    class_ = predict_pipeline(payload.text)
    return {"Class":class_}