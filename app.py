from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from src.pipeline.training_pipeline import TrainingPipeline
from uvicorn import run as app_run
from src.pipeline.prediction_pipeline import IMDBClassifier,Data
from src.exception import MyException
import os,sys
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ReviewRequest(BaseModel):
    review: str

@app.get("/train")
def train_model():
    pipe = TrainingPipeline()
    pipe.run_pipeline()
    return {"message": "Training pipeline completed."}


@app.post("/predict")
def pred(input: ReviewRequest):
    try:
        data = Data(review=input.review)
        df = data.get_dataframe()
        classifier = IMDBClassifier()
        result = classifier.predict(df)

        if hasattr(result, "tolist"):
            result = result.tolist()
        return {"prediction": result}
    
    except Exception as e:
        raise MyException(sys,e) from e
    

if __name__ == "__main__":
    app_run(app, host="0.0.0.0", port=8000)
