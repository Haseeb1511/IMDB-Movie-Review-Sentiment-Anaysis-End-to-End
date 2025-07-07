import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import re,mlflow,os,dagshub
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier
from sklearn.pipeline import Pipeline,FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score,precision_score,f1_score,classification_report
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from dotenv import load_dotenv
load_dotenv()

nltk.download('stopwords')
nltk.download("wordnet")

wn = WordNetLemmatizer()
sw = set(stopwords.words("english"))

import os

CONFIG = {
    "data_path":"notebook/IMDB.csv",
    "test_size": 0.2,
    "mlflow_tracking_uri": os.getenv("DAGSHUB_MLFLOW_TRACKING_URI"),
    "dagshub_repo_owner": os.getenv("DAGSHUB_REPO_OWNER"),
    "dagshub_repo_name": os.getenv("DAGSHUB_REPO_NAME"),
    "experiment_name": "Bow vs TfIdf"
}


mlflow.set_tracking_uri(CONFIG["mlflow_tracking_uri"])
dagshub.init(repo_owner=CONFIG["dagshub_repo_owner"], repo_name=CONFIG["dagshub_repo_name"], mlflow=True)
mlflow.set_experiment(CONFIG["experiment_name"])



def load_data(file_path):
    data = pd.read_csv(file_path)
    data["sentiment"] = data["sentiment"].map({"positive":1,"negative":0})
    return data

def clean_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) # Remove links (http, https, www)
    text = re.sub(r'[^a-zA-Z\s]', " ", text)
    text = re.sub(r'\s+', ' ', text).strip() 
    text = text.lower().split()
    text = " ".join(wn.lemmatize(word) for word in text if word not in sw )
    return text



ALGORITHMS = {
    'LogisticRegression': LogisticRegression(),
    'MultinomialNB': MultinomialNB(),
    'RandomForest': RandomForestClassifier(),
    'GradientBoosting': GradientBoostingClassifier()
}

VECTORIZERS = {
    'BoW': CountVectorizer(),
    'TF-IDF': TfidfVectorizer()
}


def train_evaluate(df):
    with mlflow.start_run(run_name="all-experiment-2") as parent_run:
        for algo_name,algorithm in ALGORITHMS.items():
            for vec_name,vectorizer in VECTORIZERS.items():
                with mlflow.start_run(run_name=f"{algo_name} with {vec_name}",nested=True) as child_run:
                    
                    x = df.drop("sentiment", axis=1)
                    y = df["sentiment"]
                    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=CONFIG["test_size"],random_state=42)


                    text_pipline = Pipeline(steps=[
                        ("clean_text", FunctionTransformer(lambda x: x.apply(clean_text))),
                        (vec_name, vectorizer)])
                    column = ColumnTransformer(transformers=[
                        ("text_pipeline", text_pipline, "review")])

                    model = algorithm
                    pipe = Pipeline(steps=[
                        ("column", column),
                        ("classifier", model)])
                    

                    #Train the model
                    pipe.fit(x_train,y_train)
                    
                    log_model_params(algo_name,model)                # custom function

                    # Log preprocessing parameters
                    mlflow.log_params({
                            "vectorizer": vec_name,
                            "algorithm": algo_name,
                            "test_size": CONFIG["test_size"]
                        })
                    
                    # Predition on test data
                    y_pred = pipe.predict(x_test)


                    # Calculate metrics
                    metrics = {
                    "accuracy" : accuracy_score(y_test, y_pred),
                    "precision" : precision_score(y_test, y_pred, average='binary'),
                    "f1-score" :f1_score(y_test, y_pred, average='binary')
                    }
                    for k, v in metrics.items():
                        mlflow.log_metric(k, v)


                    # ✅ Save classification report as artifact
                    report = classification_report(y_test, y_pred)
                    with open("classification_report.txt", "w") as f:
                        f.write(report)
                    mlflow.log_artifact("classification_report.txt")

                    
                    # Print results for verification
                    print(f"\nAlgorithm: {algo_name}, Vectorizer: {vec_name}")
                    print(f"Metrics: {metrics}")

def log_model_params(algo_name, model):
    """Logs hyperparameters of the trained model to MLflow."""
    params_to_log = {}

    if algo_name == 'LogisticRegression':
        params_to_log["C"] = model.C

    elif algo_name == 'MultinomialNB':
        params_to_log["alpha"] = model.alpha

    elif algo_name == 'RandomForest':
        params_to_log["n_estimators"] = model.n_estimators


    elif algo_name == 'GradientBoosting':
        params_to_log["n_estimators"] = model.n_estimators
        params_to_log["learning_rate"] = model.learning_rate
        params_to_log["max_depth"] = model.max_depth

    mlflow.log_params(params_to_log)


if __name__=="__main__":
    df = load_data(file_path=CONFIG["data_path"])
    train_evaluate(df=df)