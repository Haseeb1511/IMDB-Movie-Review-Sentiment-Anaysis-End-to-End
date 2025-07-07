```python
import dagshub
dagshub.init(repo_owner='haseebmanzoor1511', repo_name='IMDB-Movie-Review-Sentiment-Anaysis-End-to-End', mlflow=True)



import mlflow
with mlflow.start_run():
  # Your training code here...
  mlflow.log_metric('accuracy', 42)
  mlflow.log_param('Param name', 'Value')


mlflow.autolog()

```