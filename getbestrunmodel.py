import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Search runs in experiment '1', order by F1 score descending
runs = client.search_runs(
    experiment_ids=["1"], order_by=["metrics.f1_score DESC"], max_results=1
)

# Get the run_id of the top-performing model
best_run_id = runs[0].info.run_id
#best_model_uri = f"runs:/{best_run_id}/model"
best_model_uri = f"runs:/1/models/{best_run_id}/artifacts"
# Load the best model into your API
best_model = mlflow.sklearn.load_model(best_model_uri)
print (best_model_uri)