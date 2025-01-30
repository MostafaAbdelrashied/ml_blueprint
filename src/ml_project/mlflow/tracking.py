import os

import mlflow


class MLflowTracker:
    def __init__(self, tracking_uri: str = None, experiment_name: str = None):
        self.tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI")
        self.experiment_name = experiment_name or os.getenv("MLFLOW_EXPERIMENT_NAME")

    def setup(self):
        mlflow.set_tracking_uri(self.tracking_uri)
        # create experiment if it doesn't exist
        if self.experiment_name not in [
            exp.name for exp in mlflow.tracking.MlflowClient().search_experiments()
        ]:
            mlflow.create_experiment(self.experiment_name)
        mlflow.set_experiment(self.experiment_name)

    def start_run(self, run_name):
        return mlflow.start_run(run_name=run_name)
