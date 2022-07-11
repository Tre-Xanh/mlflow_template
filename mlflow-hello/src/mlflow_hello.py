import mlflow
import pandas as pd
from loguru import logger

class MlflowHello(mlflow.pyfunc.PythonModel):
    def __init__(self, n: int) -> None:
        self.n = n

    def load_context(self, context):
        logger.debug("Start ...")
        logger.info(f"n = {self.n}")
        logger.debug("... Done")

    @logger.catch()
    def predict(self, context, df: pd.DataFrame) -> pd.DataFrame:
        logger.debug("Start ...")
        logger.debug(f"df\n{df}")
        out = df + self.n
        logger.debug(f"out\n{out}")
        logger.debug("... Done")
        return out

def log_model(model_fp: str, n: int):
    logger.debug("Start ...")
    mlflow_model_info = dict(
        # artifact_path=None,
        code_path=["src"],
        conda_env="mlflow_hello_env.yml",
        python_model=MlflowHello(n),
    )
    # model_info = mlflow.pyfunc.log_model(**mlflow_model_info)
    mlflow.pyfunc.save_model(model_fp, **mlflow_model_info)
    logger.debug(f"model_fp={model_fp}")
    # mlflow_run_id = model_info.model_uri.removesuffix("None")
    # logger.info(f"mlflow_run_id={mlflow_run_id}")
    logger.debug("... Done")
