import os
import mlflow
import pandas as pd
from loguru import logger

test_df = pd.DataFrame([range(10)])
test_df_str = pd.DataFrame("abc".split()) # this is not good for numerical addition

def test_saved_model(model_fp: str, n: int):
    logger.debug(f"model_fp: {model_fp}")
    logger.debug(f"n: {n}")
    loaded_model = mlflow.pyfunc.load_model(model_fp)
    model_output = loaded_model.predict(test_df)
    assert model_output.equals(test_df + n)

def test_api(api_port: str="5000"):
    import httpx
    scoring_uri: str =os.getenv("SCORING_URI") or f"http://127.0.0.1:{api_port}/invocations"
    scoring_uri = scoring_uri.strip("'\"")
    logger.info(f"start {scoring_uri} ...")
    data = test_df.to_json(orient="split", index=False)
    res = httpx.post(
        scoring_uri, data=data, headers={"Content-type": "application/json"},
    )
    logger.debug(res)
    res_js = res.json()
    # logger.debug(res_js)
    res_df = pd.DataFrame(res_js)
    logger.debug(res_df)
    logger.info(f"... done {scoring_uri}")
    return res_df
