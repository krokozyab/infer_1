import asyncio
import uvicorn
from fastapi import FastAPI, Body
from transformers import pipeline

# Global variables
# todo save/load this models list in config file
g_models = {"cardiffnlp": "cardiffnlp/twitter-xlm-roberta-base-sentiment",
            "ivanlau": "ivanlau/language-detection-fine-tuned-on-xlm-roberta-base",
            "svalabs": "svalabs/twitter-xlm-roberta-crypto-spam",
            "EIStakovskii": "EIStakovskii/xlm_roberta_base_multilingual_toxicity_classifier_plus",
            "jy46604790": "jy46604790/Fake-News-Bert-Detect"}

# dict {model_name: pipeline}
g_model_pipelines: dict = {}

# dict returning API data {model_name: {score:xx, label: xx} }
g_data: dict = {}


def load_models_pipelines(models_dict: dict) -> None:
    """helper function load models pipelines by populating g_model_pipelines dict"""
    for model_name, model in models_dict.items():
        try:
            g_model_pipelines[model_name] = pipeline(model=model)
        except:
            err = " Error while loading:  %s" % model_name
            raise RuntimeError(err)


async def swap_items(result: dict) -> dict:
    """helper function swap label and score to meet required result format"""
    # initializing swap indices
    i, j = 0, 1
    # conversion to tuples
    tups = list(result.items())
    # swapping by indices
    tups[i], tups[j] = tups[j], tups[i]
    # converting back to dict
    return dict(tups)


async def inference(sentence: str, model_name: str, model_pipeline) -> None:
    """perform inference"""
    result = model_pipeline(sentence)[0]
    # append inference to resulting dict
    g_data[model_name] = await swap_items(result)


# load models before inference application starts, when run first time downloads models from huggingfaces
load_models_pipelines(g_models)

# start api
app = FastAPI()


@app.post("/")
async def all_infers(sentence: str = Body(...)) -> dict:
    """inference endpoint"""
    # create inference tasks per model
    tasks = [inference(sentence, model_name, model) for model_name, model in g_model_pipelines.items()]

    # run inference tasks and populate resulting dict
    await asyncio.gather(*tasks)
    return g_data


if __name__ == '__main__':
    uvicorn.run("main:app", port=9000, reload=True)
