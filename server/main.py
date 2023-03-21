from asyncio.log import logger
import os 
import json
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from typing import Union, List, Tuple
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import logging 

# logging config 
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)

from model.seq2seq import (
    get_backbone,
    do_event_detection,
    do_event_argument_extraction
)
from io_format import Result, Event, Argument, Input
from extract_utils import get_ed_result, get_eae_result
from extract_utils import (
    prepare_for_eae_from_input,
    prepare_for_eae_from_pred
)
from fastapi.middleware.cors import CORSMiddleware
origins = [
    "*",
]
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="static")


# model 
model_path = "/data/ph/OmniEvent_Model/s2s-mt5-ed"
ed_model, ed_tokenizer, _ = get_backbone("t5", model_path, model_path)
ed_model.cuda()
ed_model.eval()

model_path = "/data/ph/OmniEvent_Model/s2s-mt5-eae"
eae_model, eae_tokenizer, _ = get_backbone("t5", model_path, model_path)
eae_model.cuda()
eae_model.eval()


# @app.get("/")
# async def root():
#     return {"message": "Hello World"}

@app.get("/")
def root(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request
        }
    )


@app.post("/api/query")
async def main(item: Input):
    logger.info(item)
    schema = f"<{item.ontology.lower()}>"
    if item.task == "Event Detection":
        events = do_event_detection(ed_model, ed_tokenizer, [item.text], [schema])
        results = get_ed_result([item.text], events)
    elif item.task == "Event Argument Extraction":
        if len(item.triggers) == 0:
            results = [{
                "text": item.text,
                "events": []
            }]
            return results
        instances = prepare_for_eae_from_input([item.text], [item.triggers], [schema])
        arguments = do_event_argument_extraction(eae_model, eae_tokenizer, instances)
        results = get_eae_result(instances, arguments)
    elif item.task == "Event Extraction":
        events = do_event_detection(ed_model, ed_tokenizer, [item.text], [schema])
        instances = prepare_for_eae_from_pred([item.text], events, [schema])
        if len(instances[0]["triggers"]) == 0:
            results = [{
                "text": instances[0]["text"],
                "events": []
            }]
            return results
        arguments = do_event_argument_extraction(eae_model, eae_tokenizer, instances)
        results = get_eae_result(instances, arguments)
    logger.info(results)
    return results

