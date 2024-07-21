import logging
from fastapi import FastAPI

import os

local_only : str = True if os.getenv("LOCAL_FILES_ONLY", "no").lower() == "yes" else False


if "MODE" in os.environ.keys():
    mode : str = os.environ["MODE"]
else:
    logging.warning("MODE not set, defaulting to CPU")
    # Defaulting on CPU for more compatibility
    mode : str= 'cpu'

logging.info("Using device: " + mode)

if "MODEL_NAME" in os.environ.keys():
    model_name : str = os.environ["MODEL_NAME"]
else:
    logging.warning("MODEL_NAME not set, defaulting to intfloat/multilingual-e5-large-instruct")
    # Default model to load
    model_name : str = 'intfloat/multilingual-e5-large-instruct'

logging.debug("Loading the model !")
from sentence_transformers import SentenceTransformer
model = SentenceTransformer(model_name, device=mode, local_only=True)
logging.debug("Done !")

app = FastAPI()

from pydantic import BaseModel

class EmbeddingsQuery(BaseModel):
    queries: list[str]
    
class EmbeddingsResponse(BaseModel):
    vectors: list[list[float]]

@app.post("/v1/get_embeddings")
def read_root(query: EmbeddingsQuery):
    all_requested_txts: list[str] = query.queries
    return_data: list[list[float]] = model.encode(all_requested_txts, normalize_embeddings=False)
    return EmbeddingsResponse(vectors=return_data)

logging.info("Server ready !")