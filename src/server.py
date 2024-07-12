from fastapi import FastAPI

import os
mode = os.environ["MODE"]

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('intfloat/multilingual-e5-large-instruct', device=mode)

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
