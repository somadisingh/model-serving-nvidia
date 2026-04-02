from fastapi import FastAPI
from pydantic import BaseModel, Field
import numpy as np
import onnxruntime as ort
import os

app = FastAPI(
    title="Aesthetic Score API (ONNX)",
    description="API for predicting image aesthetic scores using ONNX Runtime. "
                "Accepts pre-computed 768-dim CLIP ViT-L/14 embeddings.",
    version="1.0.0"
)


# --- Request / Response schemas ---

class EmbeddingRequest(BaseModel):
    embedding: list[float]  # 768-dim CLIP embedding

class PersonalizedRequest(BaseModel):
    embedding: list[float]  # 768-dim CLIP embedding
    user_idx: int            # integer index of the user

class ScoreResponse(BaseModel):
    score: float = Field(..., ge=0, le=1)

class BatchEmbeddingRequest(BaseModel):
    embeddings: list[list[float]]  # list of 768-dim CLIP embeddings

class BatchPersonalizedRequest(BaseModel):
    embeddings: list[list[float]]
    user_indices: list[int]

class BatchScoreResponse(BaseModel):
    scores: list[float]


# --- Load ONNX models at startup ---

GLOBAL_MODEL_PATH = os.environ.get("GLOBAL_ONNX_PATH", "models/flickr_global.onnx")
PERSONAL_MODEL_PATH = os.environ.get("PERSONAL_ONNX_PATH", "models/flickr_personalized.onnx")

providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

global_session = ort.InferenceSession(GLOBAL_MODEL_PATH, providers=providers)
personal_session = ort.InferenceSession(PERSONAL_MODEL_PATH, providers=providers)

print(f"Global model loaded from {GLOBAL_MODEL_PATH}  (EP: {global_session.get_providers()})")
print(f"Personal model loaded from {PERSONAL_MODEL_PATH}  (EP: {personal_session.get_providers()})")


# --- Endpoints ---

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict/global", response_model=ScoreResponse)
def predict_global(req: EmbeddingRequest):
    emb = np.array(req.embedding, dtype=np.float32).reshape(1, -1)
    input_name = global_session.get_inputs()[0].name
    result = global_session.run(None, {input_name: emb})
    score = float(result[0].flatten()[0])
    return ScoreResponse(score=score)


@app.post("/predict/global/batch", response_model=BatchScoreResponse)
def predict_global_batch(req: BatchEmbeddingRequest):
    emb = np.array(req.embeddings, dtype=np.float32)
    input_name = global_session.get_inputs()[0].name
    result = global_session.run(None, {input_name: emb})
    scores = result[0].flatten().tolist()
    return BatchScoreResponse(scores=scores)


@app.post("/predict/personalized", response_model=ScoreResponse)
def predict_personalized(req: PersonalizedRequest):
    emb = np.array(req.embedding, dtype=np.float32).reshape(1, -1)
    user_idx = np.array([req.user_idx], dtype=np.int64)
    result = personal_session.run(None, {"embedding": emb, "user_idx": user_idx})
    score = float(result[0].flatten()[0])
    return ScoreResponse(score=score)


@app.post("/predict/personalized/batch", response_model=BatchScoreResponse)
def predict_personalized_batch(req: BatchPersonalizedRequest):
    emb = np.array(req.embeddings, dtype=np.float32)
    user_idx = np.array(req.user_indices, dtype=np.int64)
    result = personal_session.run(None, {"embedding": emb, "user_idx": user_idx})
    scores = result[0].flatten().tolist()
    return BatchScoreResponse(scores=scores)
