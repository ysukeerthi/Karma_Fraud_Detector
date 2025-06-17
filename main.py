

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import joblib
import json
import uvicorn
from fraud_detector import extract_features

app = FastAPI()


model = None
vectorizer = None


with open("config.json") as f:
    config = json.load(f)


@app.on_event("startup")
def load_model():
    global model, vectorizer
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    print("✅ Model and vectorizer loaded on startup.")

class KarmaActivity(BaseModel):
    activity_id: str
    type: str
    timestamp: str
    from_user: Optional[str] = None
    post_id: Optional[str] = None
    source: Optional[str] = None
    content: Optional[str] = None

class AnalyzeRequest(BaseModel):
    user_id: str
    karma_log: List[KarmaActivity]

class SuspiciousActivity(BaseModel):
    activity_id: str
    reason: str
    score: float

class AnalyzeResponse(BaseModel):
    user_id: str
    fraud_score: float
    suspicious_activities: List[SuspiciousActivity]
    status: str

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest):
    features, activity_scores, reasons = extract_features(request.karma_log, config)
    X = vectorizer.transform([features])
    fraud_score = float(model.predict_proba(X)[0][1])

    suspicious_activities = []
    for act, score, reason in zip(request.karma_log, activity_scores, reasons):
        if score > config["activity_score_threshold"]:
            suspicious_activities.append(SuspiciousActivity(
                activity_id=act.activity_id,
                reason=reason,
                score=score
            ))

    if fraud_score >= config["banned_threshold"]:
        status = "banned"
    elif fraud_score >= config["flagged_threshold"]:
        status = "flagged"
    else:
        status = "clean"

    return AnalyzeResponse(
        user_id=request.user_id,
        fraud_score=fraud_score,
        suspicious_activities=suspicious_activities,
        status=status
    )

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/version")
def version():
    return {
        "model_version": "1.0",
        "config_version": config.get("version", "1.0")
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
