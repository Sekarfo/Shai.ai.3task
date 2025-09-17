#!/usr/bin/env python3
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any
import joblib, numpy as np, pandas as pd
from datetime import datetime
import json, joblib

from pathlib import Path


ART = Path(r"C:\Users\trudk\OneDrive\Desktop\shai_ai_3task\artifacts")

# Загружаем новые артефакты
model = joblib.load(ART / "model.pkl")
num_cols = json.loads((ART / "num_cols.json").read_text(encoding="utf-8"))
thresholds = json.loads((ART / "thresholds.json").read_text(encoding="utf-8"))

YELLOW_THR = float(thresholds.get("yellow", 0.40))
RED_THR    = float(thresholds.get("red", 0.60))

app = FastAPI(title="SupplyRisk Scorer (numeric)", version="0.1.0")

class ScoreRequest(BaseModel):
    routes: List[Dict[str, Any]]

class ScoreItem(BaseModel):
    route_id: str = ""
    risk_score: float
    risk_level: str

@app.get("/health")
def health():
    return {"status": "ok", "loaded_model_features": num_cols, "time": datetime.utcnow().isoformat()+"Z"}

@app.post("/score", response_model=List[ScoreItem])
def score(req: ScoreRequest):
    """
    Ожидает: routes = [{...фичи...}], где ключи совпадают с num_cols.
    Неизвестные колонки игнорируются, отсутствующие — докинем как NaN.
    """
    X = pd.DataFrame(req.routes)

    # гарантируем все ожидаемые моделью колонки
    for c in num_cols:
        if c not in X.columns:
            X[c] = np.nan
    X = X[num_cols]  # порядок признаков

    proba = model.predict_proba(X)[:, 1]
    out = []
    for r, p in zip(req.routes, proba):
        level = "green"
        if p >= 0.60: level = "red"
        elif p >= 0.40: level = "yellow"
        out.append(ScoreItem(
            route_id=str(r.get("route_id","")),
            risk_score=float(np.round(p, 3)),
            risk_level=level
        ))
    return out

# (опционально) простейший what-if: можно передать любые фичи и получить новый скор
class WhatIfRequest(BaseModel):
    features: Dict[str, Any]

@app.post("/whatif")
def whatif(req: WhatIfRequest):
    X = pd.DataFrame([req.features])
    for c in num_cols:
        if c not in X.columns:
            X[c] = np.nan
    X = X[num_cols]
    p = float(model.predict_proba(X)[:,1][0])
    level = "green"
    if p >= 0.60: level = "red"
    elif p >= 0.40: level = "yellow"
    return {"risk_score": round(p,3), "risk_level": level}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)