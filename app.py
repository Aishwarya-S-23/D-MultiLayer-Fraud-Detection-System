import os
from pathlib import Path
from typing import Any, Dict, Optional

import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from inference import FraudInferenceEngine


app = FastAPI(title="Fraud + Mule Detection API", version="1.0.0")
BASE_DIR = Path(__file__).resolve().parent
UI_DIR = BASE_DIR / "ui"


VERIFY_TOKEN = os.getenv("VERIFY_TOKEN", "change-me")
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN", "EAAcHF2xZCgbcBRJdglZCYTiqtMRWzbrvep4p8COp0NoEVHxi0N7ckPV0bKGAtbmwUExm69QvYzluXGDzsEbZBTZASo2FmNZBG2FJTtt8uHZB6BAKu6U6M9dRqJNSecPAHY4o31HPGZAyGPgonOis23Y1QTpHgLd9CowFPZBmiDXXhvZClvt1tXhld709QObddHfJbtHOJ7S8ZADabVzWUodY4QGuIv7afRiWrhTFvbRRYzSntkPhwnezeGmrisUCJ5TZBINxMT4Lq1CNwZCHMgnzQZAY35SduzwZDZD")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID", "EAAcHF2xZCgbcBRJdglZCYTiqtMRWzbrvep4p8COp0NoEVHxi0N7ckPV0bKGAtbmwUExm69QvYzluXGDzsEbZBTZASo2FmNZBG2FJTtt8uHZB6BAKu6U6M9dRqJNSecPAHY4o31HPGZAyGPgonOis23Y1QTpHgLd9CowFPZBmiDXXhvZClvt1tXhld709QObddHfJbtHOJ7S8ZADabVzWUodY4QGuIv7afRiWrhTFvbRRYzSntkPhwnezeGmrisUCJ5TZBINxMT4Lq1CNwZCHMgnzQZAY35SduzwZDZD")
MODEL_THRESHOLD = float(os.getenv("MODEL_THRESHOLD", "0.5"))


class PredictRequest(BaseModel):
    message: str = Field(..., min_length=1)
    sender_account: str = Field(..., min_length=1)
    transaction_sent: bool = False
    numeric_features: Optional[Dict[str, float]] = None
    threshold: Optional[float] = None


class PredictResponse(BaseModel):
    prediction: str
    fraud_probability: float
    is_fraud: bool
    is_mule_account: bool
    warning: Optional[str]
    sender_account: str
    threshold_used: float
    sender_risk_score: float
    transaction_sent: bool
    model_path: str
    device: str
    message: str


def _create_engine() -> FraudInferenceEngine:
    return FraudInferenceEngine(threshold=MODEL_THRESHOLD)


engine = _create_engine()

if UI_DIR.exists():
    app.mount("/ui/assets", StaticFiles(directory=UI_DIR), name="ui-assets")


@app.get("/ui", include_in_schema=False)
def ui_index():
    if not UI_DIR.exists():
        raise HTTPException(status_code=404, detail="UI directory not found")
    return FileResponse(UI_DIR / "index.html")


@app.get("/ui/app", include_in_schema=False)
def ui_app():
    if not UI_DIR.exists():
        raise HTTPException(status_code=404, detail="UI directory not found")
    return FileResponse(UI_DIR / "index.html")


@app.get("/health")
def health() -> Dict[str, Any]:
    model_device = str(next(engine.model.parameters()).device)
    return {
        "status": "ok",
        "model_loaded": True,
        "model_path": engine.model_path,
        "device": model_device,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> Dict[str, Any]:
    try:
        result = engine.predict_message_and_account(
            message=payload.message,
            sender_account=payload.sender_account,
            transaction_sent=payload.transaction_sent,
            numeric_features=payload.numeric_features,
            threshold=payload.threshold,
        )
        return result
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/webhook")
def verify_webhook(
    hub_mode: str = Query(default="", alias="hub.mode"),
    hub_verify_token: str = Query(default="", alias="hub.verify_token"),
    hub_challenge: str = Query(default="", alias="hub.challenge"),
):
    if hub_mode == "subscribe" and hub_verify_token == VERIFY_TOKEN:
        return int(hub_challenge) if hub_challenge.isdigit() else hub_challenge
    raise HTTPException(status_code=403, detail="Webhook verification failed")


def send_whatsapp_text(to_number: str, body_text: str) -> None:
    if not WHATSAPP_TOKEN or not PHONE_NUMBER_ID:
        return
    url = f"https://graph.facebook.com/v21.0/{PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to_number,
        "type": "text",
        "text": {"body": body_text},
    }
    requests.post(url, headers=headers, json=payload, timeout=15)


def parse_whatsapp_message(payload: Dict[str, Any]) -> Optional[Dict[str, str]]:
    try:
        entry = payload.get("entry", [])[0]
        changes = entry.get("changes", [])[0]
        value = changes.get("value", {})
        messages = value.get("messages", [])
        if not messages:
            return None
        msg = messages[0]
        sender_number = msg.get("from", "")
        text_body = msg.get("text", {}).get("body", "")
        if not sender_number or not text_body:
            return None
        return {"sender_number": sender_number, "message": text_body}
    except Exception:
        return None


@app.post("/webhook")
def receive_webhook(payload: Dict[str, Any]) -> Dict[str, Any]:
    parsed = parse_whatsapp_message(payload)
    if not parsed:
        return {"ok": True, "message": "No user text message found"}

    sender_number = parsed["sender_number"]
    message = parsed["message"]

    # In WhatsApp-only flow we use sender number as account id.
    result = engine.predict_message_and_account(
        message=message,
        sender_account=sender_number,
        transaction_sent=False,
        numeric_features=None,
        threshold=MODEL_THRESHOLD,
    )

    if result["is_fraud"] and result["is_mule_account"]:
        reply = "High-risk alert: this message looks like a scam and sender pattern is suspicious."
    elif result["is_fraud"]:
        reply = "Alert: this message may be fraudulent. Do not share OTP/PIN or send money."
    elif result["is_mule_account"]:
        reply = "Caution: sender pattern appears suspicious. Verify identity before making payments."
    else:
        reply = "No major fraud signal detected. Still verify links, OTP requests, and urgent payment demands."

    send_whatsapp_text(sender_number, reply)

    return {
        "ok": True,
        "sender_number": sender_number,
        "model_result": result,
        "reply_sent": bool(WHATSAPP_TOKEN and PHONE_NUMBER_ID),
    }
