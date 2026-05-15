import os
import pickle
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import pandas as pd
from transformers import AutoTokenizer, BertModel


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_THRESHOLD = 0.5
ASSETS_PATH = "data/pre_processed_assets.pkl"
TRAIN_CSV_PATH = "data/final_balanced_training.csv"


class TriIntelligenceNet(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        for p in self.bert.parameters():
            p.requires_grad = False

        self.lstm = nn.LSTM(num_features, 128, num_layers=2, batch_first=True, bidirectional=True)
        self.relational_net = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(768 + 256 + 32, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, ids, mask, num_data, risk_score):
        with torch.no_grad():
            text_feat = self.bert(input_ids=ids, attention_mask=mask).pooler_output

        lstm_in = num_data.float().unsqueeze(1) if num_data.dim() == 2 else num_data.float()
        _, (h_n, _) = self.lstm(lstm_in)
        temporal_feat = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)

        rs = risk_score.to(DEVICE).float() if torch.is_tensor(risk_score) else torch.tensor(risk_score, device=DEVICE).float()
        batch_size = text_feat.size(0)
        if rs.dim() == 0:
            risk_input = rs.unsqueeze(0).expand(batch_size, 1)
        elif rs.dim() == 1:
            if rs.size(0) == batch_size:
                risk_input = rs.unsqueeze(-1)
            else:
                risk_input = rs.mean().unsqueeze(0).expand(batch_size, 1)
        elif rs.dim() == 2 and rs.size(0) == batch_size:
            risk_input = rs.mean(dim=1, keepdim=True)
        else:
            risk_input = rs.mean().unsqueeze(0).expand(batch_size, 1)

        rel_feat = self.relational_net(risk_input)
        combined = torch.cat((text_feat, temporal_feat, rel_feat), dim=1)
        return self.classifier(combined)


class FraudInferenceEngine:
    """
    API-like inference engine for app integration.
    Loads model + assets once and serves repeated predictions.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        assets_path: str = ASSETS_PATH,
        train_csv_path: str = TRAIN_CSV_PATH,
        threshold: float = DEFAULT_THRESHOLD,
    ):
        self.assets_path = assets_path
        self.train_csv_path = train_csv_path
        self.threshold = threshold

        if not os.path.exists(self.assets_path):
            raise FileNotFoundError(f"Assets file not found: {self.assets_path}")

        with open(self.assets_path, "rb") as f:
            self.assets = pickle.load(f)

        self.num_cols = self.assets["num_cols"]
        self.scaler = self.assets["scaler"]
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        self.model_path = self._resolve_model_path(model_path)
        self.model = TriIntelligenceNet(len(self.num_cols)).to(DEVICE)
        self.model.load_state_dict(torch.load(self.model_path, map_location=DEVICE))
        self.model.eval()

        self.default_numeric_values, self.account_risk_map, self.mule_set = self._build_reference_stats()

    @staticmethod
    def _resolve_model_path(model_path: Optional[str]) -> str:
        if model_path and os.path.exists(model_path):
            return model_path
        if os.path.exists("tri_intel_best.pth"):
            return "tri_intel_best.pth"
        if os.path.exists("tri_intel_e10.pth"):
            return "tri_intel_e10.pth"
        raise FileNotFoundError("No model checkpoint found. Expected tri_intel_best.pth or tri_intel_e10.pth.")

    def _build_reference_stats(self):
        if not os.path.exists(self.train_csv_path):
            defaults = {c: 0.0 for c in self.num_cols}
            return defaults, {}, set()

        df = pd.read_csv(self.train_csv_path).fillna(0)
        sampled_index = self.assets.get("sampled_index", [])
        if sampled_index:
            # Keep same universe used by preprocessing/training.
            df = df.loc[sampled_index].reset_index(drop=True)
        df = df.head(min(50000, len(df)))

        defaults = {}
        for col in self.num_cols:
            if col in df.columns:
                defaults[col] = float(df[col].median())
            else:
                defaults[col] = 0.0

        account_risk_map = {}
        mule_set = set()
        acct_col = "card1" if "card1" in df.columns else None
        if acct_col and "isFraud" in df.columns:
            stats = df.groupby(acct_col).agg(
                txn_count=(acct_col, "size"),
                fraud_count=("isFraud", "sum"),
            )
            stats["fraud_rate"] = stats["fraud_count"] / stats["txn_count"]
            txn_thresh = max(5, int(stats["txn_count"].quantile(0.90)))
            mule_mask = (stats["txn_count"] >= txn_thresh) & (stats["fraud_rate"] >= 0.05)
            mule_set = set(stats[mule_mask].index.astype(str).tolist())
            account_risk_map = stats["fraud_rate"].astype(float).to_dict()
            account_risk_map = {str(k): float(v) for k, v in account_risk_map.items()}

        return defaults, account_risk_map, mule_set

    def _build_numeric_vector(self, numeric_features: Optional[Dict[str, float]]) -> torch.Tensor:
        row = self.default_numeric_values.copy()
        if numeric_features:
            for k, v in numeric_features.items():
                if k in row and v is not None:
                    row[k] = float(v)
        ordered = [row[c] for c in self.num_cols]
        transformed = self.scaler.transform([ordered])
        return torch.tensor(transformed, dtype=torch.float, device=DEVICE)

    def predict_message_and_account(
        self,
        message: str,
        sender_account: str,
        transaction_sent: bool = False,
        numeric_features: Optional[Dict[str, float]] = None,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Predict fraud/not-fraud on an unseen message + sender account.

        Args:
            message: Text message to classify.
            sender_account: Sender id (e.g., card1/account id).
            transaction_sent: If True and sender is mule-flagged, add block-level warning.
            numeric_features: Optional numeric feature dict keyed by column names from training.
            threshold: Optional override threshold.
        """
        if not isinstance(message, str) or len(message.strip()) == 0:
            raise ValueError("message must be a non-empty string")
        if sender_account is None:
            sender_account = ""
        sender_account = str(sender_account)

        numeric_tensor = self._build_numeric_vector(numeric_features)
        sender_risk = float(self.account_risk_map.get(sender_account, 0.0))
        risk_tensor = torch.tensor([sender_risk], dtype=torch.float, device=DEVICE)

        enc = self.tokenizer(
            [message],
            max_length=48,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        ids = enc["input_ids"].to(DEVICE)
        mask = enc["attention_mask"].to(DEVICE)

        with torch.no_grad():
            logits = self.model(ids, mask, numeric_tensor, risk_tensor)
            fraud_prob = float(torch.sigmoid(logits).item())

        use_threshold = self.threshold if threshold is None else float(threshold)
        is_fraud = fraud_prob >= use_threshold
        is_mule = sender_account in self.mule_set

        warning = None
        if is_mule and transaction_sent:
            warning = "WARNING: sender account is mule-flagged; review or block transaction."
        elif is_mule:
            warning = "WARNING: sender account is mule-flagged."

        return {
            "message": message,
            "sender_account": sender_account,
            "fraud_probability": round(fraud_prob, 6),
            "prediction": "fraud" if is_fraud else "not_fraud",
            "is_fraud": bool(is_fraud),
            "threshold_used": use_threshold,
            "sender_risk_score": round(sender_risk, 6),
            "is_mule_account": bool(is_mule),
            "transaction_sent": bool(transaction_sent),
            "warning": warning,
            "model_path": self.model_path,
            "device": str(DEVICE),
        }


_DEFAULT_ENGINE = None


def _get_default_engine(model_path: Optional[str] = None) -> FraudInferenceEngine:
    global _DEFAULT_ENGINE
    if _DEFAULT_ENGINE is None:
        _DEFAULT_ENGINE = FraudInferenceEngine(model_path=model_path)
    return _DEFAULT_ENGINE


def predict_message_and_account(
    message: str,
    sender_account: str,
    transaction_sent: bool = False,
    numeric_features: Optional[Dict[str, float]] = None,
    model_path: Optional[str] = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> Dict[str, Any]:
    """
    Stateless helper for app integration.
    Reuses a cached engine after first call.
    """
    engine = _get_default_engine(model_path=model_path)
    return engine.predict_message_and_account(
        message=message,
        sender_account=sender_account,
        transaction_sent=transaction_sent,
        numeric_features=numeric_features,
        threshold=threshold,
    )


if __name__ == "__main__":
    # Example local test on unseen data.
    result = predict_message_and_account(
        message="Urgent! Send money now to unlock your account.",
        sender_account="999999",
        transaction_sent=True,
        numeric_features=None,  # You can pass real transaction features here.
    )
    print(result)
