# --------------------------------------------------------------
# api/main.py – Vercel Serverless (No OOM)
# --------------------------------------------------------------
import os
import shutil
import tempfile
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import librosa
import scipy.signal
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import StandardScaler
import pandas as pd

# ---------- GLOBAL CACHE ----------
class Cache:
    models = {}
    data = {}
    scaler = None

cache = Cache()

# ---------- FASTAPI APP ----------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- LAZY LOAD + QUANTIZE ----------
def lazy_load_model(key: str, init_fn):
    if key not in cache.models:
        model = init_fn()
        if isinstance(model, nn.Module):
            model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
        cache.models[key] = model
    return cache.models[key]

# ---------- MODEL INITIALISERS ----------
def init_textbert():
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=8)
    path = "./checkpoints/textbert/kaggle/working/textbert_model_checkpoint/model.pt"
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return {"tokenizer": tokenizer, "model": model}

def init_multilingual():
    dir_path = "./checkpoints/multilingual/multilingual_my_model_checkpoint"
    tokenizer = AutoTokenizer.from_pretrained(dir_path)
    model = AutoModelForSequenceClassification.from_pretrained(dir_path)
    model.eval()
    return {"tokenizer": tokenizer, "model": model}

def init_eeg():
    class CNNGRU(nn.Module):
        def __init__(self):
            super().__init__()
            self.cnn = nn.Sequential(
                nn.Conv1d(645, 128, 3, padding=1), nn.ReLU(),
                nn.Conv1d(128, 256, 3, padding=1), nn.ReLU(),
            )
            self.gru = nn.GRU(256, 128, batch_first=True)
            self.fc = nn.Linear(128, 2)
        def forward(self, x):
            x = self.cnn(x).transpose(1, 2)
            _, h = self.gru(x)
            return self.fc(h.squeeze(0))
    model = CNNGRU()
    path = "./checkpoints/eeg/kaggle/working/eeg_model_checkpoint/model.pt"
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model

def init_audio():
    class GRUClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(40, 128, batch_first=True)
            self.fc = nn.Linear(128, 3)
        def forward(self, x):
            _, h = self.gru(x)
            return self.fc(h.squeeze(0))
    model = GRUClassifier()
    path = "./checkpoints/audio/kaggle/working/audio_model_checkpoint/model.pt"
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model

def init_spatial():
    class Spatial1DCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv1d(1, 32, 3, padding=1)
            self.pool = nn.MaxPool1d(2)
            self.conv2 = nn.Conv1d(32, 64, 3, padding=1)
            self.fc1 = nn.Linear(64*68, 128)
            self.fc2 = nn.Linear(128, 1)
        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = self.pool(x)
            x = torch.relu(self.conv2(x))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = torch.relu(self.fc1(x))
            return torch.sigmoid(self.fc2(x))
    model = Spatial1DCNN()
    path = "./checkpoints/spatial/kaggle/working/spatial_model_checkpoint/model.pt"
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model

# ---------- SPATIAL DATA (PRE-LOADED) ----------
def init_spatial_data():
    if "spatial" in cache.data:
        return
    paths = [
        "./checkpoints/spatial-datas/ahrf2023.csv",
        "./checkpoints/spatial-datas/ahrf2024_Feb2025.csv"
    ]
    dfs = []
    for p, year in zip(paths, [2023, 2024]):
        df = pd.read_csv(p, encoding="latin1", low_memory=False)
        keep = ['fips_st_cnty'] + [c for c in df.columns if any(k in c.lower() for k in ['povty','emplymt','eductn','psych','phys'])]
        df = df[keep].copy()
        df['fips_st_cnty'] = df['fips_st_cnty'].astype(str).str.zfill(5)
        df['year'] = year
        dfs.append(df)
    merged = pd.concat(dfs, ignore_index=True)
    feat = [c for c in merged.columns if c not in ['fips_st_cnty','year']]
    scaler = StandardScaler()
    scaler.fit(merged[feat].astype(np.float32))
    cache.scaler = scaler
    cache.data['spatial'] = merged
    cache.data['feat_cols'] = feat

# ---------- EEG PSD (NO MNE) ----------
def extract_psd(mat_path):
    import scipy.io
    mat = scipy.io.loadmat(mat_path)
    key = next(k for k in mat.keys() if k not in {'__header__','__version__','__globals__'})
    data = mat[key].T
    bands = [(0.5,4),(4,8),(8,13),(13,30),(30,50)]
    feats = []
    for lo, hi in bands:
        f, p = scipy.signal.welch(data.flatten(), fs=250, nperseg=256)
        band = p[(f >= lo) & (f <= hi)]
        feats.append(band.mean() if len(band) > 0 else 0.0)
    feats = np.array(feats, dtype=np.float32)
    stats = np.load("./checkpoints/eeg_preprocessing_stats.npz")
    feats = (feats - stats["mean"]) / (stats["std"] + 1e-8)
    return torch.tensor(feats).unsqueeze(0).unsqueeze(-1)

# ---------- PREDICT ----------
@app.post("/predict")
async def predict(
    text: Optional[str] = Form(None),
    multilingual_text: Optional[str] = Form(None),
    audio: Optional[UploadFile] = File(None),
    eeg: Optional[UploadFile] = File(None),
    fips_code: Optional[str] = Form(None),
):
    try:
        # Lazy load only what's needed
        textbert = lazy_load_model("textbert", init_textbert) if text else None
        multi = lazy_load_model("multilingual", init_multilingual) if (multilingual_text or text) else None
        eeg_mod = lazy_load_model("eeg", init_eeg) if eeg else None
        audio_mod = lazy_load_model("audio", init_audio) if audio else None
        spat_mod = lazy_load_model("spatial", init_spatial) if fips_code else None
        if fips_code and "spatial" not in cache.data:
            init_spatial_data()

        # File handling
        audio_path = eeg_path = None
        if audio:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                shutil.copyfileobj(audio.file, f)
                audio_path = f.name
        if eeg:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mat") as f:
                shutil.copyfileobj(eeg.file, f)
                eeg_path = f.name

        result = {"individual_predictions": {}}

        # TEXT
        if text and textbert:
            enc = textbert["tokenizer"](text, truncation=True, max_length=128, return_tensors="pt")
            with torch.no_grad():
                probs = torch.softmax(textbert["model"](**enc).logits, dim=1).cpu().numpy()[0]
            classes = ['adhd','anxiety','autism','bipolar','bpd','depression','ptsd','schizophrenia']
            idx = int(probs.argmax())
            score = float(probs[idx])
            label = classes[idx]
            if label == "depression" and score < 0.65:
                label = "Normal"
            result["text"] = label
            result["individual_predictions"]["text_score"] = round(score, 2)

        # MULTILINGUAL
        inp = multilingual_text or text
        if inp and multi:
            enc = multi["tokenizer"](inp, truncation=True, max_length=128, return_tensors="pt")
            with torch.no_grad():
                prob = torch.softmax(multi["model"](**enc).logits, dim=1)[0,1].item()
            result["multilingual"] = "Depression" if prob > 0.5 else "Normal"
        else:
            result["multilingual"] = "N/A"

        # AUDIO
        if audio_path and audio_mod:
            y, _ = librosa.load(audio_path, sr=16000)
            mfcc = librosa.feature.mfcc(y=y, sr=16000, n_mfcc=40)[:, :100]
            x = torch.tensor(mfcc.T, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits = audio_mod(x)
            result["audio"] = ["normal","mild_depression","severe_depression"][logits.argmax().item()]
            os.unlink(audio_path)
        else:
            result["audio"] = "N/A"

        # EEG
        if eeg_path and eeg_mod:
            x = extract_psd(eeg_path)
            with torch.no_grad():
                prob = torch.softmax(eeg_mod(x), dim=1)[0,1].item()
            result["eeg"] = "MDD" if prob > 0.5 else "HC"
            result["individual_predictions"]["eeg_score"] = round(prob, 2)
            os.unlink(eeg_path)
        else:
            result["eeg"] = "N/A"

        # SPATIAL
        if fips_code and spat_mod:
            fips = str(fips_code).zfill(5)
            row = cache.data['spatial'][cache.data['spatial']['fips_st_cnty'] == fips]
            if row.empty:
                result["individual_predictions"]["spatial_risk"] = "N/A"
            else:
                vec = row.iloc[0][cache.data['feat_cols']].values.astype(np.float32)
                vec = cache.scaler.transform(vec.reshape(1, -1))[0]
                vec = np.nan_to_num(vec)
                x = torch.tensor(vec).unsqueeze(0).unsqueeze(1)
                with torch.no_grad():
                    prob = torch.sigmoid(spat_mod(x)).item()
                result["individual_predictions"]["spatial_risk"] = "HighRisk" if prob > 0.51 else "LowRisk"
                result["individual_predictions"]["spatial_score"] = round(prob, 2)
        else:
            result["individual_predictions"]["spatial_risk"] = "N/A"

        resp = {
            "text": result.get("text", "N/A"),
            "text_score": result["individual_predictions"].get("text_score", 0),
            "multilingual": result.get("multilingual", "N/A"),
            "audio": result.get("audio", "N/A"),
            "eeg": result.get("eeg", "N/A"),
            "eeg_score": result["individual_predictions"].get("eeg_score", 0),
            "spatial_risk": result["individual_predictions"].get("spatial_risk", "N/A"),
            "spatial_score": result["individual_predictions"].get("spatial_score", 0),
        }
        return {"predictions": resp}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --------------------------------------------------------------
# Vercel Handler
# --------------------------------------------------------------
def handler(event, context=None):
    from mangum import Mangum
    return Mangum(app, lifespan="off")(event, context)
