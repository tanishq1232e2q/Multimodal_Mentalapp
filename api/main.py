# --------------------------------------------------------------
# api/main.py   (Vercel serverless function)
# --------------------------------------------------------------
import os
import json
import zipfile
import tempfile
import shutil
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import mne
import scipy.io
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import StandardScaler

# ---------- GLOBAL CACHE (survives warm container) ----------
class Cache:
    models = {}
    data   = {}
    scaler = None

cache = Cache()

# ---------- FASTAPI APP ----------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- HELPERS ----------
def lazy_load_model(key: str, init_fn):
    """Load once, keep in memory."""
    if key not in cache.models:
        cache.models[key] = init_fn()
    return cache.models[key]

def quantize_model(model):
    """Dynamic 8-bit quantisation (only Linear layers)."""
    return torch.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )

# ---------- MODEL INITIALISERS ----------
def init_textbert():
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=8
    )
    path = "./checkpoints/textbert/kaggle/working/textbert_model_checkpoint/model.pt"
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return {"tokenizer": tokenizer, "model": quantize_model(model)}

def init_multilingual():
    dir_path = "./checkpoints/multilingual/multilingual_my_model_checkpoint"
    tokenizer = AutoTokenizer.from_pretrained(dir_path)
    model = AutoModelForSequenceClassification.from_pretrained(dir_path)
    model.eval()
    return {"tokenizer": tokenizer, "model": quantize_model(model)}

def init_eeg():
    class CNNGRU(nn.Module):
        def __init__(self, num_channels=129*5, num_classes=2):
            super().__init__()
            self.cnn = nn.Sequential(
                nn.Conv1d(num_channels, 128, 3, padding=1),
                nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.4),
                nn.Conv1d(128, 256, 3, padding=1),
                nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.4),
                nn.Conv1d(256, 512, 3, padding=1),
                nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.4),
            )
            self.gru = nn.GRU(512, 256, num_layers=3, batch_first=True, dropout=0.4)
            self.fc  = nn.Linear(256, num_classes)

        def forward(self, x):
            x = self.cnn(x)
            x = x.transpose(1, 2)
            _, h = self.gru(x)
            return self.fc(h[-1])

    model = CNNGRU()
    path = "./checkpoints/eeg/kaggle/working/eeg_model_checkpoint/model.pt"
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return quantize_model(model)

def init_audio():
    class GRUClassifier(nn.Module):
        def __init__(self, input_size=40, hidden_size=128, num_layers=3, num_classes=3):
            super().__init__()
            self.gru = nn.GRU(input_size, hidden_size, num_layers,
                             batch_first=True, dropout=0.3)
            self.bn  = nn.BatchNorm1d(hidden_size)
            self.fc  = nn.Linear(hidden_size, num_classes)

        def forward(self, x):
            h0 = torch.zeros(self.gru.num_layers, x.size(0), self.gru.hidden_size, device=x.device)
            out, _ = self.gru(x, h0)
            out = self.bn(out[:, -1, :])
            return self.fc(out)

    model = GRUClassifier()
    path = "./checkpoints/audio/kaggle/working/audio_model_checkpoint/model.pt"
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return quantize_model(model)

def init_spatial():
    class Spatial1DCNN(nn.Module):
        def __init__(self, input_length=272):
            super().__init__()
            self.conv1 = nn.Conv1d(1, 32, 3, padding=1)
            self.pool  = nn.MaxPool1d(2)
            self.conv2 = nn.Conv1d(32, 64, 3, padding=1)
            final_len = ((input_length + 1) // 2 + 1) // 2
            self.fc1   = nn.Linear(64 * final_len, 128)
            self.fc2   = nn.Linear(128, 1)
            self.drop  = nn.Dropout(0.5)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = self.pool(x)
            x = F.relu(self.conv2(x))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            x = self.drop(x)
            return torch.sigmoid(self.fc2(x))

    model = Spatial1DCNN()
    path = "./checkpoints/spatial/kaggle/working/spatial_model_checkpoint/model.pt"
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return quantize_model(model)

# ---------- DATA INITIALISERS ----------
def init_spatial_data():
    # Download only once
    import gdown, zipfile, os
    os.makedirs("./checkpoints/spatial-datas", exist_ok=True)
    zip_path = "./checkpoints/spatial-datas/dataset.zip"
    if not os.path.exists(zip_path):
        gdown.download(id="13g_HB5u_MuFZ7XMVaRwqRvuvsS9MZuju", output=zip_path, quiet=True)
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall("./checkpoints/spatial-datas")
        os.remove(zip_path)

    ahrf_paths = [
        "./checkpoints/spatial-datas/ahrf2023.csv",
        "./checkpoints/spatial-datas/ahrf2024_Feb2025.csv"
    ]
    years = [2023, 2024]
    centroids = pd.read_csv(
        "https://raw.githubusercontent.com/davidingold/county_centroids/master/county_centroids.csv"
    )

    dfs = []
    for p, y in zip(ahrf_paths, years):
        df = pd.read_csv(p, encoding="latin1", low_memory=False)
        # keep only needed columns (reduce RAM)
        keep = ['fips_st_cnty'] + [c for c in df.columns if any(k in c.lower() for k in
                     ['povty','emplymt','eductn','psych','phys_nf_prim_care'])]
        df = df[keep].copy()
        df['fips_st_cnty'] = df['fips_st_cnty'].astype(str).str.zfill(5)
        df['high_mh_disorder'] = (pd.to_numeric(df.iloc[:,1], errors='coerce') > 0).astype(int)
        df = df.dropna(subset=['high_mh_disorder'])
        df['year'] = y
        centroids['GEOID'] = centroids['GEOID'].astype(str).str.zfill(5)
        df = df.merge(centroids[['GEOID','x','y']], left_on='fips_st_cnty',
                      right_on='GEOID', how='left')
        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True)
    feat = [c for c in merged.columns if c not in
            ['fips_st_cnty','st_name','cnty_name','GEOID','x','y','high_mh_disorder','year']]
    scaler = StandardScaler()
    scaler.fit(merged[feat].values.astype(np.float32))
    cache.scaler = scaler
    cache.data['spatial'] = merged
    cache.data['feat_cols'] = feat

# ---------- PREDICTION HELPERS ----------
def extract_psd(mat_path):
    mat = scipy.io.loadmat(mat_path)
    key = next(k for k in mat.keys() if k not in
               {'__header__','__version__','__globals__','samplingRate','Impedances_0','DIN_1'})
    eeg = mat[key]
    if eeg.ndim > 2: eeg = np.squeeze(eeg)
    if eeg.shape[1] == 129*5: eeg = eeg.T
    info = mne.create_info([f"ch{i}" for i in range(129*5)], sfreq=250, ch_types="eeg")
    raw  = mne.io.RawArray(eeg, info, verbose=False)

    bands = [(0.5,4),(4,8),(8,13),(13,30),(30,50)]
    psds  = []
    for fmin,fmax in bands:
        psd,_ = mne.time_frequency.psd_array_welch(raw.get_data(), sfreq=250,
                     fmin=fmin, fmax=fmax, n_fft=256, verbose=False)
        psds.append(psd.mean(axis=1))
    feats = np.concatenate(psds)
    # normalise with pre-computed stats (you must upload the .npz)
    stats = np.load("./checkpoints/eeg_preprocessing_stats.npz")
    feats = (feats - stats['mean']) / (stats['std'] + 1e-8)
    return torch.tensor(feats, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)

# ---------- MAIN PREDICT ----------
@app.post("/predict")
async def predict(
    text: Optional[str] = Form(None),
    multilingual_text: Optional[str] = Form(None),
    audio: Optional[UploadFile] = File(None),
    eeg: Optional[UploadFile] = File(None),
    fips_code: Optional[str] = Form(None)
):
    try:
        # ---------- LAZY LOAD EVERYTHING ----------
        textbert = lazy_load_model("textbert", init_textbert)
        multi    = lazy_load_model("multilingual", init_multilingual)
        eeg_mod  = lazy_load_model("eeg", init_eeg)
        audio_mod= lazy_load_model("audio", init_audio)
        spat_mod = lazy_load_model("spatial", init_spatial)
        if "spatial" not in cache.data:
            init_spatial_data()

        # ---------- FILE HANDLING ----------
        audio_path = eeg_path = None
        if audio:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                shutil.copyfileobj(audio.file, f)
                audio_path = f.name
        if eeg:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mat") as f:
                shutil.copyfileobj(eeg.file, f)
                eeg_path = f.name

        # ---------- SPATIAL ----------
        spatial_input = None
        if fips_code:
            fips = str(fips_code).zfill(5)
            row  = cache.data['spatial'][cache.data['spatial']['fips_st_cnty'] == fips]
            if row.empty:
                raise ValueError("FIPS not found")
            vec = row.iloc[0][cache.data['feat_cols']].values.astype(np.float32)
            vec = cache.scaler.transform(vec.reshape(1, -1))[0]
            vec = np.nan_to_num(vec)
            spatial_input = vec.tolist()

        # ---------- TEXT ----------
        result = {"individual_predictions": {}}
        if text:
            enc = textbert["tokenizer"](text, truncation=True, padding="max_length",
                                        max_length=128, return_tensors="pt")
            with torch.no_grad():
                logits = textbert["model"](**enc).logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            classes = ['adhd','anxiety','autism','bipolar','bpd','depression','ptsd','schizophrenia']
            pred_idx = int(np.argmax(probs))
            pred = classes[pred_idx]
            score = float(probs[pred_idx])
            if pred == "depression" and score < 0.65:
                pred = "Normal"
            result["text"] = pred
            result["individual_predictions"]["text_score"] = round(score, 2)

        # ---------- MULTILINGUAL ----------
        inp = multilingual_text or text
        if inp:
            enc = multi["tokenizer"](inp, truncation=True, padding="max_length",
                                    max_length=128, return_tensors="pt")
            with torch.no_grad():
                logits = multi["model"](**enc).logits
            prob = torch.softmax(logits, dim=1)[0,1].item()
            result["multilingual"] = "Depression" if prob > 0.5 else "Normal"
        else:
            result["multilingual"] = "N/A"

        # ---------- AUDIO ----------
        if audio_path:
            y, _ = librosa.load(audio_path, sr=16000)
            mfcc = librosa.feature.mfcc(y=y, sr=16000, n_mfcc=40)
            if mfcc.shape[1] < 100:
                mfcc = np.pad(mfcc, ((0,0),(0,100-mfcc.shape[1])))
            else:
                mfcc = mfcc[:,:100]
            x = torch.tensor(mfcc.T, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits = audio_mod(x)
            cls = ['normal','mild_depression','severe_depression']
            result["audio"] = cls[torch.argmax(logits, dim=1).item()]
        else:
            result["audio"] = "N/A"

        # ---------- EEG ----------
        if eeg_path:
            x = extract_psd(eeg_path)
            with torch.no_grad():
                logits = eeg_mod(x)
            prob = torch.softmax(logits, dim=1)[0,1].item()
            result["eeg"] = "MDD(Major Depressive Disorder)" if prob > 0.5 else "HC(Healthy Control)"
            result["individual_predictions"]["eeg_score"] = round(prob, 2)
        else:
            result["eeg"] = "N/A"
            result["individual_predictions"]["eeg_score"] = 0.0

        # ---------- SPATIAL ----------
        if spatial_input is not None:
            arr = np.array(spatial_input, dtype=np.float32)
            if arr.shape[0] != 272:
                pad = 272 - arr.shape[0]
                arr = np.pad(arr, (0, max(pad,0))) if pad>0 else arr[:272]
            x = torch.tensor(arr).unsqueeze(0).unsqueeze(1)
            with torch.no_grad():
                prob = torch.sigmoid(spat_mod(x)).item()
            result["individual_predictions"]["spatial_score"] = round(prob, 2)
            result["individual_predictions"]["spatial_risk"] = "HighRisk" if prob>0.51 else "LowRisk"
        else:
            result["individual_predictions"]["spatial_risk"] = "N/A"
            result["individual_predictions"]["spatial_score"] = 0.0

        # ---------- CLEANUP ----------
        for p in (audio_path, eeg_path):
            if p and os.path.exists(p):
                os.unlink(p)

        # ---------- RESPONSE ----------
        resp = {
            "text": result.get("text","N/A"),
            "text_score": result["individual_predictions"].get("text_score",0),
            "multilingual": result.get("multilingual","N/A"),
            "audio": result.get("audio","N/A"),
            "eeg": result.get("eeg","N/A"),
            "eeg_score": result["individual_predictions"].get("eeg_score",0.0),
            "spatial_risk": result["individual_predictions"].get("spatial_risk","N/A"),
            "spatial_score": result["individual_predictions"].get("spatial_score",0.0),
        }
        return {"predictions": resp}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --------------------------------------------------------------
# Vercel entry point
# --------------------------------------------------------------
def handler(event, context=None):
    from mangum import Mangum
    return Mangum(app)(event, context)