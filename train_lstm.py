#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, glob, time, argparse, hashlib
import numpy as np, pandas as pd, joblib
from typing import List, Tuple
from sklearn.preprocessing import MinMaxScaler

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

def _parse_cols(s: str) -> List[str]:
    cols = [c.strip() for c in s.split(",") if c.strip()]
    if not cols: raise ValueError("Empty column list")
    return cols

def _fp(paths: List[str]) -> str:
    h = hashlib.sha256()
    for p in sorted(paths):
        try:
            st = os.stat(p)
            h.update(p.encode('utf-8'))
            h.update(str(st.st_size).encode('utf-8'))
            h.update(str(int(st.st_mtime)).encode('utf-8'))
        except OSError:
            pass
    return h.hexdigest()

def _load_csv(path: str, needed: List[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    
    # Column mapping for UCwinRoad log files
    column_map = {
        'x': 'position X',
        'y': 'position Y', 
        'z': 'position Z',
        'custom_id': 'customID'
    }
    
    # Map columns to actual CSV column names
    actual_needed = []
    for col in needed:
        if col in column_map:
            actual_needed.append(column_map[col])
        else:
            actual_needed.append(col)
    
    miss = [c for c in actual_needed if c not in df.columns]
    if miss: raise ValueError(f"{path} missing {miss}")
    
    # Rename columns to expected names
    rename_map = {v: k for k, v in column_map.items()}
    df_selected = df[actual_needed].rename(columns=rename_map)
    
    # Convert customID: float -> int -> string (1003.0 -> 1003 -> "1003")
    if 'custom_id' in df_selected.columns:
        df_selected['custom_id'] = df_selected['custom_id'].astype(float).astype(int).astype(str)
    
    return df_selected

def build_dataset_by_id(pattern: str, cols: List[str], tcols: List[str], L: int, H: int, stride: int, diff: bool, custom_id: str) -> Tuple[np.ndarray, np.ndarray]:
    paths = sorted(glob.glob(pattern))
    if not paths: raise FileNotFoundError(f"No CSV match: {pattern}")
    keep = list(dict.fromkeys(cols + tcols + ['custom_id']))
    Xs, Ys = [], []
    
    print(f"Looking for customID: {custom_id}")
    
    for p in paths:
        try:
            df = _load_csv(p, keep)
            df['custom_id'] = df['custom_id'].astype(str)
            
            # Debug: print unique customID values
            unique_ids = df['custom_id'].unique()
            print(f"File {os.path.basename(p)}: customID values = {unique_ids}")
            
            df_filtered = df[df['custom_id'] == custom_id]
            print(f"  Filtered data for ID {custom_id}: {len(df_filtered)} records")
            
            if len(df_filtered) == 0: continue
            T = len(df_filtered)
            max_t = T - L - H
            if max_t <= 0: continue
            feat = df_filtered[cols].values
            targ = df_filtered[tcols].values
            for t in range(0, max_t + 1, stride):
                x = feat[t:t+L]
                if diff:
                    y = (targ[t+L-1+H] - targ[t+L-1]).astype(np.float32)
                else:
                    y = targ[t+L-1+H].astype(np.float32)
                Xs.append(x); Ys.append(y)
        except Exception as e:
            print(f"Skip {p}: {e}")
            continue
    
    print(f"Total sequences collected for ID {custom_id}: {len(Xs)}")
    
    if not Xs: raise RuntimeError(f"Empty dataset for ID {custom_id}")
    X = np.stack(Xs, axis=0).astype(np.float32)
    Y = np.stack(Ys, axis=0).astype(np.float32)
    return X, Y

def make_model(in_dim: int, out_dim: int, hidden: int, layers: int):
    m = Sequential()
    m.add(LSTM(hidden, return_sequences=(layers>1), input_shape=(None, in_dim)))
    for i in range(1, layers):
        last = (i == layers-1)
        m.add(LSTM(max(hidden//2, 16), return_sequences=not last))
    m.add(Dense(out_dim))
    m.compile(optimizer=Adam(1e-3), loss="huber", metrics=["mae"])
    return m

def train_model_for_id(X: np.ndarray, Y: np.ndarray, custom_id: str, in_dim: int, out_dim: int,
                       hidden: int, layers: int, epochs: int, bs: int, val_split: float, seed: int):
    sc = MinMaxScaler()
    Xf = X.reshape(-1, X.shape[-1])
    sc.fit(Xf)
    Xs = sc.transform(Xf).reshape(X.shape)
    tf.keras.utils.set_random_seed(seed)
    model = make_model(in_dim, out_dim, hidden, layers)
    model.fit(Xs, Y, batch_size=bs, epochs=epochs, validation_split=val_split, verbose=1)
    
    os.makedirs("models", exist_ok=True)
    model_name = "vehicle" if custom_id == "1003" else "person"
    model_path = f"models/{model_name}_lstm_model.h5"
    scaler_path = f"models/{model_name}_scaler.pkl"
    
    model.save(model_path)
    joblib.dump(sc, scaler_path)
    print(f"Saved {model_path}")
    print(f"Saved {scaler_path}")

def main():
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--data_dir", default=r"C:\UCwinRoad Data 17.2\Log")
    ap.add_argument("--cols", default="x,y,z")
    ap.add_argument("--target_cols", default="x,y,z")
    ap.add_argument("--L", type=int, default=5)
    ap.add_argument("--future_horizon", type=int, default=3)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--no_diff_target", action="store_true")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--val_split", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--interval_sec", type=int, default=300)
    args = ap.parse_args()

    pattern = os.path.join(args.data_dir, "*.csv")
    cols = _parse_cols(args.cols)
    tcols = _parse_cols(args.target_cols)
    in_dim, out_dim = len(cols), len(tcols)

    def _cycle():
        print(f"[scan] {pattern}")
        
        for custom_id, name in [("1003", "Vehicle"), ("1001", "Person")]:
            print(f"\n=== Training {name} (ID: {custom_id}) ===")
            try:
                X, Y = build_dataset_by_id(pattern, cols, tcols, args.L, args.future_horizon, args.stride, not args.no_diff_target, custom_id)
                print(f"Dataset: {X.shape[0]} sequences")
                train_model_for_id(X, Y, custom_id, in_dim, out_dim, args.hidden, args.layers, args.epochs, args.bs, args.val_split, args.seed)
                print(f"{name} model training completed")
            except Exception as e:
                print(f"{name} training failed: {e}")

    if not args.watch:
        _cycle()
        return

    last = ""
    while True:
        paths = glob.glob(pattern)
        fp = _fp(paths)
        if fp != last:
            print("[watch] change → retrain")
            _cycle()
            last = fp
        else:
            print("[watch] no change")
        time.sleep(args.interval_sec)

if __name__ == "__main__":
    main()