import argparse
import os
import random
import time
import warnings
import re
from typing import List, Tuple, Optional
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, auc
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from datetime import datetime
import json
import torch.nn.functional as F
from scipy import stats
import seaborn as sns
from tqdm import tqdm

try:
    import optuna
    OPTUNA_AVAILABLE = True
except Exception:
    OPTUNA_AVAILABLE = False

class Config:
    ESM2_MODEL = "facebook/esm2_t33_650M_UR50D" 
    BATCH_SIZE = 64
    EPOCHS = 50
    LR = 5e-5
    WEIGHT_DECAY = 5e-6
    DEVICE = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu') 
    SAVE_PATH = 'best_model-adaptive.pt'

    STRUCTURE_DIM = 256
    ESM_DIM = 1280

    GAUSS_STD = 0.01
    DROP_PROB = 0.05
    MAX_CROP_FRAC = 0.08

    CONV_FILTERS = 128
    LSTM_UNITS = 128
    DROPOUT_LSTM = 0.58
    DROPOUT_HEAD = 0.3
    SPATIAL_DROPOUT = 0.3

    CLIP_NORM = 1.0
    LABEL_SMOOTH_EPS = 0.01
    PATIENCE = 6

    USE_SMOTE = False
    USE_FOCAL = True
    FOCAL_ALPHA = 0.8
    FOCAL_GAMMA = 2.0
    USE_CLASS_WEIGHT = True

    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.1
    TEST_RATIO = 0.2

    IGNORE_ID_SUFFIX = True 
    DEBUG_MODE = True 

def parse_gnn_filename(filename: str) -> Tuple[str, str]:
    try:
        fname = filename.replace('.csv', '').strip()
        parts = fname.split('_')
        
        prefix_len = 2 if parts[0] == 'hbond' else 0
        remaining_parts = parts[prefix_len:]

        id_parts = []
        site_parts = []
        id_complete = False
        
        for part in remaining_parts:
            if not id_complete:
                if '-' in part or len(part) > 5 or (any(c.isalpha() for c in part) and any(c.isdigit() for c in part) and len(id_parts) > 0):
                    id_parts.append(part)
                else:
                    id_complete = True
                    site_parts.append(part)
            else:
                site_parts.append(part)
        
        protein_id = '_'.join(id_parts).strip().upper()
        if Config.IGNORE_ID_SUFFIX and '-' in protein_id:
            protein_id = protein_id.split('-')[0]

        site = ""
        if len(site_parts) >= 2:
            chain = site_parts[0].strip().upper()
            pos = ''.join(site_parts[1:]).strip()
            site = f"{chain}:{pos}"
        elif len(site_parts) == 1:
            site = f":{site_parts[0]}" if site_parts[0].isdigit() else site_parts[0].upper()
        
        return protein_id, site
    except Exception as e:
        warnings.warn(f"Parsing of the GNN file name {filename} failed: {str(e)}")
        return "", ""

def standardize_seq_site(site_str: str) -> str:
    try:
        s = str(site_str).strip().upper()
        s = s.replace('_', ':').replace('-', ':').replace(' ', ':')
        if re.match(r'^[A-Z]\d+$', s):
            s = f"{s[0]}:{s[1:]}"
        elif re.match(r'^\d+$', s):
            s = f":{s}"
        return s if s not in ["", ":"] else ""
    except Exception as e:
        warnings.warn(f"standardizing Site {site_str} failed: {str(e)}")
        return ""

def standardize_seq_id(id_str: str) -> str:
    try:
        pid = str(id_str).strip().upper()
        if Config.IGNORE_ID_SUFFIX and '-' in pid:
            pid = pid.split('-')[0]
        return pid
    except Exception as e:
        warnings.warn(f"standardizing ID {id_str} failed: {str(e)}")
        return ""

def load_gnn_features(gnn_csv_paths: list) -> pd.DataFrame:
    try:
        gnn_dfs = []
        for csv_path in gnn_csv_paths:
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"GNN file not exist: {csv_path}")
            
            df = pd.read_csv(csv_path)
            if 'file_name' not in df.columns:
                raise ValueError(f"GNN file {csv_path} lack file_name col")
            
            df[['protein_id', 'site']] = df['file_name'].apply(
                lambda x: pd.Series(parse_gnn_filename(x))
            )
            gnn_dfs.append(df)
        
        gnn_df = pd.concat(gnn_dfs, ignore_index=True)
        
        gnn_df = gnn_df[
            (gnn_df['protein_id'] != "") & 
            (gnn_df['site'] != "")
        ].reset_index(drop=True)
        
        feature_cols = [str(i) for i in range(Config.STRUCTURE_DIM)]
        missing_cols = [col for col in feature_cols if col not in gnn_df.columns]
        if missing_cols:
            raise ValueError(f"GNN feature lack col: {missing_cols}")
        
        gnn_df = gnn_df[['protein_id', 'site'] + feature_cols].dropna()
        gnn_df = gnn_df.drop_duplicates(subset=['protein_id', 'site'], keep='first')
        
        if Config.DEBUG_MODE:
            print(f"\n=== GNN Feature Statistics ===")
            print(f"Total samples: {len(gnn_df)}")
            print(f"Unique ID number: {gnn_df['protein_id'].nunique()}")
            print(f"The total number of sites: {gnn_df['site'].nunique()}")
        
        return gnn_df
    except Exception as e:
        raise RuntimeError(f"Failed to load GNN features: {str(e)}")

def load_sequence_data(pos_seq_csv: str, neg_seq_csv: str, seq_col: str = 'Fragment') -> pd.DataFrame:
    try:
        pos_df = pd.read_csv(pos_seq_csv)
        neg_df = pd.read_csv(neg_seq_csv)
        pos_df['label'] = 1
        neg_df['label'] = 0

        required_cols = [seq_col, 'ID', 'Site']
        for df, name in [(pos_df, "Positive sample sequence"), (neg_df, "Negative sample sequence")]:
            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                raise ValueError(f"{name} CSV lack col: {missing}")
        
        seq_df = pd.concat([pos_df, neg_df], ignore_index=True)
        seq_df['protein_id'] = seq_df['ID'].apply(standardize_seq_id)
        seq_df['site'] = seq_df['Site'].apply(standardize_seq_site)
        
        seq_df = seq_df[
            (seq_df['protein_id'] != "") & 
            (seq_df['site'] != "") & 
            (seq_df[seq_col].notna())
        ].reset_index(drop=True)

        seq_df = seq_df.drop_duplicates(subset=['protein_id', 'site'], keep='first')
        
        if Config.DEBUG_MODE:
            print(f"\n=== Sequence Data Statistics ===")
            print(f"Total samples: {len(seq_df)}")
            print(f"Positive samples: {sum(seq_df['label']==1)}")
            print(f"Negative samples: {sum(seq_df['label']==0)}")
            print(f"Unique ID number: {seq_df['protein_id'].nunique()}")
            print(f"Unique Site number: {seq_df['site'].nunique()}")
        
        return seq_df[[seq_col, 'protein_id', 'site', 'label']]
    except Exception as e:
        raise RuntimeError(f"Failed to load sequence data: {str(e)}")

def merge_sequence_gnn(seq_df: pd.DataFrame, gnn_df: pd.DataFrame) -> pd.DataFrame:
    seq_df['match_key'] = seq_df['protein_id'] + '|' + seq_df['site']
    gnn_df['match_key'] = gnn_df['protein_id'] + '|' + gnn_df['site']
    
    if Config.DEBUG_MODE:
        seq_keys = set(seq_df['match_key'])
        gnn_keys = set(gnn_df['match_key'])
        common_keys = seq_keys & gnn_keys
        
        print(f"\n=== Matching Analysis ===")
        print(f"Number of unique keys in sequence data: {len(seq_keys)}")
        print(f"Number of unique keys in GNN data: {len(gnn_keys)}")
        print(f"Number of common matching keys: {len(common_keys)}")
        
        if len(common_keys) == 0:
            raise ValueError(
                "No matching items between sequence data and GNN features! Please check:\n"
                "1. Whether ID/Site formats are consistent (case/separators)\n"
                "2. Whether IGNORE_ID_SUFFIX parameter is enabled\n"
                "3. Whether sequence and GNN files correspond to the same batch of samples"
            )
    
    merged_df = pd.merge(
        seq_df, gnn_df,
        on=['match_key', 'protein_id', 'site'],
        how='inner'
    ).drop('match_key', axis=1)
    
    if Config.DEBUG_MODE:
        print(f"\n=== Merged Data Statistics ===")
        print(f"Valid samples: {len(merged_df)}")
        print(f"Positive samples: {sum(merged_df['label']==1)}")
        print(f"Negative samples: {sum(merged_df['label']==0)}")
    
    return merged_df

from transformers import AutoTokenizer, AutoModelForMaskedLM

def load_esm2_model(model_name: str, device: torch.device):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForMaskedLM.from_pretrained(model_name)
        model = model.to(device)
        model.eval()
        return tokenizer, model
    except Exception as e:
        raise RuntimeError(f"Failed to load ESM2 model: {str(e)}")

def pad_features(features: np.ndarray, maxlen: int) -> np.ndarray:
    N, L, D = features.shape
    out = np.zeros((N, maxlen, D), dtype=features.dtype)
    length = min(L, maxlen)
    out[:, :length, :] = features[:, :length, :]
    return out

def extract_esm2_residue_features(sequences: List[str], tokenizer, model, max_len: int, batch_size: int = 8, device: torch.device = Config.DEVICE):
    features = []
    for i in tqdm(range(0, len(sequences), batch_size), desc='Extracting ESM2 features'):
        batch = sequences[i:i+batch_size]
        try:
            inputs = tokenizer(
                batch, 
                add_special_tokens=True, 
                padding='max_length', 
                truncation=True, 
                max_length=max_len+2,
                return_tensors='pt'
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                last_hidden = outputs.hidden_states[-1]
            
            residue_feat = last_hidden[:, 1:-1, :].cpu().numpy()
            features.append(residue_feat)
            
            del inputs, outputs, last_hidden
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        except Exception as e:
            warnings.warn(f"Processing batch {i//batch_size + 1} failed: {str(e)}, padding with 0")
            dummy = np.zeros((len(batch), max_len, Config.ESM_DIM), dtype=np.float32)
            features.append(dummy)
    
    features = np.concatenate(features, axis=0)
    if features.shape[1] != max_len:
        features = pad_features(features, maxlen=max_len)
    
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    
    return features

def load_gnn_structure_features(merged_df: pd.DataFrame) -> np.ndarray:
    feature_cols = [str(i) for i in range(Config.STRUCTURE_DIM)]
    try:
        struct_features = merged_df[feature_cols].values.astype(np.float32)
        
        if struct_features.shape[1] != Config.STRUCTURE_DIM:
            raise ValueError(f"GNN feature dimension should be {Config.STRUCTURE_DIM}, actual is {struct_features.shape[1]}")
        
        struct_features = (struct_features - struct_features.mean(axis=0)) / (struct_features.std(axis=0) + 1e-8)
        
        print(f"\nGNN features loaded successfully: shape {struct_features.shape}")
        return struct_features
    except Exception as e:
        raise RuntimeError(f"Failed to load GNN features: {str(e)}")

class ESM2Dataset(Dataset):
    def __init__(self, X_esm: np.ndarray, X_struct: np.ndarray, y: np.ndarray, 
                 augment: bool = False):
        self.X_esm = X_esm.astype('float32') 
        self.X_struct = X_struct.astype('float32') 
        self.y = y.astype('float32')
        self.augment = augment
        
        if self.X_esm.shape[0] != self.X_struct.shape[0] or self.X_esm.shape[0] != len(self.y):
             raise ValueError("Number of samples in ESM features, GNN features and labels do not match")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x_esm = self.X_esm[idx].copy()
        x_struct = self.X_struct[idx].copy()
        y = self.y[idx]
        
        if self.augment:
            L, D = x_esm.shape
            pad_mask = np.all(np.isclose(x_esm, 0.0), axis=1)
            
            noise = np.random.normal(0, Config.GAUSS_STD, size=x_esm.shape).astype('float32')
            noise[pad_mask] = 0.0
            x_esm += noise
            
            if 0.0 < Config.DROP_PROB < 1.0:
                drop_mask = np.random.rand(L) < Config.DROP_PROB
                drop_mask = np.logical_and(drop_mask, ~pad_mask)
                x_esm[drop_mask] = 0.0
            
            if Config.MAX_CROP_FRAC > 0:
                max_crop = max(1, int(L * Config.MAX_CROP_FRAC))
                crop = np.random.randint(-max_crop, max_crop + 1)
                if crop > 0:
                    x_esm = np.vstack([x_esm[crop:], np.zeros((crop, D), dtype='float32')])
                elif crop < 0:
                    crop = -crop
                    x_esm = np.vstack([np.zeros((crop, D), dtype='float32'), x_esm[:-crop]])
            
            scale = 1.0 + np.random.uniform(-0.03, 0.03)
            x_esm *= scale
            x_esm = np.nan_to_num(x_esm, nan=0.0, posinf=0.0, neginf=0.0)
        
        return (torch.tensor(x_esm), torch.tensor(x_struct)), torch.tensor(y)

class ESM2_CNN_BiLSTM(nn.Module):
    def __init__(self, seq_len: int, esm_dim: int = Config.ESM_DIM, struct_dim: int = Config.STRUCTURE_DIM):
        super().__init__()
        
        self.seq_len = seq_len
        self.esm_dim = esm_dim
        self.struct_dim = struct_dim
        self.fused_dim = esm_dim + struct_dim

        self.view_weight_layer = nn.Sequential(
            nn.Linear(self.fused_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)
        )

        self.conv = nn.Conv1d(in_channels=self.fused_dim, out_channels=Config.CONV_FILTERS, kernel_size=5, padding=2)
        self.bn_conv = nn.BatchNorm1d(Config.CONV_FILTERS)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.spatial_dropout = nn.Dropout2d(Config.SPATIAL_DROPOUT)

        self.lstm = nn.LSTM(
            input_size=Config.CONV_FILTERS, 
            hidden_size=Config.LSTM_UNITS, 
            num_layers=1, 
            batch_first=True, 
            bidirectional=True,
            dropout=Config.DROPOUT_LSTM if Config.LSTM_UNITS > 1 else 0
        )
        self.lstm_dropout = nn.Dropout(Config.DROPOUT_LSTM)

        self.att_linear = nn.Linear(2 * Config.LSTM_UNITS, Config.LSTM_UNITS)
        self.att_tanh = nn.Tanh()
        self.att_score = nn.Linear(Config.LSTM_UNITS, 1)

        self.classifier = nn.Sequential(
            nn.Dropout(Config.DROPOUT_HEAD),
            nn.Linear(2 * Config.LSTM_UNITS, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(Config.DROPOUT_HEAD),
            nn.Linear(64, 1)
        )
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0)

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor], return_weights: bool = False):
        x_esm, x_struct = inputs
        B, L, D_esm = x_esm.shape

        x_struct_seq = x_struct.unsqueeze(1).repeat(1, self.seq_len, 1)
        x_esm_global = x_esm.mean(dim=1)
        x_global_fused = torch.cat([x_esm_global, x_struct], dim=1)
        
        view_logits = self.view_weight_layer(x_global_fused)
        view_weights = F.softmax(view_logits, dim=1)
        
        alpha1 = view_weights[:, 0].unsqueeze(1).unsqueeze(2)
        alpha2 = view_weights[:, 1].unsqueeze(1).unsqueeze(2)
        x_esm_weighted = x_esm * alpha1
        x_struct_weighted = x_struct_seq * alpha2
        
        x = torch.cat([x_esm_weighted, x_struct_weighted], dim=2)
        
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.bn_conv(x)
        x = self.relu(x)
        x = self.spatial_dropout(x.unsqueeze(-1)).squeeze(-1)
        x = self.pool(x)

        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        lstm_out = self.lstm_dropout(lstm_out)

        att = self.att_linear(lstm_out)
        att = self.att_tanh(att)
        att = self.att_score(att)
        att_weights = torch.softmax(att, dim=1)
        att_out = torch.sum(lstm_out * att_weights, dim=1)

        logits = self.classifier(att_out).squeeze(1)
        
        if return_weights:
            return logits, view_weights
        return logits

class FocalLoss(nn.Module):
    def __init__(self, alpha: float = Config.FOCAL_ALPHA, gamma: float = Config.FOCAL_GAMMA, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        pt = torch.exp(-bce_loss)
        alpha_tensor = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        focal_loss = alpha_tensor * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

def compute_auc(y_true, y_pred):
    try:
        if len(np.unique(y_true)) < 2:
            return 0.5
        return float(roc_auc_score(y_true, y_pred))
    except Exception:
        return 0.5

def find_optimal_threshold(y_true, y_prob):
    try:
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
        f1s = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-8)
        return thresholds[np.argmax(f1s)] if len(f1s) > 0 else 0.5
    except Exception:
        return 0.5

def evaluate_model(model, loader, device, criterion=None, threshold=0.5, collect_weights=False):
    model.eval()
    ys = []
    y_probs = []
    total_loss = 0.0
    n = 0
    view_weights_list = []
    
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, matthews_corrcoef
    
    with torch.no_grad():
        for inputs, yb in loader:
            Xb_esm, Xb_struct = inputs
            Xb_esm = Xb_esm.to(device)
            Xb_struct = Xb_struct.to(device)
            yb = yb.to(device)
            
            if collect_weights:
                logits, view_weights = model((Xb_esm, Xb_struct), return_weights=True)
                view_weights_list.append(view_weights.cpu().numpy())
            else:
                logits = model((Xb_esm, Xb_struct))
            
            if criterion is not None:
                loss = criterion(logits, yb)
                total_loss += loss.item() * Xb_esm.size(0)
            
            probs = torch.sigmoid(logits).cpu().numpy()
            y_probs.extend(probs.tolist())
            ys.extend(yb.cpu().numpy().tolist())
            n += Xb_esm.size(0)
    
    avg_loss = total_loss / n if n>0 else 0.0
    y_probs = np.array(y_probs)
    ys = np.array(ys)
    y_pred = (y_probs > threshold).astype(int)
    
    auc = compute_auc(ys, y_probs)
    acc = accuracy_score(ys, y_pred)
    f1 = f1_score(ys, y_pred, zero_division=0)
    precision = precision_score(ys, y_pred, zero_division=0)
    recall = recall_score(ys, y_pred, zero_division=0)
    
    cm = confusion_matrix(ys, y_pred, labels=[0,1]) if len(np.unique(ys))>1 else np.array([[0,0],[0,0]])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0,0,0,0)
    specificity = tn/(tn+fp) if (tn+fp)>0 else 0.0
    mcc = matthews_corrcoef(ys, y_pred) if len(np.unique(y_pred))>1 and len(np.unique(ys))>1 else 0.0
    
    result = {
        'auc': auc, 'acc': acc, 'f1': f1, 'precision': precision, 
        'recall': recall, 'specificity': specificity, 'mcc': mcc, 
        'loss': avg_loss, 'y_probs': y_probs, 'y_true': ys,
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp
    }
    
    if collect_weights and view_weights_list:
        result['view_weights'] = np.concatenate(view_weights_list, axis=0)
    
    return result

def train_one_fold(model, train_loader, val_loader, device, test_loader=None, fold_idx=0):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    criterion = FocalLoss() if Config.USE_FOCAL else nn.BCEWithLogitsLoss()

    best_val_auc = 0.0
    best_state = None
    history = {'train_loss':[], 'val_loss':[], 'train_auc':[], 'val_auc':[]}

    for epoch in range(1, Config.EPOCHS+1):
        model.train()
        losses = []
        preds = []
        targets = []
        
        for inputs, yb in train_loader:
            Xb_esm, Xb_struct = inputs
            Xb_esm = Xb_esm.to(device)
            Xb_struct = Xb_struct.to(device)
            yb = yb.to(device)
            
            optimizer.zero_grad()
            logits = model((Xb_esm, Xb_struct))
            yb_smoothed = yb * (1.0 - Config.LABEL_SMOOTH_EPS) + 0.5 * Config.LABEL_SMOOTH_EPS
            loss = criterion(logits, yb_smoothed)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.CLIP_NORM)
            optimizer.step()
            
            losses.append(loss.item())
            preds.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())
            targets.extend(yb.detach().cpu().numpy().tolist())
        
        train_loss = float(np.mean(losses)) if len(losses)>0 else 0.0
        train_auc = compute_auc(np.array(targets), np.array(preds))

        val_metrics = evaluate_model(model, val_loader, device, criterion)
        val_auc = val_metrics['auc']
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['train_auc'].append(train_auc)
        history['val_auc'].append(val_auc)
        
        scheduler.step(val_auc)

        print(f"Fold {fold_idx} | Epoch {epoch}/{Config.EPOCHS} "
              f"| Train Loss: {train_loss:.4f} | Validation Loss: {val_metrics['loss']:.4f} "
              f"| Train AUC: {train_auc:.4f} | Validation AUC: {val_auc:.4f}")

        if val_auc > best_val_auc + 1e-4:
            best_val_auc = val_auc
            best_state = {k:v.cpu() for k,v in model.state_dict().items()}
            save_path = Config.SAVE_PATH.replace('.pt', f'_fold{fold_idx}.pt') if fold_idx > 0 else Config.SAVE_PATH
            torch.save(best_state, save_path)
            print(f"Fold {fold_idx} | Best model saved: {save_path}")

        if epoch - np.argmax(np.array(history['val_auc'])) > Config.PATIENCE:
            print(f"Fold {fold_idx} | Early stopping triggered (validation AUC no longer improves)")
            break

    test_metrics = None
    if best_state is not None:
        model.load_state_dict(best_state)
        if test_loader is not None:
            val_metrics_best = evaluate_model(model, val_loader, device, criterion)
            optimal_threshold = find_optimal_threshold(val_metrics_best['y_true'], val_metrics_best['y_probs'])
            test_metrics = evaluate_model(model, test_loader, device, criterion, 
                                         threshold=optimal_threshold, collect_weights=True)
            print(f"\nFold {fold_idx} | Test set metrics (optimal threshold {optimal_threshold:.4f}):")
            print(f"  AUC: {test_metrics['auc']:.4f} | F1: {test_metrics['f1']:.4f} | MCC: {test_metrics['mcc']:.4f}")
            print(f"  Accuracy: {test_metrics['acc']:.4f} | Recall: {test_metrics['recall']:.4f} | Specificity: {test_metrics['specificity']:.4f}")

    return model, history, test_metrics

def plot_test_roc_curve(y_true, y_probs, save_path: str):
    try:
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title('Test ROC Curve')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"ROC curve saved: {save_path}")
    except Exception as e:
        warnings.warn(f"Failed to plot ROC curve: {str(e)}")

def plot_view_weight_kde(view_weights: np.ndarray, save_path: str):
    try:
        esm_weights = view_weights[:, 0]
        gnn_weights = view_weights[:, 1]
        
        plt.figure(figsize=(10, 6))
        sns.kdeplot(esm_weights, label='ESM view weight', fill=True, alpha=0.5, linewidth=2, color='blue')
        sns.kdeplot(gnn_weights, label='GNN view weight', fill=True, alpha=0.5, linewidth=2, color='orange')
        
        plt.axvline(np.mean(esm_weights), color='blue', linestyle='--', 
                    label=f'ESM Mean value: {np.mean(esm_weights):.4f}')
        plt.axvline(np.mean(gnn_weights), color='orange', linestyle='--', 
                    label=f'GNN Mean value: {np.mean(gnn_weights):.4f}')
        
        plt.xlabel('Weight')
        plt.ylabel('Density')
        plt.title('Weight distribution for multi-view')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"View weight distribution plot saved: {save_path}")
    except Exception as e:
        warnings.warn(f"Failed to plot weight distribution: {str(e)}")

def save_metrics_to_json(metrics_dict, save_path):
    try:
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(i) for i in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            return obj
        
        metrics_converted = convert(metrics_dict)
        metrics_converted['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(metrics_converted, f, ensure_ascii=False, indent=2)
        print(f"Metrics saved: {save_path}")
    except Exception as e:
        warnings.warn(f"Failed to save metrics: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='ESM2+GNN multi-view fusion training')
    parser.add_argument('--pos_seq_csv', type=str, required=True, help='Positive sample sequence CSV path')
    parser.add_argument('--neg_seq_csv', type=str, required=True, help='Negative sample sequence CSV path')
    parser.add_argument('--pos_gnn_csv', type=str, required=True, help='Positive sample GNN feature CSV path')
    parser.add_argument('--neg_gnn_csv', type=str, required=True, help='Negative sample GNN feature CSV path')
    parser.add_argument('--seq_col', type=str, default='Fragment', help='Sequence column name')
    parser.add_argument('--mode', type=str, default='train', choices=['train','optuna','kfold'], help='Running mode')
    parser.add_argument('--n_trials', type=int, default=20, help='Optuna trial number')
    parser.add_argument('--n_splits', type=int, default=5, help='K-fold number')
    parser.add_argument('--max_len', type=int, default=31, help='Maximum sequence length')
    parser.add_argument('--save_dir', type=str, default='./results_adaptive_gnn', help='Result save directory')
    parser.add_argument('--ignore_id_suffix', action='store_true', help='Whether to ignore hyphen suffix in ID')
    args = parser.parse_args()

    Config.IGNORE_ID_SUFFIX = args.ignore_id_suffix
    
    os.makedirs(args.save_dir, exist_ok=True)
    metrics_dir = os.path.join(args.save_dir, 'metrics')
    fig_dir = os.path.join(args.save_dir, 'figures')
    model_dir = os.path.join(args.save_dir, 'models')
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    Config.SAVE_PATH = os.path.join(model_dir, 'best_model_adaptive_gnn.pt')

    print("=== Start data loading and preprocessing ===")
    
    seq_df = load_sequence_data(args.pos_seq_csv, args.neg_seq_csv, args.seq_col)
    
    gnn_df = load_gnn_features([args.pos_gnn_csv, args.neg_gnn_csv])
    
    merged_df = merge_sequence_gnn(seq_df, gnn_df)
    
    sequences = merged_df[args.seq_col].astype(str).tolist()
    labels = merged_df['label'].values.astype(int)
    print(f"\nValid samples: {len(sequences)} (Positive: {sum(labels)}, Negative: {len(labels)-sum(labels)})")

    print("\n=== Load ESM2 model and extract features ===")
    tokenizer, esm2_model = load_esm2_model(Config.ESM2_MODEL, Config.DEVICE)
    X_esm = extract_esm2_residue_features(sequences, tokenizer, esm2_model, max_len=args.max_len)
    print(f"ESM2 feature shape: {X_esm.shape} (Samples × Sequence length × Feature dimension)")
    
    X_struct = load_gnn_structure_features(merged_df)

    print("\n=== Split dataset (7:1:2) ===")
    X_train_esm, X_temp_esm, y_train, y_temp = train_test_split(
        X_esm, labels, test_size=1-Config.TRAIN_RATIO, random_state=42, stratify=labels)
    X_train_struct, X_temp_struct, _, _ = train_test_split(
        X_struct, labels, test_size=1-Config.TRAIN_RATIO, random_state=42, stratify=labels)
    
    val_ratio = Config.VAL_RATIO / (Config.VAL_RATIO + Config.TEST_RATIO)
    X_val_esm, X_test_esm, y_val, y_test = train_test_split(
        X_temp_esm, y_temp, test_size=1-val_ratio, random_state=42, stratify=y_temp)
    X_val_struct, X_test_struct, _, _ = train_test_split(
        X_temp_struct, y_temp, test_size=1-val_ratio, random_state=42, stratify=y_temp)
    
    print(f"Training set: {len(X_train_esm)} samples ({Config.TRAIN_RATIO*100:.0f}%)")
    print(f"Validation set: {len(X_val_esm)} samples ({Config.VAL_RATIO*100:.0f}%)")
    print(f"Test set: {len(X_test_esm)} samples ({Config.TEST_RATIO*100:.0f}%)")

    seq_len = X_esm.shape[1]
    esm_dim = X_esm.shape[2]
    struct_dim = X_struct.shape[1]

    if args.mode == 'optuna':
        if not OPTUNA_AVAILABLE:
            raise RuntimeError('Optuna is not installed, please run: pip install optuna')
        
        print("\n=== Start Optuna hyperparameter search ===")
        def objective(trial):
            conv = trial.suggest_categorical('conv_filters', [16,32,64,128])
            lstm = trial.suggest_categorical('lstm_units', [32,64,128])
            dropout_lstm = trial.suggest_float('dropout_lstm', 0.2, 0.6)
            dropout_head = trial.suggest_float('dropout_head', 0.2, 0.6)
            spatial_dropout = trial.suggest_float('spatial_dropout', 0.1, 0.4)
            lr = trial.suggest_loguniform('lr', 1e-5, 1e-3)
            weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-3)
            
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            scores = []
            
            for train_idx, val_idx in skf.split(X_train_esm, y_train):
                Xtr_esm, Xv_esm = X_train_esm[train_idx], X_train_esm[val_idx]
                Xtr_struct, Xv_struct = X_train_struct[train_idx], X_train_struct[val_idx]
                ytr, yv = y_train[train_idx], y_train[val_idx]
                
                train_ds = ESM2Dataset(Xtr_esm, Xtr_struct, ytr, augment=True)
                val_ds = ESM2Dataset(Xv_esm, Xv_struct, yv, augment=False)