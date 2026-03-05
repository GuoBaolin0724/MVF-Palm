import argparse
import os
import random
import time
import warnings
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_recall_curve
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import json
import torch.nn.functional as F

try:
    import optuna
    OPTUNA_AVAILABLE = True
except Exception:
    OPTUNA_AVAILABLE = False

class Config:
    ESM2_MODEL = "facebook/esm2_t36_3B_UR50D"
    BATCH_SIZE = 64
    EPOCHS = 50
    LR = 5e-5
    WEIGHT_DECAY = 5e-6
    DEVICE = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    SAVE_PATH = 'best_model-adaptive.pt'
    STRUCTURE_EMB_PATH = "embeddings.npy"
    STRUCTURE_DIM = 256
    ESM_DIM = 2560

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

def read_seqs_from_csv(pos_csv: str, neg_csv: str, seq_col: str = 'Fragment') -> pd.DataFrame:
    pos = pd.read_csv(pos_csv)
    neg = pd.read_csv(neg_csv)
    if seq_col not in pos.columns or seq_col not in neg.columns:
        raise ValueError(f"CSV must contain column '{seq_col}'")
    pos = pos[[seq_col]].dropna().drop_duplicates().reset_index(drop=True)
    neg = neg[[seq_col]].dropna().drop_duplicates().reset_index(drop=True)
    pos['label'] = 1
    neg['label'] = 0
    df = pd.concat([pos, neg], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    return df

from transformers import AutoTokenizer, AutoModelForMaskedLM

def load_esm2_model(model_name: str, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    return tokenizer, model

def pad_features(features: np.ndarray, maxlen: int) -> np.ndarray:
    N, L, D = features.shape
    if L == maxlen:
        return features
    out = np.zeros((N, maxlen, D), dtype=features.dtype)
    length = min(L, maxlen)
    out[:, :length, :] = features[:, :length, :]
    return out

def extract_esm2_residue_features(sequences: List[str], tokenizer, model, max_len: int, batch_size: int = 8, device: torch.device = Config.DEVICE):
    features = []
    for i in tqdm(range(0, len(sequences), batch_size), desc='Extracting ESM2 features'):
        batch = sequences[i:i+batch_size]
        inputs = tokenizer(batch, add_special_tokens=True, padding='max_length', truncation=True, max_length=max_len+2, return_tensors='pt')
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(** inputs, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1]
        residue = last_hidden[:, 1:-1, :].cpu().numpy()
        features.append(residue)
        del inputs, outputs, last_hidden
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    features = np.concatenate(features, axis=0)
    if features.shape[1] != max_len:
        features = pad_features(features, maxlen=max_len)
    if np.isnan(features).any():
        features = np.nan_to_num(features, nan=0.0)
    return features

def load_structure_features(struct_emb_path: str = Config.STRUCTURE_EMB_PATH):
    if not os.path.exists(struct_emb_path):
        raise FileNotFoundError(f"Structure feature file not found: {struct_emb_path}")
    struct_features = np.load(struct_emb_path)
    
    if len(struct_features.shape) != 2 or struct_features.shape[1] != Config.STRUCTURE_DIM:
        raise ValueError(
            f"Structure feature shape should be (N, {Config.STRUCTURE_DIM}), but got {struct_features.shape}"
        )
    
    print(f"Structure features loaded: shape {struct_features.shape}")
    
    return struct_features

class ESM2Dataset(Dataset):
    def __init__(self, X_esm: np.ndarray, X_struct: np.ndarray, y: np.ndarray, 
                 augment: bool = False, gauss_std: float = Config.GAUSS_STD,
                 drop_prob: float = Config.DROP_PROB, max_crop_frac: float = Config.MAX_CROP_FRAC, 
                 pad_value: float = 0.0):
        self.X_esm = X_esm.astype('float32')
        self.X_struct = X_struct.astype('float32')
        self.y = y.astype('float32')
        self.augment = augment
        self.gauss_std = gauss_std
        self.drop_prob = drop_prob
        self.max_crop_frac = max_crop_frac
        self.pad_value = pad_value
        
        if self.X_esm.shape[0] != self.X_struct.shape[0]:
             raise ValueError("Number of samples in ESM features and structure features do not match.")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x_esm = self.X_esm[idx].copy()
        x_struct = self.X_struct[idx].copy()
        y = self.y[idx]
        
        if self.augment:
            L, D = x_esm.shape
            pad_mask = np.all(np.isclose(x_esm, self.pad_value), axis=1)
            
            if self.gauss_std > 0:
                noise = np.random.normal(0, self.gauss_std, size=x_esm.shape).astype('float32')
                noise[pad_mask] = 0.0
                x_esm = x_esm + noise
            
            if 0.0 < self.drop_prob < 1.0:
                mask = np.random.rand(L) >= self.drop_prob
                mask = np.logical_or(mask, pad_mask)
                x_esm = x_esm * mask[:, None].astype('float32')
            
            if self.max_crop_frac > 0:
                max_crop = max(1, int(L * self.max_crop_frac))
                crop = np.random.randint(-max_crop, max_crop + 1)
                if crop > 0:
                    x_esm = np.vstack([x_esm[crop:], np.zeros((crop, D), dtype='float32')])
                elif crop < 0:
                    crop = -crop
                    x_esm = np.vstack([np.zeros((crop, D), dtype='float32'), x_esm[:-crop]])
            
            scale = 1.0 + np.random.uniform(-0.03, 0.03)
            x_esm = x_esm * scale
            x_esm = np.nan_to_num(x_esm, nan=0.0, posinf=0.0, neginf=0.0)
            
        return (torch.tensor(x_esm, dtype=torch.float32), 
                torch.tensor(x_struct, dtype=torch.float32)), \
               torch.tensor(y, dtype=torch.float32)

class ESM2_CNN_BiLSTM(nn.Module):
    def __init__(self, seq_len: int, esm_dim: int = Config.ESM_DIM, struct_dim: int = Config.STRUCTURE_DIM,
                 conv_filters: int = Config.CONV_FILTERS, lstm_units: int = Config.LSTM_UNITS,
                 dropout_lstm: float = Config.DROPOUT_LSTM, dropout_head: float = Config.DROPOUT_HEAD, 
                 spatial_dropout: float = Config.SPATIAL_DROPOUT):
        super().__init__()
        
        self.seq_len = seq_len
        self.esm_dim = esm_dim
        self.struct_dim = struct_dim
        self.fused_dim = esm_dim + struct_dim

        self.view_weight_layer = nn.Sequential(
            nn.Linear(self.fused_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

        self.conv = nn.Conv1d(in_channels=self.fused_dim, out_channels=conv_filters, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.bn = nn.BatchNorm1d(conv_filters)
        self.spatial_dropout = nn.Dropout2d(spatial_dropout)

        self.lstm = nn.LSTM(input_size=conv_filters, hidden_size=lstm_units, num_layers=1, batch_first=True, bidirectional=True)
        self.lstm_dropout = nn.Dropout(dropout_lstm)

        self.att_linear = nn.Linear(2 * lstm_units, lstm_units)
        self.att_tanh = nn.Tanh()
        self.att_score = nn.Linear(lstm_units, 1)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_head),
            nn.Linear(2 * lstm_units, 64),
            nn.ReLU(),
            nn.Dropout(dropout_head),
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

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]):
        x_esm, x_struct = inputs

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
        
        x = x.permute(0,2,1)
        x = self.conv(x)
        x = self.relu(x)
        x = self.spatial_dropout(x.unsqueeze(-1)).squeeze(-1)
        x = self.pool(x)
        x = self.bn(x)
        x = x.permute(0,2,1)
        lstm_out, _ = self.lstm(x)
        lstm_out = self.lstm_dropout(lstm_out)

        att = self.att_linear(lstm_out)
        att = self.att_tanh(att)
        att = self.att_score(att)
        att_weights = torch.softmax(att, dim=1)
        att_out = torch.sum(lstm_out * att_weights, dim=1)

        logits = self.classifier(att_out).squeeze(1)
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
        focal = alpha_tensor * (1 - pt) ** self.gamma * bce_loss
        if self.reduction == 'mean':
            return focal.mean()
        elif self.reduction == 'sum':
            return focal.sum()
        return focal

def compute_auc(y_true, y_pred):
    try:
        return float(roc_auc_score(y_true, y_pred))
    except Exception:
        return 0.5

def find_optimal_threshold(y_true, y_prob):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1s = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-8)
    return thresholds[np.argmax(f1s)] if len(f1s) > 0 else 0.5

def evaluate_model(model, loader, device, criterion=None, threshold=0.5):
    model.eval()
    ys = []
    y_probs = []
    total_loss = 0.0
    n = 0
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, matthews_corrcoef
    with torch.no_grad():
        for inputs, yb in loader:
            Xb_esm, Xb_struct = inputs
            Xb_esm = Xb_esm.to(device)
            Xb_struct = Xb_struct.to(device)
            yb = yb.to(device)
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
    auc = compute_auc(ys, y_probs) if len(np.unique(ys))>1 else 0.0
    
    acc = accuracy_score(ys, y_pred)
    f1 = f1_score(ys, y_pred, zero_division=0)
    precision = precision_score(ys, y_pred, zero_division=0)
    recall = recall_score(ys, y_pred, zero_division=0)
    cm = confusion_matrix(ys, y_pred) if len(ys)>0 else np.array([[0]])
    if cm.size == 1:
        tn = cm[0,0] if y_pred[0]==0 else 0
        fp = fn = tp = 0
    else:
        tn, fp, fn, tp = cm.ravel()
    specificity = tn/(tn+fp) if (tn+fp)>0 else 0.0
    mcc = matthews_corrcoef(ys, y_pred) if len(np.unique(y_pred))>1 and len(np.unique(ys))>1 else 0.0
    return {'auc':auc, 'acc':acc, 'f1':f1, 'precision':precision, 'recall':recall, 'specificity':specificity, 'mcc':mcc, 'loss':avg_loss, 'y_probs':y_probs, 'y_true':ys}

def train_one_fold(model, train_loader, val_loader, device, test_loader=None,
                   epochs:int=Config.EPOCHS, lr:float=Config.LR, weight_decay:float=Config.WEIGHT_DECAY,
                   clip_norm:float=Config.CLIP_NORM, label_smoothing_eps:float=Config.LABEL_SMOOTH_EPS,
                   patience:int=Config.PATIENCE, use_focal:bool=Config.USE_FOCAL, fold_idx=0):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    criterion = FocalLoss(alpha=Config.FOCAL_ALPHA, gamma=Config.FOCAL_GAMMA) if use_focal else nn.BCEWithLogitsLoss()

    best_val_auc = 0.0
    best_state = None
    history = {'train_loss':[], 'val_loss':[], 'train_auc':[], 'val_auc':[]}

    for epoch in range(1, epochs+1):
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
            yb_smoothed = yb * (1.0 - label_smoothing_eps) + 0.5 * label_smoothing_eps
            loss = criterion(logits, yb_smoothed)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
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

        print(f"Fold {fold_idx} | Epoch {epoch}/{epochs} - train_loss: {train_loss:.4f}, val_loss: {val_metrics['loss']:.4f}, train_auc: {train_auc:.4f}, val_auc: {val_auc:.4f}")

        if val_auc > best_val_auc + 1e-4:
            best_val_auc = val_auc
            best_state = {k:v.cpu() for k,v in model.state_dict().items()}
            save_path = Config.SAVE_PATH.replace('.pth', f'_fold{fold_idx}.pth') if fold_idx > 0 else Config.SAVE_PATH
            torch.save(best_state, save_path)
            print(f"Fold {fold_idx} | Best model saved to: {save_path}")

        if epoch - np.argmax(np.array(history['val_auc'])) > patience:
            print(f"Fold {fold_idx} | Early stopping triggered")
            break

    test_metrics = None
    if best_state is not None:
        model.load_state_dict(best_state)
        if test_loader is not None:
            val_metrics_best = evaluate_model(model, val_loader, device, criterion)
            optimal_threshold = find_optimal_threshold(val_metrics_best['y_true'], val_metrics_best['y_probs'])
            test_metrics = evaluate_model(model, test_loader, device, criterion, threshold=optimal_threshold)
            print(f"Fold {fold_idx} | Test metrics (optimal threshold): {test_metrics}")

    return model, history, test_metrics

def optuna_objective(trial, X_esm, X_struct, y, args):
    conv = trial.suggest_categorical('conv_filters', [16,32,64])
    lstm = trial.suggest_categorical('lstm_units', [32,64])
    dropout_lstm = trial.suggest_float('dropout_lstm', 0.2, 0.6)
    dropout_head = trial.suggest_float('dropout_head', 0.2, 0.6)
    spatial_dropout = trial.suggest_float('spatial_dropout', 0.1, 0.4)
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-3)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-3)
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = []
    
    seq_len = X_esm.shape[1]
    esm_dim = X_esm.shape[2]
    struct_dim = X_struct.shape[1]
    
    for train_idx, val_idx in skf.split(X_esm, y):
        Xtr_esm, Xv_esm = X_esm[train_idx], X_esm[val_idx]
        Xtr_struct, Xv_struct = X_struct[train_idx], X_struct[val_idx]
        ytr, yv = y[train_idx], y[val_idx]
        
        train_ds = ESM2Dataset(Xtr_esm, Xtr_struct, ytr, augment=True)
        val_ds = ESM2Dataset(Xv_esm, Xv_struct, yv, augment=False)
        
        tr_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True)
        v_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False)
        
        model = ESM2_CNN_BiLSTM(seq_len=seq_len, esm_dim=esm_dim, struct_dim=struct_dim, 
                                conv_filters=conv, lstm_units=lstm, dropout_lstm=dropout_lstm, 
                                dropout_head=dropout_head, spatial_dropout=spatial_dropout).to(Config.DEVICE)
        
        model, history, _ = train_one_fold(model, tr_loader, v_loader, Config.DEVICE, 
                                           epochs=8, lr=lr, weight_decay=weight_decay, 
                                           clip_norm=Config.CLIP_NORM, label_smoothing_eps=Config.LABEL_SMOOTH_EPS, 
                                           patience=3, use_focal=Config.USE_FOCAL, fold_idx=0)
        
        val_metrics = evaluate_model(model, v_loader, Config.DEVICE)
        scores.append(val_metrics['auc'])
        
        del model
        torch.cuda.empty_cache()
        
    return float(np.mean(scores))

def convert_ndarray_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_ndarray_to_list(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_ndarray_to_list(item) for item in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    else:
        return obj

def save_metrics_to_txt(metrics_dict, save_path, mode="single"):
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    metrics_dict_converted = convert_ndarray_to_list(metrics_dict)
    metrics_with_time = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mode": mode,
        "metrics": metrics_dict_converted
    }

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_with_time, f, ensure_ascii=False, indent=2)
    print(f"Metrics saved to: {save_path}")

def plot_auc_curve(history_list, test_aucs=None, mode="single", save_path="auc_curve.png"):
    plt.figure(figsize=(10, 6))
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    if mode == "single":
        history = history_list[0]
        epochs = range(1, len(history['train_auc']) + 1)
        
        plt.plot(epochs, history['train_auc'], 'b-o', label='Train AUC', linewidth=2, markersize=4)
        plt.plot(epochs, history['val_auc'], 'r-s', label='Val AUC', linewidth=2, markersize=4)
        
        test_auc = test_aucs[0]
        plt.axhline(y=test_auc, color='green', linestyle='--', label=f'Test AUC = {test_auc:.4f}', linewidth=2)
        
        plt.title('Single Fold - Train/Val/Test AUC Curve', fontsize=14, fontweight='bold')
    
    elif mode == "kfold":
        n_folds = len(history_list)
        n_epochs = len(history_list[0]['train_auc'])
        
        avg_train_auc = np.zeros(n_epochs)
        avg_val_auc = np.zeros(n_epochs)
        
        for history in history_list:
            avg_train_auc += np.array(history['train_auc'])
            avg_val_auc += np.array(history['val_auc'])
        
        avg_train_auc /= n_folds
        avg_val_auc /= n_folds
        avg_test_auc = np.mean(test_aucs) if test_aucs else 0.0
        
        epochs = range(1, n_epochs + 1)
        plt.plot(epochs, avg_train_auc, 'b-o', label=f'Avg Train AUC (n={n_folds})', linewidth=2, markersize=4)
        plt.plot(epochs, avg_val_auc, 'r-s', label=f'Avg Val AUC (n={n_folds})', linewidth=2, markersize=4)
        plt.axhline(y=avg_test_auc, color='green', linestyle='--', label=f'Avg Test AUC = {avg_test_auc:.4f}', linewidth=2)
        
        plt.title(f'{n_folds}-Fold Cross Validation - Avg AUC Curve', fontsize=14, fontweight='bold')
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('AUC', fontsize=12)
    plt.xlim(1, len(epochs))
    plt.ylim(0.5, 1.0)
    plt.grid(True, alpha=0.3, linestyle='-')
    plt.legend(loc='lower right', fontsize=11)
    plt.tight_layout()
    
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"AUC curve saved to: {save_path}")

def plot_history(history, save_path_prefix='esm2_run'):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curve')
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_auc'], label='Train AUC')
    plt.plot(history['val_auc'], label='Val AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()
    plt.title('AUC Curve (Train/Val)')
    
    plt.tight_layout()
    plt.savefig(f'{save_path_prefix}_loss_auc.png', dpi=300)
    plt.close()
    print(f"Loss-AUC curve saved to: {save_path_prefix}_loss_auc.png")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pos_csv', type=str, required=True)
    parser.add_argument('--neg_csv', type=str, required=True)
    parser.add_argument('--seq_col', type=str, default='Fragment')
    parser.add_argument('--mode', type=str, default='train', choices=['train','optuna','kfold'])
    parser.add_argument('--n_trials', type=int, default=20)
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--max_len', type=int, default=31)
    parser.add_argument('--save_dir', type=str, default='./results_adaptive', help='Root directory to save metrics, figures, and models')
    parser.add_argument('--struct_emb', type=str, default=Config.STRUCTURE_EMB_PATH, help='Path to structure feature file')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    metrics_dir = os.path.join(args.save_dir, 'metrics')
    fig_dir = os.path.join(args.save_dir, 'figures')
    model_dir = os.path.join(args.save_dir, 'models')
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    Config.SAVE_PATH = os.path.join(model_dir, 'best_model_adaptive.pth')
    Config.STRUCTURE_EMB_PATH = args.struct_emb

    df = read_seqs_from_csv(args.pos_csv, args.neg_csv, seq_col=args.seq_col)
    sequences = df[args.seq_col].astype(str).tolist()
    labels = df['label'].values.astype(int)
    max_len = 31
    print(f"Using {len(sequences)} sequences, fixed sequence length={max_len} (matching ESM features)")

    print('Loading ESM2 model...')
    tokenizer, esm2_model = load_esm2_model(Config.ESM2_MODEL, Config.DEVICE)
    X_esm = extract_esm2_residue_features(sequences, tokenizer, esm2_model, max_len=max_len, batch_size=8, device=Config.DEVICE)
    print(f"ESM features shape: {X_esm.shape} (should be N×31×2560)")
    
    X_struct = load_structure_features(args.struct_emb)

    if X_esm.shape[0] != X_struct.shape[0]:
        raise ValueError(f"Number of samples mismatch: ESM features {X_esm.shape[0]}, structure features {X_struct.shape[0]}")
    
    seq_len = X_esm.shape[1]
    esm_dim = X_esm.shape[2]
    struct_dim = X_struct.shape[1]

    X_train_esm, X_temp_esm, y_train, y_temp = train_test_split(
        X_esm, labels, test_size=0.2, random_state=42, stratify=labels)
    X_train_struct, X_temp_struct, _, _ = train_test_split(
        X_struct, labels, test_size=0.2, random_state=42, stratify=labels)
    
    X_val_esm, X_test_esm, y_val, y_test = train_test_split(
        X_temp_esm, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    X_val_struct, X_test_struct, _, _ = train_test_split(
        X_temp_struct, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    if args.mode == 'optuna':
        if not OPTUNA_AVAILABLE:
            raise RuntimeError('Optuna is not installed, please install first: pip install optuna')
        study = optuna.create_study(direction='maximize')
        func = lambda trial: optuna_objective(trial, X_train_esm, X_train_struct, y_train, args)
        study.optimize(func, n_trials=args.n_trials)
        print('Best hyperparameters:', study.best_trial.params)
        save_metrics_to_txt(study.best_trial.params, os.path.join(metrics_dir, 'optuna_best_params.txt'), mode="optuna")
        return

    if args.mode == 'train':
        train_ds = ESM2Dataset(X_train_esm, X_train_struct, y_train, augment=True)
        val_ds = ESM2Dataset(X_val_esm, X_val_struct, y_val, augment=False)
        test_ds = ESM2Dataset(X_test_esm, X_test_struct, y_test, augment=False)

        n_pos = int(np.sum(y_train==1))
        n_neg = len(y_train) - n_pos
        sampler = None
        if n_pos > 0 and n_neg > 0:
            weights = np.array([(n_neg / n_pos) if y==1 else 1.0 for y in y_train], dtype=np.float32)
            sampler = WeightedRandomSampler(torch.tensor(weights, dtype=torch.double), len(weights), replacement=True)

        train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=(sampler is None), sampler=sampler)
        val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=Config.BATCH_SIZE, shuffle=False)

        model = ESM2_CNN_BiLSTM(seq_len=seq_len, esm_dim=esm_dim, struct_dim=struct_dim).to(Config.DEVICE)
        
        if Config.USE_FOCAL:
            criterion = FocalLoss(alpha=Config.FOCAL_ALPHA, gamma=Config.FOCAL_GAMMA)
        elif Config.USE_CLASS_WEIGHT and n_pos>0 and n_neg>0:
            pos_weight = torch.tensor([n_neg / n_pos], device=Config.DEVICE)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            criterion = nn.BCEWithLogitsLoss()

        model, history, test_metrics = train_one_fold(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=Config.DEVICE,
            fold_idx=0
        )

        val_metrics = evaluate_model(model, val_loader, Config.DEVICE, criterion)
        all_metrics = {
            "train_metrics": {
                "auc": history['train_auc'][-1],
                "loss": history['train_loss'][-1]
            },
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "optimal_threshold": find_optimal_threshold(val_metrics['y_true'], val_metrics['y_probs'])
        }

        metrics_save_path = os.path.join(metrics_dir, 'single_fold_metrics.txt')
        save_metrics_to_txt(all_metrics, metrics_save_path, mode="single")

        print("\n=== Final metrics of single fold training ===")
        print(f"Training set - AUC: {all_metrics['train_metrics']['auc']:.4f}, Loss: {all_metrics['train_metrics']['loss']:.4f}")
        print(f"Validation set - AUC: {all_metrics['val_metrics']['auc']:.4f}, F1: {all_metrics['val_metrics']['f1']:.4f}, MCC: {all_metrics['val_metrics']['mcc']:.4f}")
        print(f"Test set - AUC: {all_metrics['test_metrics']['auc']:.4f}, F1: {all_metrics['test_metrics']['f1']:.4f}, MCC: {all_metrics['test_metrics']['mcc']:.4f}")

        auc_curve_save_path = os.path.join(fig_dir, 'single_fold_auc_curve.png')
        plot_auc_curve(
            history_list=[history],
            test_aucs=[all_metrics['test_metrics']['auc']],
            mode="single",
            save_path=auc_curve_save_path
        )
        plot_history(history, save_path_prefix=os.path.join(fig_dir, 'single_fold'))
        return

    if args.mode == 'kfold':
        skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=42)
        fold_metrics_list = []
        history_list = []
        test_aucs = []
        best_fold_idx = 0
        best_val_auc = 0.0

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_esm, labels), 1):
            print(f"\n=== Starting fold {fold_idx}/{args.n_splits} ===")
            
            Xtr_esm, Xv_esm = X_esm[train_idx], X_esm[val_idx]
            Xtr_struct, Xv_struct = X_struct[train_idx], X_struct[val_idx]
            ytr, yv = labels[train_idx], labels[val_idx]

            train_ds = ESM2Dataset(Xtr_esm, Xtr_struct, ytr, augment=True)
            val_ds = ESM2Dataset(Xv_esm, Xv_struct, yv, augment=False)
            test_ds = ESM2Dataset(X_test_esm, X_test_struct, y_test, augment=False)

            tr_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True)
            v_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False)
            te_loader = DataLoader(test_ds, batch_size=Config.BATCH_SIZE, shuffle=False)

            model = ESM2_CNN_BiLSTM(seq_len=seq_len, esm_dim=esm_dim, struct_dim=struct_dim).to(Config.DEVICE)

            model, history, test_metrics = train_one_fold(
                model=model,
                train_loader=tr_loader,
                val_loader=v_loader,
                test_loader=te_loader,
                device=Config.DEVICE,
                fold_idx=fold_idx
            )

            val_metrics = evaluate_model(model, v_loader, Config.DEVICE)
            fold_metrics = {
                "fold_idx": fold_idx,
                "train_metrics": {
                    "auc": history['train_auc'][-1],
                    "loss": history['train_loss'][-1]
                },
                "val_metrics": val_metrics,
                "test_metrics": test_metrics,
                "optimal_threshold": find_optimal_threshold(val_metrics['y_true'], val_metrics['y_probs'])
            }
            fold_metrics_list.append(fold_metrics)
            history_list.append(history)
            test_aucs.append(test_metrics['auc'])

            if val_metrics['auc'] > best_val_auc + 1e-4:
                best_val_auc = val_metrics['auc']
                best_fold_idx = fold_idx

            print(f"Summary of fold {fold_idx} metrics:")
            print(f"  Training AUC: {fold_metrics['train_metrics']['auc']:.4f}, Validation AUC: {fold_metrics['val_metrics']['auc']:.4f}, Test AUC: {fold_metrics['test_metrics']['auc']:.4f}")

        avg_metrics = {
            "n_folds": args.n_splits,
            "best_fold_idx": best_fold_idx,
            "best_val_auc": best_val_auc,
            "avg_train_auc": np.mean([m['train_metrics']['auc'] for m in fold_metrics_list]),
            "avg_val_auc": np.mean([m['val_metrics']['auc'] for m in fold_metrics_list]),
            "avg_test_auc": np.mean(test_aucs),
            "avg_test_f1": np.mean([m['test_metrics']['f1'] for m in fold_metrics_list]),
            "avg_test_mcc": np.mean([m['test_metrics']['mcc'] for m in fold_metrics_list]),
            "each_fold_metrics": fold_metrics_list
        }

        metrics_save_path = os.path.join(metrics_dir, f'{args.n_splits}fold_metrics.txt')
        save_metrics_to_txt(avg_metrics, metrics_save_path, mode="kfold")

        print("\n=== Average metrics of K-fold cross validation ===")
        print(f"Number of folds: {args.n_splits}")
        print(f"Best fold index: {best_fold_idx} (Validation AUC: {best_val_auc:.4f})")
        print(f"Average training AUC: {avg_metrics['avg_train_auc']:.4f}")
        print(f"Average validation AUC: {avg_metrics['avg_val_auc']:.4f}")
        print(f"Average test AUC: {avg_metrics['avg_test_auc']:.4f}")

        auc_curve_save_path = os.path.join(fig_dir, f'{args.n_splits}fold_auc_curve.png')
        plot_auc_curve(
            history_list=history_list,
            test_aucs=test_aucs,
            mode="kfold",
            save_path=auc_curve_save_path
        )

        best_model_path = os.path.join(model_dir, 'best_fold_model_adaptive.pth')
        best_fold_model_path = os.path.join(model_dir, f'best_model_fold{best_fold_idx}_adaptive.pth')
        if os.path.exists(best_fold_model_path):
            if os.path.exists(best_model_path):
                 os.remove(best_model_path)
            os.rename(best_fold_model_path, best_model_path)
            print(f"\nBest fold model saved as: {best_model_path}")
            
        for f in os.listdir(model_dir):
            if f.startswith('best_model_fold') and f.endswith('_adaptive.pth'):
                os.remove(os.path.join(model_dir, f))
        
        return

if __name__ == '__main__':
    main()