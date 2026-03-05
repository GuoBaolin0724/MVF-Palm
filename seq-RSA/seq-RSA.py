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
from sklearn.metrics import roc_auc_score, precision_recall_curve, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, matthews_corrcoef
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import json

try:
    import optuna
    OPTUNA_AVAILABLE = True
except Exception:
    OPTUNA_AVAILABLE = False

class Config:
    ESM2_MODEL = "facebook/esm1v_t33_650M_UR90S_1"
    BATCH_SIZE = 64
    EPOCHS = 50
    LR = 5e-5
    WEIGHT_DECAY = 5e-6
    DEVICE = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    SAVE_PATH = 'best_model-2.pt'

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

    RSA_POOL_DIM = 128
    RSA_QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

def read_seqs_from_csv(pos_csv: str, neg_csv: str, seq_col: str = 'Fragment', rsa_col: str = 'rsa'):
    pos = pd.read_csv(pos_csv)
    neg = pd.read_csv(neg_csv)
    
    if seq_col not in pos.columns or seq_col not in neg.columns:
        raise ValueError(f"Both CSVs must contain column '{seq_col}'")
    
    if rsa_col not in pos.columns or rsa_col not in neg.columns:
        raise ValueError(f"Both CSVs must contain column '{rsa_col}'")
    
    pos = pos[[seq_col, rsa_col]].dropna().drop_duplicates().reset_index(drop=True)
    neg = neg[[seq_col, rsa_col]].dropna().drop_duplicates().reset_index(drop=True)
    
    pos['label'] = 1
    neg['label'] = 0
    
    df = pd.concat([pos, neg], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    
    def parse_rsa(rsa_str):
        try:
            return np.array(eval(rsa_str), dtype=np.float32)
        except:
            return np.array([float(rsa_str)], dtype=np.float32)
    
    df['rsa_sequence'] = df[rsa_col].apply(parse_rsa)
    return df

def pool_rsa_features(rsa_sequences: List[np.ndarray], max_len: int, pool_dim: int = Config.RSA_POOL_DIM) -> np.ndarray:
    pooled_features = []
    quantiles = Config.RSA_QUANTILES
    n_quantiles = len(quantiles)
    n_basic_stats = 8
    
    min_required_dim = n_basic_stats + n_quantiles
    if pool_dim < min_required_dim:
        raise ValueError(f"RSA pool dim must be >= {min_required_dim} (current: {pool_dim})")
    
    for rsa_seq in rsa_sequences:
        if len(rsa_seq) > max_len:
            rsa_seq = rsa_seq[:max_len]
        else:
            rsa_seq = np.pad(rsa_seq, (0, max_len - len(rsa_seq)), mode='constant', constant_values=0.0)
        
        basic_stats = np.array([
            np.mean(rsa_seq),
            np.std(rsa_seq),
            np.min(rsa_seq),
            np.max(rsa_seq),
            np.median(rsa_seq),
            np.ptp(rsa_seq),
            np.sum(rsa_seq > 0.2),
            np.sum(rsa_seq <= 0.2)
        ], dtype=np.float32)
        
        quantile_vals = np.quantile(rsa_seq, quantiles, axis=0).astype(np.float32)
        
        n_slide_windows = pool_dim - n_basic_stats - n_quantiles
        window_size = max(1, max_len // n_slide_windows)
        slide_stats = []
        for i in range(n_slide_windows):
            start = i * window_size
            end = min(start + window_size, max_len)
            if start >= end:
                slide_stats.append(0.0)
            else:
                slide_stats.append(np.mean(rsa_seq[start:end]))
        slide_stats = np.array(slide_stats, dtype=np.float32)
        
        sample_feature = np.concatenate([basic_stats, quantile_vals, slide_stats], axis=0)
        pooled_features.append(sample_feature)
    
    return np.array(pooled_features)

def align_raw_rsa(rsa_sequences: List[np.ndarray], max_len: int) -> np.ndarray:
    aligned_rsa = []
    for rsa_seq in rsa_sequences:
        if len(rsa_seq) > max_len:
            rsa_seq = rsa_seq[:max_len]
        else:
            rsa_seq = np.pad(rsa_seq, (0, max_len - len(rsa_seq)), mode='constant', constant_values=0.0)
        aligned_rsa.append(rsa_seq[:, np.newaxis])
    return np.array(aligned_rsa, dtype=np.float32)

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
    for i in tqdm(range(0, len(sequences), batch_size), desc='ESM2 extract'):
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

class ESM2Dataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, augment: bool = False, gauss_std: float = Config.GAUSS_STD,
                 drop_prob: float = Config.DROP_PROB, max_crop_frac: float = Config.MAX_CROP_FRAC, pad_value: float = 0.0):
        self.X = X.astype('float32')
        self.y = y.astype('float32')
        self.augment = augment
        self.gauss_std = gauss_std
        self.drop_prob = drop_prob
        self.max_crop_frac = max_crop_frac
        self.pad_value = pad_value

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx].copy()
        y = self.y[idx]
        if self.augment:
            L, D = x.shape
            pad_mask = np.all(np.isclose(x, self.pad_value), axis=1)
            if self.gauss_std > 0:
                noise = np.random.normal(0, self.gauss_std, size=x.shape).astype('float32')
                noise[pad_mask] = 0.0
                x = x + noise
            if 0.0 < self.drop_prob < 1.0:
                mask = np.random.rand(L) >= self.drop_prob
                mask = np.logical_or(mask, pad_mask)
                x = x * mask[:, None].astype('float32')
            if self.max_crop_frac > 0:
                max_crop = max(1, int(L * self.max_crop_frac))
                crop = np.random.randint(-max_crop, max_crop + 1)
                if crop > 0:
                    x = np.vstack([x[crop:], np.zeros((crop, D), dtype='float32')])
                elif crop < 0:
                    crop = -crop
                    x = np.vstack([np.zeros((crop, D), dtype='float32'), x[:-crop]])
            scale = 1.0 + np.random.uniform(-0.03, 0.03)
            x = x * scale
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

class ESM2_CNN_BiLSTM(nn.Module):
    def __init__(self, seq_len: int, esm2_dim: int, conv_filters: int = Config.CONV_FILTERS, lstm_units: int = Config.LSTM_UNITS,
                 dropout_lstm: float = Config.DROPOUT_LSTM, dropout_head: float = Config.DROPOUT_HEAD, spatial_dropout: float = Config.SPATIAL_DROPOUT):
        super().__init__()
        self.seq_len = seq_len
        self.conv = nn.Conv1d(in_channels=esm2_dim, out_channels=conv_filters, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.pool_kernel = 1 if seq_len <= 2 else 2
        self.pool = nn.MaxPool1d(kernel_size=self.pool_kernel)
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

    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.relu(x)
        x = self.spatial_dropout(x.unsqueeze(-1)).squeeze(-1)
        x = self.pool(x)
        x = self.bn(x)
        x = x.permute(0, 2, 1)
        
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
    with torch.no_grad():
        for Xb, yb in loader:
            Xb = Xb.to(device)
            yb = yb.to(device)
            logits = model(Xb)
            if criterion is not None:
                loss = criterion(logits, yb)
                total_loss += loss.item() * Xb.size(0)
            probs = torch.sigmoid(logits).cpu().numpy()
            y_probs.extend(probs.tolist())
            ys.extend(yb.cpu().numpy().tolist())
            n += Xb.size(0)
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
        for Xb, yb in train_loader:
            Xb = Xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(Xb)
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

def optuna_objective(trial, X, y, args):
    conv = trial.suggest_categorical('conv_filters', [16,32,64])
    lstm = trial.suggest_categorical('lstm_units', [32,64])
    dropout_lstm = trial.suggest_float('dropout_lstm', 0.2, 0.6)
    dropout_head = trial.suggest_float('dropout_head', 0.2, 0.6)
    spatial_dropout = trial.suggest_float('spatial_dropout', 0.1, 0.4)
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-3)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-3)
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = []
    for train_idx, val_idx in skf.split(X, y):
        Xtr, Xv = X[train_idx], X[val_idx]
        ytr, yv = y[train_idx], y[val_idx]
        train_ds = ESM2Dataset(Xtr, ytr, augment=True)
        val_ds = ESM2Dataset(Xv, yv, augment=False)
        tr_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True)
        v_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False)
        model = ESM2_CNN_BiLSTM(seq_len=X.shape[1], esm2_dim=X.shape[2], conv_filters=conv, lstm_units=lstm, dropout_lstm=dropout_lstm, dropout_head=dropout_head, spatial_dropout=spatial_dropout).to(Config.DEVICE)
        model, history, _ = train_one_fold(model, tr_loader, v_loader, Config.DEVICE, epochs=8, lr=lr, weight_decay=weight_decay, clip_norm=Config.CLIP_NORM, label_smoothing_eps=Config.LABEL_SMOOTH_EPS, patience=3, use_focal=Config.USE_FOCAL)
        val_metrics = evaluate_model(model, v_loader, Config.DEVICE)
        scores.append(val_metrics['auc'])
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
    parser.add_argument('--rsa_col', type=str, default='rsa')
    parser.add_argument('--mode', type=str, default='train', choices=['train','optuna','kfold'])
    parser.add_argument('--n_trials', type=int, default=20)
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--max_len', type=int, default=None)
    parser.add_argument('--save_dir', type=str, default='./results', help='Root directory for saving metrics, figures, and models')
    parser.add_argument('--use_rsa', action='store_true', help='Whether to use RSA as one of the features')
    parser.add_argument('--rsa_no_pool', action='store_true', help='Whether to not pool RSA (use raw residue-level RSA sequences)')
    parser.add_argument('--rsa_pool_dim', type=int, default=Config.RSA_POOL_DIM, help='Dimension of pooled RSA features (minimum 17, only effective when --rsa_no_pool=False)')
    args = parser.parse_args()

    if args.rsa_no_pool and not args.use_rsa:
        raise ValueError("--rsa_no_pool must be used with --use_rsa")

    os.makedirs(args.save_dir, exist_ok=True)
    metrics_dir = os.path.join(args.save_dir, 'metrics')
    fig_dir = os.path.join(args.save_dir, 'figures')
    model_dir = os.path.join(args.save_dir, 'models')
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    Config.SAVE_PATH = os.path.join(model_dir, 'best_model.pth')

    df = read_seqs_from_csv(args.pos_csv, args.neg_csv, seq_col=args.seq_col, rsa_col=args.rsa_col)
    sequences = df[args.seq_col].astype(str).tolist()
    labels = df['label'].values.astype(int)
    rsa_sequences = df['rsa_sequence'].tolist()

    if args.max_len is None:
        max_len = min(max(len(s) for s in sequences), 1024)
    else:
        max_len = args.max_len
    print(f"Using {len(sequences)} sequences, max_len={max_len}")

    print('Loading ESM2 model...')
    tokenizer, esm2_model = load_esm2_model(Config.ESM2_MODEL, Config.DEVICE)
    X = extract_esm2_residue_features(sequences, tokenizer, esm2_model, max_len=max_len, batch_size=8, device=Config.DEVICE)
    print('ESM2 features shape before processing:', X.shape)

    if args.use_rsa:
        if args.rsa_no_pool:
            print(f"Using raw RSA (no pooling), aligning to max_len={max_len}...")
            
            X_rsa_raw = align_raw_rsa(rsa_sequences, max_len)
            print(f"Raw RSA features shape: {X_rsa_raw.shape}")
            
            X = np.concatenate([X, X_rsa_raw], axis=2)
            print(f"Combined features shape (ESM2 + raw RSA): {X.shape}")
        else:
            print(f"Using pooled RSA (pool dim={args.rsa_pool_dim}), processing features...")
            
            X_esm2_pooled = np.mean(X, axis=1)
            print(f"ESM2 features after pooling: {X_esm2_pooled.shape}")
            
            X_rsa_pooled = pool_rsa_features(rsa_sequences, max_len, pool_dim=args.rsa_pool_dim)
            print(f"RSA features after pooling: {X_rsa_pooled.shape}")
            
            X_combined = np.concatenate([X_esm2_pooled, X_rsa_pooled], axis=1)
            
            X = X_combined.reshape(-1, 1, X_combined.shape[-1])
            print(f"Final input shape for model: {X.shape}")
    else:
        topo_npy_default = "combined_fragments_topo_onehot.npy"
        topo_loaded = None
        try:
            if os.path.exists(topo_npy_default):
                topo_loaded = np.load(topo_npy_default, allow_pickle=False)
                print(f"[INFO] Loaded topology one-hot: {topo_npy_default} shape={topo_loaded.shape}")
            else:
                alt = os.path.splitext(Config.SAVE_PATH)[0] + "_topo_onehot.npy"
                if os.path.exists(alt):
                    topo_loaded = np.load(alt, allow_pickle=False)
                    print(f"[INFO] Loaded topology one-hot: {alt} shape={topo_loaded.shape}")
        except Exception as e:
            print(f"[WARN] Could not load topology one-hot: {e}")

        if topo_loaded is not None:
            if topo_loaded.ndim != 3 or topo_loaded.shape[1] != X.shape[1] or topo_loaded.shape[2] != 4:
                if topo_loaded.ndim == 3 and topo_loaded.shape[0] == X.shape[0]:
                    L_esm = X.shape[1]
                    L_topo = topo_loaded.shape[1]
                    if L_topo >= L_esm:
                        topo_loaded = topo_loaded[:, :L_esm, :]
                        print(f"[WARN] Truncated topo to {L_esm} residues")
                    else:
                        topo_loaded = np.pad(topo_loaded, ((0,0),(0,L_esm-L_topo),(0,0)), mode='constant')
                        print(f"[WARN] Padded topo to {L_esm} residues")
                else:
                    raise ValueError(f"Topology shape {topo_loaded.shape} incompatible with ESM {X.shape}")
            if topo_loaded.shape[0] != X.shape[0]:
                raise ValueError(f"Sample count mismatch: topo {topo_loaded.shape[0]} vs esm {X.shape[0]}")
            X = np.concatenate([X, topo_loaded.astype(np.float32)], axis=2)
            print(f"[INFO] New feature shape: {X.shape}")
        else:
            print("[INFO] Continuing with ESM2 features only")

    X_train, X_temp, y_train, y_temp = train_test_split(X, labels, test_size=0.2, random_state=42, stratify=labels)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    if args.mode == 'optuna':
        if not OPTUNA_AVAILABLE:
            raise RuntimeError('Optuna not installed')
        study = optuna.create_study(direction='maximize')
        func = lambda trial: optuna_objective(trial, X_train, y_train, args)
        study.optimize(func, n_trials=args.n_trials)
        print('Best params:', study.best_trial.params)
        save_metrics_to_txt(study.best_trial.params, os.path.join(metrics_dir, 'optuna_best_params.txt'), mode="optuna")
        return

    if args.mode == 'train':
        train_ds = ESM2Dataset(X_train, y_train, augment=True)
        val_ds = ESM2Dataset(X_val, y_val, augment=False)
        test_ds = ESM2Dataset(X_test, y_test, augment=False)

        n_pos = int(np.sum(y_train==1))
        n_neg = len(y_train) - n_pos
        if n_pos == 0 or n_neg == 0:
            sampler = None
        else:
            weights = np.array([(n_neg / n_pos) if y==1 else 1.0 for y in y_train], dtype=np.float32)
            sampler = WeightedRandomSampler(torch.tensor(weights, dtype=torch.double), len(weights), replacement=True)

        train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=(sampler is None), sampler=sampler)
        val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=Config.BATCH_SIZE, shuffle=False)

        model = ESM2_CNN_BiLSTM(
            seq_len=X.shape[1],
            esm2_dim=X.shape[2]
        ).to(Config.DEVICE)
        print(f"Model initialized with seq_len={X.shape[1]}, esm2_dim={X.shape[2]}, pool_kernel={model.pool_kernel}")

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
            "optimal_threshold": find_optimal_threshold(val_metrics['y_true'], val_metrics['y_probs']),
            "feature_config": {
                "use_rsa": args.use_rsa,
                "rsa_no_pool": args.rsa_no_pool if args.use_rsa else None,
                "rsa_pool_dim": args.rsa_pool_dim if args.use_rsa and not args.rsa_no_pool else None,
                "esm2_dim": 2560,
                "final_input_dim": X.shape[2]
            }
        }

        metrics_save_path = os.path.join(metrics_dir, 'single_fold_metrics.txt')
        save_metrics_to_txt(all_metrics, metrics_save_path, mode="single")
        print("\n=== Final metrics of single fold training ===")
        print(f"Training set - AUC: {all_metrics['train_metrics']['auc']:.4f}, Loss: {all_metrics['train_metrics']['loss']:.4f}")
        print(f"Validation set - AUC: {all_metrics['val_metrics']['auc']:.4f}, F1: {all_metrics['val_metrics']['f1']:.4f}, MCC: {all_metrics['val_metrics']['mcc']:.4f}")
        print(f"Test set - AUC: {all_metrics['test_metrics']['auc']:.4f}, F1: {all_metrics['test_metrics']['f1']:.4f}, MCC: {all_metrics['test_metrics']['mcc']:.4f}")
        print(f"Optimal threshold: {all_metrics['optimal_threshold']:.4f}")

if __name__ == "__main__":
    main()