#!/usr/bin/env python3
"""
topo_get.py

Reads:
 - DeepTMHMM full-protein prediction text (FASTA-like or "id <sep> topo" lines)
 - positive fragments CSV (--pos) and negative fragments CSV (--neg). Each CSV must have header containing:
     Fragment or fragment_id, parent_id, site_pos
   where site_pos is 1-based coordinate of the fragment's 16th residue in the parent protein.
 - ESM2 embeddings (.npy or .npz). Optionally provide --esm_ids (one id per line) to map a .npy without ids.

Outputs:
 - {out}_topo_onehot.npy : (N,31,4) uint8 one-hot topo channels (I,O,S,M)
 - {out}_esm2.npy       : (N,31,C) float32 ESM embeddings (C typically 1024)
 - {out}.npz           : compressed file with fragment_ids, parent_ids, site_pos, labels, topo, esm2

Usage example:
 python topo_get.py \
   --deeptmhmm predicted_topologies.txt \
   --pos topo_po.csv \
   --neg topo_ne.csv \
   --esm2 esm2.npy \
   --esm_ids esm_ids.txt \
   --out combined_fragments
"""

import argparse
from pathlib import Path
import re
import numpy as np
import csv
from typing import Dict, List, Tuple, Optional

# mapping order: I,O,S,M -> one-hot index 0..3
TOPO_CHARS = ['I', 'O', 'S', 'M']
TOPO_MAP = {c: i for i, c in enumerate(TOPO_CHARS)}
PAD_CHAR = 'I'
WINDOW = 31
HALF = WINDOW // 2


def parse_deeptmhmm_file(path: str) -> Dict[str, str]:
    """
    Robust parse for DeepTMHMM-like output that may contain:
      >header (possibly like '>A0A1B2JLU2 | GLOB')
      SEQUENCE_LINE (AA letters)
      TOPO_LINE (I/O/S/M...), possibly folded across multiple lines
    This function will:
      - extract id as the token after '>' up to first whitespace or '|' char
      - scan subsequent lines until it finds a line made only of I/O/S/M chars
      - if topo spans multiple consecutive I/O/S/M-only lines, concatenate them
      - return dict: id -> topo_string (uppercase, only I/O/S/M)
    """
    import re
    d: Dict[str, str] = {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"DeepTMHMM file not found: {path}")
    lines = p.read_text().splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line.startswith('>'):
            # extract id: after '>' up to whitespace or '|' (if present)
            header = line[1:].strip()
            if '|' in header:
                sid = header.split('|', 1)[0].strip().split()[0]
            else:
                sid = header.split()[0]
            # now scan forward to find next line(s) that are solely I/O/S/M (possibly folded)
            j = i + 1
            topo_lines = []
            while j < len(lines):
                s = lines[j].strip()
                if not s:
                    j += 1
                    continue
                # if line is composed only of I/O/S/M letters (any case) -> treat as topo
                if re.fullmatch(r'[IOSMiosm]+', s):
                    topo_lines.append(s.upper())
                    j += 1
                    # also allow subsequent contiguous topo lines (folded topo)
                    while j < len(lines) and re.fullmatch(r'[IOSMiosm]+', lines[j].strip()):
                        topo_lines.append(lines[j].strip().upper())
                        j += 1
                    break
                else:
                    # not a topo line (likely sequence); continue scanning
                    j += 1
            if topo_lines:
                topo = ''.join(topo_lines)
                d[sid] = topo
            else:
                # no topo found for this header; store empty or warn
                d[sid] = ''
            i = j
            continue
        else:
            # handle the rare case of "id topo" in the same line (no leading '>')
            parts = re.split(r'\s+', line, maxsplit=1)
            if len(parts) == 2 and re.fullmatch(r'[IOSMiosm]+', parts[1]):
                pid, topo = parts[0], parts[1]
                d[pid] = topo.upper()
            i += 1
    return d


def read_fragments_from_csv(path: str) -> List[Tuple[str, str, int]]:
    """Read CSV/TSV fragment table. Accepts header that contains either 'fragment_id' or 'Fragment'."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Fragment file not found: {path}")
    data = []
    with p.open() as fh:
        first = fh.readline()
        fh.seek(0)
        delim = ',' if ',' in first else '\t' if '\t' in first else ','
        reader = csv.DictReader(fh, delimiter=delim)
        headers = [h.strip() for h in reader.fieldnames] if reader.fieldnames else []
        frag_col = None
        for cand in ['fragment_id', 'Fragment', 'fragment', 'frag_id']:
            if cand in headers:
                frag_col = cand
                break
        if 'parent_id' not in headers or 'site_pos' not in headers or frag_col is None:
            raise ValueError(f"Fragment file {path} must contain columns: one of [fragment_id,Fragment], parent_id, site_pos. Found: {headers}")
        for row in reader:
            fid = row[frag_col].strip()
            pid = row['parent_id'].strip()
            pos = int(row['site_pos'])
            data.append((fid, pid, pos))
    return data


def extract_window_from_topo(topo: str, site_pos: int, window: int = WINDOW, pad_char: str = PAD_CHAR) -> str:
    """Extract window of length `window` centered at site_pos (1-based). Pad with pad_char if OOB."""
    L = len(topo)
    start = site_pos - HALF  # 1-based
    end = site_pos + HALF
    out_chars = []
    for p in range(start, end + 1):
        if 1 <= p <= L:
            ch = topo[p - 1].upper()
            out_chars.append(ch if ch in TOPO_MAP else pad_char)
        else:
            out_chars.append(pad_char)
    return ''.join(out_chars)


def topo_to_onehot_matrix(topo_window: str) -> np.ndarray:
    """Convert topo window string (length WINDOW) to one-hot matrix (WINDOW,4)"""
    arr = np.zeros((WINDOW, 4), dtype=np.uint8)
    for i, ch in enumerate(topo_window):
        idx = TOPO_MAP.get(ch.upper(), TOPO_MAP[PAD_CHAR])
        arr[i, idx] = 1
    return arr


def load_esm2_embeddings(path: str, esm_ids_file: Optional[str] = None) -> Dict:
    """Load esm2 embeddings from .npy or .npz. Return {'embeddings': arr, 'ids': Optional[list]}.
    If .npy + esm_ids_file provided, read ids from that file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"ESM2 file not found: {path}")
    out: Dict = {}
    if p.suffix == '.npy':
        emb = np.load(str(p), allow_pickle=False)
        out['embeddings'] = emb
        if esm_ids_file:
            ids = [line.strip() for line in Path(esm_ids_file).read_text().splitlines() if line.strip()]
            out['ids'] = ids
        return out
    elif p.suffix == '.npz':
        npz = np.load(str(p), allow_pickle=True)
        if 'ids' in npz:
            out['ids'] = [str(x) for x in npz['ids'].tolist()]
        # try common keys for embeddings
        if 'embeddings' in npz:
            out['embeddings'] = npz['embeddings']
        elif 'esm2' in npz:
            out['embeddings'] = npz['esm2']
        elif 'emb' in npz:
            out['embeddings'] = npz['emb']
        else:
            # take first array-like key that's not 'ids'
            for k in npz.files:
                if k != 'ids':
                    out['embeddings'] = npz[k]
                    break
        return out
    else:
        raise ValueError("ESM2 file must be .npy or .npz")


def reorder_embeddings_by_fragments(esm_dict: Dict, fragments: List[Tuple[str, str, int]]) -> np.ndarray:
    """Return embeddings array ordered to match fragments list.

    Cases handled:
      - If embeddings shape == (N_fragments, WINDOW, C) and N_fragments == len(fragments): assume aligned, return.
      - If esm_dict contains 'ids' and embeddings shape == (N_proteins, L, C): extract windows per parent_id.
      - If esm_dict has embeddings (N, L, C) with N == len(fragments) and L == WINDOW: assume aligned and return.
    """
    emb = esm_dict.get('embeddings')
    esm_ids = esm_dict.get('ids', None)
    N = len(fragments)

    if emb is None:
        raise ValueError("ESM embeddings not found in provided file.")

    # Already per-fragment aligned
    if emb.ndim == 3 and emb.shape[0] == N and emb.shape[1] == WINDOW:
        return emb.astype(np.float32)

    # Per-protein embeddings + ids provided -> extract windows
    if esm_ids is not None and emb.ndim == 3:
        id_to_idx = {sid: i for i, sid in enumerate(esm_ids)}
        C = emb.shape[2]
        out = np.zeros((N, WINDOW, C), dtype=emb.dtype)
        for i, (fid, pid, pos) in enumerate(fragments):
            if pid in id_to_idx:
                pidx = id_to_idx[pid]
                prot_emb = emb[pidx]  # (L, C)
                Lp = prot_emb.shape[0]
                start = pos - HALF - 1  # 0-based
                end = pos + HALF - 1
                if start >= 0 and end < Lp:
                    out[i] = prot_emb[start:end + 1]
                else:
                    tmp = np.zeros((WINDOW, C), dtype=emb.dtype)
                    for j, p in enumerate(range(start, end + 1)):
                        if 0 <= p < Lp:
                            tmp[j] = prot_emb[p]
                        else:
                            tmp[j] = 0.0
                    out[i] = tmp
            else:
                # missing parent id -> zero pad
                out[i] = 0.0
        return out.astype(np.float32)

    # If emb has same first dim as fragments but different window, try trunc/pad
    if emb.ndim == 3 and emb.shape[0] == N:
        L = emb.shape[1]
        C = emb.shape[2]
        if L >= WINDOW:
            return emb[:, :WINDOW, :].astype(np.float32)
        else:
            # pad to WINDOW
            pad_width = ((0, 0), (0, WINDOW - L), (0, 0))
            return np.pad(emb, pad_width, mode='constant', constant_values=0.0).astype(np.float32)

    raise ValueError('Cannot reconcile ESM embeddings with fragments. Provide .npz with "ids" matching parent ids and embeddings as (N_proteins, L, C), or provide per-fragment embeddings of shape (N_fragments,31,C), or provide an --esm_ids file to map a .npy file.')


def main(args):
    # parse DeepTMHMM
    topo_dict = parse_deeptmhmm_file(args.deeptmhmm)
    print(f"[INFO] Parsed {len(topo_dict)} proteins from {args.deeptmhmm}")

    # read pos & neg fragment CSVs
    pos_fragments = read_fragments_from_csv(args.pos)
    neg_fragments = read_fragments_from_csv(args.neg)
    print(f"[INFO] Positive fragments: {len(pos_fragments)}, Negative fragments: {len(neg_fragments)}")

    # combine fragments with labels
    fragments: List[Tuple[str, str, int]] = []
    fragment_ids: List[str] = []
    parent_ids: List[str] = []
    site_positions: List[int] = []
    labels: List[int] = []

    for fid, pid, pos in pos_fragments:
        fragments.append((fid, pid, pos))
        fragment_ids.append(fid)
        parent_ids.append(pid)
        site_positions.append(pos)
        labels.append(1)
    for fid, pid, pos in neg_fragments:
        fragments.append((fid, pid, pos))
        fragment_ids.append(fid)
        parent_ids.append(pid)
        site_positions.append(pos)
        labels.append(0)

    N = len(fragments)
    if N == 0:
        raise SystemExit("[ERROR] No fragments were read from pos/neg files.")

    # build topo one-hot
    topo_out = np.zeros((N, WINDOW, 4), dtype=np.uint8)
    missing_parents = set()
    for i, (fid, pid, pos) in enumerate(fragments):
        parent_topo = topo_dict.get(pid)
        if parent_topo is None:
            missing_parents.add(pid)
            window = PAD_CHAR * WINDOW
        else:
            window = extract_window_from_topo(parent_topo, pos)
        topo_out[i] = topo_to_onehot_matrix(window)

    if missing_parents:
        print(f"[WARN] {len(missing_parents)} parent proteins missing in DeepTMHMM topo dict. Example: {list(missing_parents)[:5]}")

    # load esm embeddings (optionally with esm_ids mapping)
    esm_dict = load_esm2_embeddings(args.esm2, esm_ids_file=args.esm_ids)
    print(f"[INFO] Loaded ESM2 embeddings, keys present: {list(esm_dict.keys())}")

    esm_arr = reorder_embeddings_by_fragments(esm_dict, fragments)
    print(f"[INFO] Reordered/constructed ESM array with shape {esm_arr.shape}")

    # final check
    if topo_out.shape[0] != esm_arr.shape[0] or topo_out.shape[1] != esm_arr.shape[1]:
        raise ValueError(f"Shape mismatch after alignment: topo {topo_out.shape} vs esm {esm_arr.shape}")

    # save outputs
    out_prefix = args.out
    np.save(out_prefix + "_topo_onehot.npy", topo_out)
    np.save(out_prefix + "_esm2.npy", esm_arr)
    np.savez_compressed(out_prefix + ".npz",
                        fragment_ids=np.array(fragment_ids),
                        parent_ids=np.array(parent_ids),
                        site_pos=np.array(site_positions),
                        labels=np.array(labels, dtype=np.uint8),
                        topo=topo_out,
                        esm2=esm_arr)
    print(f"[INFO] Saved {out_prefix}_topo_onehot.npy (shape {topo_out.shape}), {out_prefix}_esm2.npy (shape {esm_arr.shape}), and {out_prefix}.npz")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare per-fragment DeepTMHMM one-hot (N,31,4) and align with ESM2 embeddings using positive/negative CSVs')
    parser.add_argument('--deeptmhmm', required=True, help='path to DeepTMHMM text output (FASTA-like or id topo lines)')
    parser.add_argument('--pos', required=True, help='path to positive fragments CSV (label=1)')
    parser.add_argument('--neg', required=True, help='path to negative fragments CSV (label=0)')
    parser.add_argument('--esm2', required=True, help='path to esm2 embeddings (.npy or .npz)')
    parser.add_argument('--esm_ids', required=False, help='optional: path to a file with esm ids (one per line) to map a .npy esm2 file')
    parser.add_argument('--out', default='combined_fragments', help='output prefix')
    args = parser.parse_args()
    main(args)

