import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import re
from Bio.PDB import PDBParser
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_add_pool
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform

class ProteinDataIntegrator:
    def __init__(self, pdb_dir, dist_threshold=8.0):
        self.pdb_dir = pdb_dir
        self.dist_threshold = dist_threshold
        self.scaler = StandardScaler()
        self.parser = PDBParser(QUIET=True)

    def _extract_id_from_filename(self, filename):

        match = re.search(r'hbond_sidechain_([^_]+)', filename)
        if match:
            return match.group(1)
        return None

    def _get_pdb_coords(self, csv_filename):
        protein_id = self._extract_id_from_filename(csv_filename)
        if not protein_id:
            return None

        pdb_path = os.path.join(self.pdb_dir, f"{protein_id}.pdb")
        if not os.path.exists(pdb_path):
            pdb_path = os.path.join(self.pdb_dir, f"{protein_id}.PDB")
            if not os.path.exists(pdb_path):
                short_id = protein_id.split('-')[0]
                pdb_path = os.path.join(self.pdb_dir, f"{short_id}.pdb")
                if not os.path.exists(pdb_path):
                    return None
        
        coords = {}
        try:
            structure = self.parser.get_structure(protein_id, pdb_path)
            for model in structure:
                for chain in model:
                    for residue in chain:
                        res_num = residue.get_id()[1]
                        if 'CA' in residue:
                            coords[res_num] = residue['CA'].get_coord()
            return coords
        except Exception as e:
            print(f"Decode PDB {pdb_path} Error: {e}")
            return None

    def fit_global_scaler(self, csv_paths):
        all_vals = []
        for path in csv_paths:
            try:
                df = pd.read_csv(path)
                df.columns = [c.lower() for c in df.columns]
                if not df.empty:
                    all_vals.append(df[['density', 'acc']].values)
            except: continue
        if all_vals:
            self.scaler.fit(np.vstack(all_vals))

    def build_complete_graph(self, csv_path, label):
        try:
            df = pd.read_csv(csv_path)
            df.columns = [c.lower() for c in df.columns]
            
            pdb_coords = self._get_pdb_coords(os.path.basename(csv_path))
            if not pdb_coords:
                return None
            
            df['coord'] = df['resnum'].map(pdb_coords)
            df = df.dropna(subset=['coord']).reset_index(drop=True)
            if df.empty: return None
            
            xyz = np.vstack(df['coord'].values)
            
            node_cont = self.scaler.transform(df[['density', 'acc']].values)
            hbond_raw = np.clip(df['hbond_type'].values.astype(int), 0, 3)
            hbond_oh = np.eye(4)[hbond_raw]
            x = torch.tensor(np.hstack([node_cont, hbond_oh]), dtype=torch.float)

            dist_matrix = squareform(pdist(xyz))
            row, col = np.where((dist_matrix < self.dist_threshold) & (dist_matrix > 0))
            edge_index = torch.tensor(np.array([row, col]), dtype=torch.long)
            
            edge_attrs = []
            res_idx = df['resnum'].values
            for i, j in zip(row, col):
                inv_dist = 1.0 / (dist_matrix[i, j] + 1e-6)
                seq_dist = abs(res_idx[i] - res_idx[j])
                edge_attrs.append([inv_dist, float(seq_dist), 1.0 if seq_dist == 1 else 0.0])
            
            return Data(x=x, edge_index=edge_index, edge_attr=torch.tensor(edge_attrs, dtype=torch.float),
                        y=torch.tensor([label], dtype=torch.long), filename=os.path.basename(csv_path))
        except:
            return None


class ProteinGNN(torch.nn.Module):
    def __init__(self, hidden_channels=256):
        super().__init__()
        def mlp(in_c, out_c):
            return nn.Sequential(nn.Linear(in_c, out_c), nn.BatchNorm1d(out_c), nn.ReLU(), nn.Linear(out_c, out_c))
        self.conv1 = GINEConv(mlp(6, hidden_channels), edge_dim=3)
        self.conv2 = GINEConv(mlp(hidden_channels, hidden_channels), edge_dim=3)
        self.conv3 = GINEConv(mlp(hidden_channels, hidden_channels), edge_dim=3)
        self.fc = nn.Sequential(nn.Linear(hidden_channels, 256), nn.ReLU(), nn.Dropout(0.2))
        self.out = nn.Linear(256, 2)

    def forward(self, data, return_emb=False):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = F.relu(self.conv3(x, edge_index, edge_attr))
        emb = self.fc(global_add_pool(x, batch))
        return emb if return_emb else self.out(emb)

def main_pipeline(pos_csv_dir, neg_csv_dir, pdb_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    integrator = ProteinDataIntegrator(pdb_dir)
    
    pos_files = [os.path.join(pos_csv_dir, f) for f in os.listdir(pos_csv_dir) if f.endswith('.csv')]
    neg_files = [os.path.join(neg_csv_dir, f) for f in os.listdir(neg_csv_dir) if f.endswith('.csv')]
    
    print("The normalization parameters are being fitted...")
    integrator.fit_global_scaler(pos_files + neg_files)
    
    dataset = []
    print(" (CSV + PDB)...")
    for f in tqdm(pos_files, desc="positive sample"):
        g = integrator.build_complete_graph(f, 1)
        if g: dataset.append(g)
    for f in tqdm(neg_files, desc="negative sample"):
        g = integrator.build_complete_graph(f, 0)
        if g: dataset.append(g)
        
    if not dataset:
        print("\n[Error] Still unable to match any PDB file!")
        return

    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = ProteinGNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print(f"Starting training (total number of samples: {len(dataset)})...")
    model.train()
    for epoch in range(20):
        total_loss = 0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(batch), batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch+1)%5 == 0:
            print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")

    model.eval()
    all_data = []
    with torch.no_grad():
        for g in dataset:
            emb = model(g.to(device), return_emb=True)
            res = {'file': g.filename, 'label': g.y.item()}
            for i, v in enumerate(emb.cpu().numpy()[0]): res[f'd{i}'] = v
            all_data.append(res)
            
    pd.DataFrame(all_data).to_csv("protein_256d_features.csv", index=False)
    print("Done!")

if __name__ == "__main__":
    main_pipeline("./pos_results/hbond_data", 
				"./neg_results/hbond_data", 
				"./pdbs")
