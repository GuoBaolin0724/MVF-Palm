import os
import numpy as np
import pandas as pd
import iotbx.pdb
import mmtbx.model
from collections import defaultdict


def GetAtoms(pdbf):
    with open(pdbf, 'r') as f:
        return [line for line in f.readlines() if line.startswith(("ATOM", "HETATM"))]

def GetSiteCenter(pdbf, site_str):
    try:
        target_chain, target_resnum = site_str.split(':')
        target_resnum = int(target_resnum)
    except ValueError:
        print(f"Warning: Site format incorrect {site_str}，e.g. 'Chain:ResNum'")
        return None

    atomlines = GetAtoms(pdbf)
    coords = []
    for line in atomlines:
        chain = line[21]
        resnum = int(line[22: 26].strip())
        if chain == target_chain and resnum == target_resnum:
            x = float(line[30: 38].strip())
            y = float(line[38: 46].strip())
            z = float(line[46: 54].strip())
            coords.append([x, y, z])
    
    if not coords:
        print(f"Warning: {pdbf} no Site {site_str}")
        return None
    
    return np.mean(coords, axis=0) 

def GetPocAtoms(pdbf, cx, cy, cz, radius=12.0):
    atomlines = GetAtoms(pdbf)
    pocatoms = defaultdict(list)
    for line in atomlines:
        try:
            name = line[12:16].strip()
            resname = line[17:20].strip()
            chain = line[21]
            resnum = int(line[22:26].strip())
            ins_code = line[26].strip()
            x = float(line[30:38].strip())
            y = float(line[38:46].strip())
            z = float(line[46:54].strip())
            
            dist = np.sqrt((x-cx)**2 + (y-cy)**2 + (z-cz)**2)
            
            if dist <= radius:
                res_key = f"{chain}_{resname}_{resnum}{ins_code}"
                pocatoms["res_key"].append(res_key)
                pocatoms["chain"].append(chain)
                pocatoms["resname"].append(resname)
                pocatoms["resnum"].append(resnum)
                pocatoms["x"].append(x)
                pocatoms["y"].append(y)
                pocatoms["z"].append(z)
        except:
            continue
            
    return pd.DataFrame(pocatoms)

def FcalcAtAtoms(pdbf, atom_coords, resolution=2.0):
    pdb_inp = iotbx.pdb.input(file_name=pdbf)
    model = mmtbx.model.manager(model_input=pdb_inp)
    xrs = model.get_xray_structure()
    
    fcalc = xrs.structure_factors(d_min=resolution).f_calc()
    fft_map = fcalc.fft_map(resolution_factor=0.25)
    fft_map.apply_volume_scaling()
    fcalc_map_data = fft_map.real_map_unpadded()
    uc = fft_map.crystal_symmetry().unit_cell()
    
    densities = []
    for p in atom_coords:
        frac = uc.fractionalize(p)
        densities.append(max(0, fcalc_map_data.value_at_closest_grid_point(frac)))
    return np.array(densities)

def ProcessBatch(csv_path, pdb_dir, output_dir, temp_dir):
    for d in [output_dir, temp_dir]:
        if not os.path.exists(d): os.makedirs(d)

    df_index = pd.read_csv(csv_path)
    id_col = df_index.columns[0]
    site_col = df_index.columns[1]

    for index, row in df_index.iterrows():
        pdb_id = str(row[id_col])
        site_str = str(row[site_col])
        
        print(f"\n>>> Currently processing: {pdb_id} (Site: {site_str})")
        
        pdb_path = os.path.join(pdb_dir, f"{pdb_id}.pdb")
        if not os.path.exists(pdb_path):
            print(f"Error: The file {pdb_path} could not be found. Skipping...")
            continue

        center = GetSiteCenter(pdb_path, site_str)
        if center is None: continue
        cx, cy, cz = center

        temp_pdb = os.path.join(temp_dir, f"tmp_{pdb_id}.pdb")
        with open(temp_pdb, "w") as f_out:
            f_out.write("CRYST1  150.000  150.000  150.000  90.00  90.00  90.00 P 1\n")
            with open(pdb_path, "r") as f_in:
                for line in f_in:
                    if line.startswith(("ATOM", "HETATM")):
                        newline = line[:56]+"1.00"+line[60:61]+" 0.00"+line[66:]
                        f_out.write(newline)

        poc_df = GetPocAtoms(pdb_path, cx, cy, cz, radius=12.0)
        if poc_df.empty: continue

        coords = poc_df[["x", "y", "z"]].to_numpy()
        densities = FcalcAtAtoms(temp_pdb, coords)
        poc_df["density"] = densities

        res_result = poc_df.groupby(["chain", "resname", "resnum", "res_key"]).agg({
            "density": "sum"
        }).reset_index()

        output_name = f"{pdb_id}_{site_str.replace(':', '_')}.csv"
        res_result.to_csv(os.path.join(output_dir, output_name), index=False)
        print(f"Done! The result has been saved to: {output_name}")

        if os.path.exists(temp_pdb): os.remove(temp_pdb)


if __name__ == "__main__":
    INPUT_CSV = "./example/pos-tasks.csv" 
    PDB_FOLDER = "./pdbs"
    RESULT_FOLDER = "./pos_results/ED_data"
    TEMP_FOLDER = "./temp_work"

    ProcessBatch(INPUT_CSV, PDB_FOLDER, RESULT_FOLDER, TEMP_FOLDER)