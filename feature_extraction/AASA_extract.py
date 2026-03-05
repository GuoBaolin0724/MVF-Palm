import os
import pandas as pd
import re

def parse_dssp(dssp_path):
    dssp_data = []
    with open(dssp_path, 'r') as f:
        lines = f.readlines()
        
    start_idx = 0
    for i, line in enumerate(lines):
        if line.strip().startswith('#'):
            start_idx = i + 1
            break
            
    for line in lines[start_idx:]:
        if len(line) < 40: continue
        
        try:
            res_num_raw = line[5:10].strip()
            res_num = int(re.sub(r'\D', '', res_num_raw)) 
            chain = line[11].strip()
            acc = float(line[34:38].strip())
            
            dssp_data.append({
                'chain': chain,
                'resnum': res_num,
                'ACC': acc
            })
        except (ValueError, IndexError):
            continue
            
    return pd.DataFrame(dssp_data)

def merge_density_with_acc(csv_dir, dssp_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
    
    for csv_name in csv_files:
        pdb_id = csv_name.split('_')[0]
        
        dssp_path = os.path.join(dssp_dir, f"{pdb_id}.dssp")
        if not os.path.exists(dssp_path):
            dssp_path = os.path.join(dssp_dir, f"{pdb_id}.txt")
            
        if not os.path.exists(dssp_path):
            print(f"Warning: Could not find the DSSP file corresponding to {pdb_id}, skipping...")
            continue

        print(f"Currently processing: {csv_name} (Matching DSSP: {pdb_id})")

        density_df = pd.read_csv(os.path.join(csv_dir, csv_name))
        
        dssp_df = parse_dssp(dssp_path)
        
        if dssp_df.empty:
            print(f"Error: {dssp_path} failed or empty")
            continue

        density_df['resnum'] = density_df['resnum'].astype(int)
        dssp_df['resnum'] = dssp_df['resnum'].astype(int)
        
        merged_df = pd.merge(
            density_df, 
            dssp_df, 
            on=['chain', 'resnum'], 
            how='left'
        )

        output_path = os.path.join(output_dir, f"acc_{csv_name}")
        merged_df.to_csv(output_path, index=False)
        print(f"Successfully saved to: acc_{csv_name}")


if __name__ == "__main__":
    DENSITY_CSV_DIR = "./pos_results/ED_data"
    DSSP_FILES_DIR = "./dssp_results"
    FINAL_OUTPUT_DIR = "./pos_results/AASA_data"

    merge_density_with_acc(DENSITY_CSV_DIR, DSSP_FILES_DIR, FINAL_OUTPUT_DIR)