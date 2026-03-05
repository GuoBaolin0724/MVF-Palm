import os
import pandas as pd
from Bio.PDB import PDBParser, NeighborSearch

def get_sidechain_hbond_status(pdb_path, target_res_keys):
    parser = PDBParser(QUIET=True)
    if not os.path.exists(pdb_path):
        return {}
    
    structure = parser.get_structure('protein', pdb_path)
    
    mainchain_atoms = {'N', 'CA', 'C', 'O', 'OXT'}
    all_atoms = list(structure.get_atoms())
    sidechain_atoms = [a for a in all_atoms if a.get_name() not in mainchain_atoms]
    
    side_donors = {'NG', 'ND1', 'ND2', 'NE', 'NE1', 'NE2', 'NH1', 'NH2', 'NZ', 'OG', 'OG1', 'OH', 'SG'}
    side_acceptors = {'OD1', 'OD2', 'OE1', 'OE2', 'OG', 'OG1', 'OH', 'ND1', 'NE2'}

    ns = NeighborSearch(sidechain_atoms)
    
    hbond_results = {key: 0 for key in target_res_keys}

    for atom in sidechain_atoms:
        res = atom.get_parent()
        chain_id = res.get_parent().get_id()
        res_id = res.get_id()[1] 
        res_key = (chain_id, res_id)

        if res_key not in target_res_keys:
            continue

        neighbors = ns.search(atom.get_coord(), 3.5)
        
        is_donor = False
        is_acceptor = False

        for nb_atom in neighbors:
            nb_res = nb_atom.get_parent()
            if nb_res == res:
                continue
            
            atom_name = atom.get_name()
            nb_name = nb_atom.get_name()

            if atom_name in side_donors and nb_name in side_acceptors:
                is_donor = True
            if atom_name in side_acceptors and nb_name in side_donors:
                is_acceptor = True

        current_status = hbond_results[res_key]
        if is_donor and is_acceptor:
            hbond_results[res_key] = 3
        elif is_donor:
            hbond_results[res_key] = 1 if current_status != 2 else 3
        elif is_acceptor:
            hbond_results[res_key] = 2 if current_status != 1 else 3

    return hbond_results

def process_batch_hbond(csv_dir, pdb_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    csv_files = [f for f in os.listdir(csv_dir) if f.startswith('acc_') and f.endswith('.csv')]

    for csv_file in csv_files:
        parts = csv_file.split('_')
        pdb_id = parts[1]
        
        pdb_path = os.path.join(pdb_dir, f"{pdb_id}.pdb")
        if not os.path.exists(pdb_path):
            print(f"Skipping {csv_file}: PDB not found.")
            continue

        print(f"Processing Sidechain H-Bond: {csv_file}")
        
        df = pd.read_csv(os.path.join(csv_dir, csv_file))

        target_keys = set(zip(df['chain'].astype(str), df['resnum'].astype(int)))
        
        hbond_map = get_sidechain_hbond_status(pdb_path, target_keys)
        
        df['hbond_type'] = df.apply(lambda x: hbond_map.get((str(x['chain']), int(x['resnum'])), 0), axis=1)

        out_name = csv_file.replace('acc_', 'hbond_sidechain_')
        df.to_csv(os.path.join(output_dir, out_name), index=False)


if __name__ == "__main__":
    ACC_CSV_DIR = "./pos_results/AASA_data"  
    PDB_FILES_DIR = "./pdbs"
    HBOND_OUTPUT_DIR = "./pos_results/hbond_data" 

    process_batch_hbond(ACC_CSV_DIR, PDB_FILES_DIR, HBOND_OUTPUT_DIR)