import os
import pandas as pd

ASA_MAX = {
    'A': 121, 'R': 265, 'N': 187, 'D': 187, 'C': 148,
    'Q': 214, 'E': 214, 'G': 97,  'H': 216, 'I': 195,
    'L': 191, 'K': 230, 'M': 203, 'F': 228, 'P': 154,
    'S': 143, 'T': 163, 'W': 264, 'Y': 255, 'V': 165
}

def parse_dssp_file(dssp_path):
    rows = []
    start = False
    with open(dssp_path) as f:
        for line in f:
            if line.startswith("  #  RESIDUE AA"):
                start = True
                continue
            if not start or len(line) < 80:
                continue
            try:
                res_num = int(line[5:10])
                aa = line[13]
                asa = float(line[34:38])
                rsa = asa / ASA_MAX.get(aa, 200)
                rows.append([res_num, aa, asa, rsa])
            except:
                continue
    return pd.DataFrame(rows, columns=["residue", "aa", "asa", "rsa"])

all_results = []

for file in os.listdir("./"):
    if file.endswith(".dssp"):
        name = file.split(".")[0]
        df = parse_dssp_file(os.path.join("./", file))
        df["protein"] = name
        all_results.append(df)

final = pd.concat(all_results)
final.to_csv("all_RSA_results.csv", index=False)

