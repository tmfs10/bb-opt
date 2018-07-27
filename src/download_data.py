import numpy as np
import pandas as pd
from subprocess import run
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect as get_fingerprint
from os.path import realpath, join

data_dir = realpath(join(__file__, "../../data"))
fname = join(data_dir, "malaria.xlsx")

# see https://www.mmv.org/research-development/open-source-research/open-access-malaria-box/malaria-box-supporting-information for more info
run(
    f"wget -O {fname} https://www.mmv.org/sites/default/files/uploads/docs/RandD/Dataset.xlsx".split(
        " "
    ),
    check=True,
)

data = (
    pd.read_excel(fname)
    .drop(columns=["Index"])
    .rename(columns={"Canonical_Smiles": "smile", "Activity (EC50 uM)": "ec50"})
    .rename(columns=lambda name: name.lower())
)

data.to_csv(join(data_dir, "malaria.csv"), index=False)
run(f"rm {fname}".split(" "), check=True)

# generate Morgan fingerprints as features
bond_radius = 2
n_bits = 512

molecules = [MolFromSmiles(smile) for smile in data.smile]
fingerprints = [
    get_fingerprint(mol, radius=bond_radius, nBits=n_bits) for mol in molecules
]
fingerprints = np.array([[int(i) for i in fp.ToBitString()] for fp in fingerprints])
np.save(join(data_dir, "fingerprints.npy"), fingerprints)
