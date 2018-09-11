import numpy as np
import pandas as pd
import os
from subprocess import run
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect as get_fingerprint
from bb_opt.src.utils import get_path

data_root = get_path(__file__, "..", "..", "data")


def download_malaria():
    data_dir = get_path(data_root, "malaria")
    fname = get_path(data_dir, "malaria.xlsx")

    os.makedirs(data_dir, exist_ok=True)

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

    data.to_csv(get_path(data_dir, "malaria.csv"), index=False)
    run(f"rm {fname}".split(" "), check=True)

    # generate Morgan fingerprints as features
    bond_radius = 2
    n_bits = 512

    molecules = [MolFromSmiles(smile) for smile in data.smile]
    fingerprints = [
        get_fingerprint(mol, radius=bond_radius, nBits=n_bits) for mol in molecules
    ]
    fingerprints = np.array([[int(i) for i in fp.ToBitString()] for fp in fingerprints])

    np.save(get_path(data_dir, "inputs.npy"), fingerprints.astype(np.float32))
    np.save(get_path(data_dir, "labels.npy"), data.ec50.values.astype(np.float32))


def download_dna_binding():
    sub_dataset_names = ["CRX_REF_R1", "VSX1_G160D_R1"]

    for sub_dataset_name in sub_dataset_names:
        data_dir = get_path(data_root, "dna_binding", sub_dataset_name.lower())
        os.makedirs(data_dir, exist_ok=True)

        fname = get_path(data_dir, sub_dataset_name.lower() + ".tsv")
        run(
            f"wget -O {fname} https://worksheets.codalab.org/rest/bundles/0xaa332863c4574ec5a2a0cf40ccc1687d/contents/blob/{sub_dataset_name}_8mers.txt".split(
                " "
            ),
            check=True,
        )

        # based on the scale of the figure in the paper, they must be using the
        # "Median" column as the target value, not the "E-score" or "Z-score"
        # ones (they negate it as well, since they're doing minimization)

        df = pd.read_csv(fname, sep="\t", usecols=["8-mer", "Median"]).rename(
            columns={"8-mer": "seq", "Median": "affinity"}
        )

        inputs = pd.DataFrame(list(map(list, df.seq.tolist())))
        inputs = pd.get_dummies(inputs)
        inputs = inputs.values.astype(np.float32)

        # if you want 3D input instead of 2D
        # seq_len = 8
        # n_bases = 4
        # inputs = inputs.values.reshape(-1, seq_len, n_bases)

        labels = df.affinity.values.astype(np.float32)

        np.save(get_path(data_dir, "inputs.npy"), inputs)
        np.save(get_path(data_dir, "labels.npy"), labels)


def main():
    download_malaria()
    download_dna_binding()


if __name__ == "__main__":
    main()
