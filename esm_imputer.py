#!/usr/bin/env python3
"""
Basic denoising diffusion model for imputing missing MAVE data
"""
import numpy as np
import pandas as pd
from Bio import SeqIO

from embeddings import fetch_esm_embeddings_batched
from helpers import pad_variable_length_sequences

def sub_seq(seq, position, mut):
        """
        Add a substitution to a sequence
        """
        mut = mut if not mut == "stop" else "*"
        return "".join([aa if not i == position else mut for i, aa in enumerate(seq)])

class MAVELoader():
    """MAVE Dataset Loader"""
    def __init__(self):
        sequences = {
            i.id.split("|")[1]: str(i.seq) for i in SeqIO.parse("mavedb_uniprot.fa", "fasta")
        }

        mavedb = pd.read_csv("mavedb_singles_wide.tsv", sep="\t")

        df = mavedb[
            mavedb.identifier_uniprot.isin(sequences.keys())
            ].filter(regex="identifier_uniprot|position|_norm_score$")
        df = df.loc[[len(sequences[i]) <= 2048 for i in df.identifier_uniprot]]

        score_df = pd.melt(df, id_vars=["identifier_uniprot", "position"], var_name="mut", value_name="norm_score")
        score_df = score_df.dropna(subset="norm_score")
        score_df.mut = [i.split("_")[0] for i in score_df.mut]

        score_df["seq"] = [sub_seq(sequences[u], p, m) for u, p, m in zip(score_df.identifier_uniprot, score_df.position, score_df.mut)]

        data = score_df.merge(df, on = ["identifier_uniprot", "position"], how = "left")

        self.score = data.norm_score.to_numpy()
        self.position = data.position.to_numpy()
        self.seq = pad_variable_length_sequences(data.seq.to_numpy())
        self.score_matrix = data.filter(regex="_norm_score$").to_numpy()
        self.score_matrix = np.nan_to_num(self.score_matrix, 0)

    def __len__(self):
        return len(self.score)

    def __getitem__(self, index):
        return {
            "score": self.score[index],
            "position": self.position[index],
            "seq": self.seq[index],
            "score_vector": self.score_matrix[index,]
        }

class model:
    pass

def main():
    data = MAVELoader()
    print(data)

if __name__ == "__main__":
    main()
