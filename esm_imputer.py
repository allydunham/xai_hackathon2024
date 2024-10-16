#!/usr/bin/env python3
"""
Basic denoising diffusion model for imputing missing MAVE data
"""
import argparse
import sys

import numpy as np
import pandas as pd
from Bio import SeqIO

from embeddings import setup_esm, fetch_esm_embeddings_batched

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

        score_df = pd.melt(df, id_vars=["identifier_uniprot", "position"], var_name="mut", value_name="norm_score")
        score_df.mut = [i.split("_")[0] for i in score_df.mut]

        score_df["seq"] = [self.sub_seq(sequences[u], p, m) for u, p, m in zip(score_df.identifier_uniprot, score_df.position, score_df.mut)]

        data = score_df.merge(df, on = ["identifier_uniprot", "position"], how = "left").dropna(subset="mut")

        self.score = data.norm_score.to_numpy()
        self.position = data.position.to_numpy()
        self.seq = data.seq.to_numpy()
        self.score_matrix = data.filter(regex="_norm_score$").to_numpy()
        self.score_matrix = np.nan_to_num(self.score_matrix, 0, copy=False)

    def sub_seq(seq, position, mut):
        """
        Add a substitution to a sequence
        """
        mut = mut if not mut == "stop" else "*"
        return "".join([aa if not i == position else mut for i, aa in enumerate(seq)])

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
    pass

# def parse_args(arg_list=None):
#     """
#     Construct argument parser
#     """
#     parser = argparse.ArgumentParser(description=__doc__,
#                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

#     parser.add_argument("fastas", nargs="+", metavar="F",
#                         help="Files to count sequences from")

#     return parser.parse_args(arg_list)

if __name__ == "__main__":
    main()
