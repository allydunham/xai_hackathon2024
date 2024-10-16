#!/usr/bin/env python3
"""
Basic denoising diffusion model for imputing missing MAVE data
"""
import argparse
import sys

import numpy as np
import pandas as pd

from Bio import SeqIO

class MAVELoader():
    """MAVE Dataset Loader"""
    def __init__(self):
        self.sequences = {
            i.id.split("|")[1]: str(i.seq) for i in SeqIO.parse("mavedb_uniprot.fa", "fasta")
        }

        self.mavedb = pd.read_csv("mavedb_singles_wide.tsv", sep="\t")

        df = self.mavedb[self.mavedb.identifier_uniprot.isin(self.sequences.keys())]

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass

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