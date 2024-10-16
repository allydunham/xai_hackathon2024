#!/usr/bin/env python3
"""
Basic denoising diffusion model for imputing missing MAVE data
"""
import sys
import numpy as np
import pandas as pd
from Bio import SeqIO
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split

from embeddings import fetch_esm_embeddings_batched, setup_esm
from helpers import pad_variable_length_sequences
from downstream import ESMImputer, setup_model

def sub_seq(seq, position, mut):
        """
        Add a substitution to a sequence
        """
        mut = mut if not mut == "stop" else "*"
        return "".join([aa if not i == position else mut for i, aa in enumerate(seq)])

class MAVELoader(Dataset):
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
        self.esm = np.zeros((len(self.score), 2))

    def __len__(self):
        return len(self.score)

    def __getitem__(self, index):
        return {
            "score": self.score[index],
            "position": self.position[index],
            "seq": self.seq[index],
            "score_vector": self.score_matrix[index,],
            "esm": self.esm[index,]
        }

class ESMImputer:
    pass

def train(model, loss_fn, optimizer, train_loader, val_loader, epochs=1, path="models/"):
    best_vloss = float('inf')

    for i in range(epochs):
        print(f'EPOCH {i}')
        model.train(True)
        running_loss = 0.

        for j, data in enumerate(train_loader):
            inputs, outputs = data
            optimizer.zero_grad()
            predicted = model(inputs)
            loss = loss_fn(predicted, outputs)
            loss.backward()
            optimizer.step()
            running_loss += loss
            print(f'  batch {j} loss: {loss}')

        mean_loss = running_loss / (j + 1)

        model.eval()
        running_vloss = 0.
        with torch.no_grad():
            for j, vdata in enumerate(val_loader):
                vinput, voutput = vdata
                vpred = model(vinput)
                vloss = loss_fn(vpred, voutput)
                running_vloss += vloss

        mean_vloss = running_vloss / (j + 1)
        print(f'  EPOCH LOSS train: {mean_loss} val: {mean_vloss}')

        if mean_vloss < best_vloss:
            best_vloss = mean_vloss
            model_path = f"{path}/model_{i}"
            torch.save(model.state_dict(), model_path)

def main():
    print("Loading ESM2-35M", file=sys.stderr)
    esm, alphabet, batch_converter, embedding_size, n_layers = setup_esm()

    print("Importing MAVE Data", file=sys.stderr)
    data = MAVELoader()

    print("Generating ESM Representations", file=sys.stderr)
    fetch_esm_embeddings_batched(data, esm, alphabet, batch_converter, n_layers)

    print("Training Model", file=sys.stderr)
    model = ESMImputer()

    train_data, val_data = random_split(data, [0.9, 0.1])

    train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=10, shuffle=True)

    loss_fn = nn.L1Loss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    train(model, loss_fn, optimizer, train_loader, val_loader, 100, path = "models/nn")

    torch.save(model.state_dict(), "models/countries/model_final")




if __name__ == "__main__":
    main()
