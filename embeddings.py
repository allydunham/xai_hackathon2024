import esm
import torch
from torch.utils.data import DataLoader


def setup_esm():

    model, alphabet = esm.pretrained.esm1_t6_43M_UR50S()
    batch_converter = alphabet.get_batch_converter()
    model = model.to(device)
    model.eval()
    embedding_size = model.embed_dim
    n_layers = len(model.layers)
    
    return model, batch_converter, embedding_size, n_layers

def fetch_esm_embeddings_batched(dataset, model, alphabet, batch_converter, representation_layer: int, device: str = "mps", batch_size: int = 32) -> None:

    """
    Fetch ESM representations in batches, usually the most efficient method.
    Recommended for all but the smallest datasets.
    """

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    total_batches = len(dataloader)

    for batch_idx, batch in enumerate(dataloader):
        
        domain_names = batch["domain_name"]
        sequences = batch["variant_aa_seq"]

        batch_tuples = list(zip(domain_names, sequences))
        batch_labels, batch_strs, batch_tokens = batch_converter(batch_tuples)
        batch_tokens = batch_tokens.to(device)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
        
        with torch.no_grad():
            
            results = model(batch_tokens, repr_layers=[representation_layer], return_contacts=False)
            
        token_representations = results["representations"][representation_layer]

        for i, idx in enumerate(range(batch_idx * batch_size, min((batch_idx + 1) * batch_size, len(dataset)))):
            dataset.sequence_representations[idx] = token_representations[i, 1 : batch_lens[i] - 1].mean(0).float().to(device)

        if device == 'cuda' or device == 'mps':
            torch.cuda.empty_cache()

        print(f"Fetched ESM representations for batch {batch_idx + 1} of {total_batches}")

    print(f"Completed fetching ESM representations for all {len(dataset)} items")
