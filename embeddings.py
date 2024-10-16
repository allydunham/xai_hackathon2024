import torch
from torch.utils.data import DataLoader

def fetch_esm_embeddings_batched(
    dataset,
    model,
    alphabet,
    batch_converter,
    representation_layer: int,
    device: str = "mps",
    batch_size: int = 32
) -> None:
    """
    Fetch ESM representations for specific positions in each sequence, processed in batches.

    Parameters:
    - dataset: The dataset containing sequences and their target positions.
    - model: The ESM model to use for generating embeddings.
    - alphabet: The alphabet used for tokenization.
    - batch_converter: Function to convert batches to tokens.
    - representation_layer: The layer from which to extract representations.
    - device: The device to run the model on ("cpu", "cuda", "mps").
    - batch_size: Number of sequences per batch.

    Raises:
    - KeyError: If 'target_position' is not found in any dataset item.
    - IndexError: If a target position is out of bounds for its sequence.
    """
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    total_batches = len(dataloader)
    
    for batch_idx, batch in enumerate(dataloader):
        domain_names = batch["domain_name"]
        sequences = batch["variant_aa_seq"]
        target_positions = batch["target_position"]  # Retrieve target positions
        
        # Prepare batch tuples for conversion
        batch_tuples = list(zip(domain_names, sequences))
        
        # Convert batch to tokens
        batch_labels, batch_strs, batch_tokens = batch_converter(batch_tuples)
        batch_tokens = batch_tokens.to(device)
        
        # Calculate actual sequence lengths (excluding padding)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
        
        with torch.no_grad():
            results = model(
                batch_tokens,
                repr_layers=[representation_layer],
                return_contacts=False
            )
        
        token_representations = results["representations"][representation_layer]
        
        for i in range(len(sequences)):
            seq_length = batch_lens[i].item()  # Total tokens including special tokens
            pos = target_positions[i].item()    # Assuming target_positions is a tensor
            
            # Validate target position
            # ESM models typically add start and end tokens, so adjust accordingly
            if pos < 1 or pos > (seq_length - 2):
                raise IndexError(
                    f"Target position {pos} is out of bounds for sequence '{domain_names[i]}' "
                    f"with length {seq_length - 2}."
                )
            
            # Adjust for special tokens (e.g., start token)
            token_idx = pos  # Assuming 1-based index and first token is at position 1
            
            # Extract the embedding at the target position
            embedding = token_representations[i, token_idx].float().to(device)
            
            # Store the embedding
            dataset.sequence_representations[batch_idx * batch_size + i] = embedding
        
        if device in ['cuda', 'mps']:
            torch.cuda.empty_cache()
        
        print(f"Fetched ESM representations for batch {batch_idx + 1} of {total_batches}")
    
    print(f"Completed fetching ESM representations for all {len(dataset)} items")
