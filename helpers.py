def pad_variable_length_sequences(sequences):

    longest_sequence = len(max(sequences, key = len))
    assert longest_sequence <= 2048, "There are sequences longer than expected."
    sequences = [sequence.ljust(longest_sequence, "-") for sequence in sequences]

    return sequences
