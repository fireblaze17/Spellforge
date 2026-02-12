# Batching and Padding for Transformer Training
# Takes encoded spells and prepares them for training

import numpy as np
import random 


def pad_batch(batch, pad_token=0):

    #looping through the batch, finding the sequence with the most tokens in the batch
    #then padding the spells in that batch to that length
    # more efficient padding than padding ALL spells to same length
    maxLength = 0
    for sequence in batch:
        if len(sequence) > maxLength:
            maxLength = len(sequence)
    
    #loop through the batch again
    #for every sequence, calculate how many padding tokens
    #add padding tokens with another for loop
    for sequence in batch: 
        padNeeded = maxLength - len(sequence)
        for i in range(padNeeded):
            sequence.append(pad_token)
            
    return np.array(batch)
    
            
    
    
    


def create_batches(encoded_spells, batch_size, sort_by_length=True, shuffle=True):
    """
    Split encoded spells into padded batches.
    
    Args:
        encoded_spells: List of all encoded sequences (List[List[int]])
        batch_size: Number of sequences per batch
        sort_by_length: Whether to sort by length before batching (minimizes padding waste)
        shuffle: Whether to shuffle batch order after batching (recommended for training)
    
    Returns:
        List of padded batches, each of shape (batch_size, max_len_in_batch)
        Note: Last batch may have fewer than batch_size sequences
    
    Steps:
        1. Make a copy of the data (to avoid modifying original)
        2. If sort_by_length: sort sequences by length (groups similar lengths together)
        3. Split into chunks of batch_size
        4. Pad each chunk using pad_batch()
        5. If shuffle: shuffle the ORDER of batches (not within batches)
        6. Return list of padded batches
    """
    spellData = [seq.copy() for seq in encoded_spells]
    if sort_by_length:
        spellData.sort(key=len)
    
    
    batches = []
    for i in range(0,len(spellData), batch_size):
        chunk = spellData[i:i+batch_size]
        batches.append(pad_batch(chunk))
    
    if shuffle:
        random.shuffle(batches)
        
    return batches
    
        
    
