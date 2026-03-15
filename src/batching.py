# Batching and Padding for Transformer Training
# Takes encoded spells and prepares them for training

import torch
import random 


def pad_batch(batch, pad_token=0):
    if not isinstance(batch, list):
        raise TypeError("batch must be a list of token sequences")
    if len(batch) == 0:
        raise ValueError("batch cannot be empty")

    #looping through the batch, finding the sequence with the most tokens in the batch
    #then padding the spells in that batch to that length
    # more efficient padding than padding ALL spells to same length
    maxLength = 0
    for sequence in batch:
        if not isinstance(sequence, list):
            raise TypeError("each sequence in batch must be a list")
        if len(sequence) == 0:
            raise ValueError("sequences cannot be empty")
        if len(sequence) > maxLength:
            maxLength = len(sequence)
    
    #loop through the batch again
    #for every sequence, calculate how many padding tokens
    #add padding tokens with another for loop
    for sequence in batch: 
        padNeeded = maxLength - len(sequence)
        for i in range(padNeeded):
            sequence.append(pad_token)
            
    return torch.tensor(batch, dtype=torch.long)
    
            
    
    
    


def create_batches(encoded_spells, batch_size, sort_by_length=True, shuffle=True):
    if not isinstance(encoded_spells, list):
        raise TypeError("encoded_spells must be a list of token sequences")
    if len(encoded_spells) == 0:
        raise ValueError("encoded_spells cannot be empty")
    if not isinstance(batch_size, int):
        raise TypeError("batch_size must be an integer")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    #Deep copy of the encoded spells, because we are modifying the list
    #sorting by length of spells so that higher length spells are together
    
    spellData = [seq.copy() for seq in encoded_spells]
    if sort_by_length:
        spellData.sort(key=len)
    
    #appending the padding tokens and batching
    batches = []
    for i in range(0,len(spellData), batch_size):
        chunk = spellData[i:i+batch_size]
        batches.append(pad_batch(chunk))
    #shuffling the batches 
    if shuffle:
        random.shuffle(batches)
        
    return batches
    
        
    
