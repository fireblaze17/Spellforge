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
    
        
    
