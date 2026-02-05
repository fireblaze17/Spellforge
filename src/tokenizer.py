# Building a character level tokenizer
# From what has been gathered, given limited data this is the best (also easiest to understand) implementation

def build_vocabulary(text):
    """
    Build character-to-ID and ID-to-character mappings from text.
    Initializing with special tokens:
    - <PAD>: padding token (will be used later)
    - <BOS>: beginning of sequence token
    - <EOS>: ending of sequence token
    - <UNK>: unknown token for unknown characters in vocab
    
    Returns (charToId, idToChar) tuple.
    Note: idToChar doesn't need to be a hashmap; can be an array for O(1) decoding using index.
    """
    charToId = {"<PAD>": 0, "<BOS>": 1, "<EOS>": 2, "<UNK>": 3}
    idToChar = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]
    
    nextId = 4  # Track ID per unique character
    for c in text:
        # If unique character is found, add to charToId hashmap and idToChar array
        if c not in charToId:
            charToId[c] = nextId
            idToChar.append(c)
            nextId += 1
    
    return charToId, idToChar


def load_spells(filepath):
    """
    Load spells from formatted text file.
    Need to split the txt file back into a list of strings (each string is a spell block).
    Returns list of spell strings.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    spells = ['<<< New Spell Forged >>>' + spell for spell in content.split('<<< New Spell Forged >>>')[1:]]
    return spells


def encode_spells(spells, charToId):
    """
    Encode spell strings to token sequences using charToId mapping.
    Returns list of encoded sequences.
    Uses a double for loop, encodedList is a list of lists
    goes through each string and using the vocab dictionary, converts the string into a list of tokens and appends/prepends
    BOS and EOS tokens, then appends to encodedList.
    """
    encodedList = []
    for text in spells:
        tempList = [charToId["<BOS>"]]
        for c in text:
            if c not in charToId:
                tempList.append(charToId["<UNK>"])
            else:
                tempList.append(charToId[c])
        tempList.append(charToId["<EOS>"])
        encodedList.append(tempList)
    
    return encodedList
