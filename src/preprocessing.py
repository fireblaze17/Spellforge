# Starting of preprocessing phase
# The file has features: name, classes, level, school, cast_time, range, duration, verbal, somatic, material
# Out of all of these, we will only keep name, classes, school, range, duration, description

import pandas as pd
import numpy as np
import re

def preprocess_and_save(csv_path, output_path):
    """
    Load data as a pandas dataframe and prepare formatted spells.
    """
    # Loading data as a pandas dataframe
    data = pd.read_csv(csv_path)
    # Keeping only relevant columns
    data = data[['name', 'classes', 'school', 'range', 'duration', 'description']]
    # Dropping rows with missing values
    data = data.dropna()
    
    # Need to decide on a format for each spell
    # Format:
    # <<< New Spell Forged >>>
    #     Name: 
    #     Classes:
    #     School:
    #     Range:
    #     Duration:
    #     Description:
    #     <<< May it Serve You Well >>>
    
    # List to hold formatted spells
    formatted_spells = []
    
    # For loop to format spells and then append a string with the spells to the list
    for row in data.itertuples(index=False):
        # Clean description and keep only the first two sentences.
        cleaned_desc = re.sub(r"\s+", " ", row.description.replace("\r", " ").replace("\n", " ")).strip()
        desc_sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', cleaned_desc)
        truncated_desc = " ".join(desc_sentences[:2]).strip()
        
        spell_str = f'''<<< New Spell Forged >>>
  Name: {row.name}
  Classes: {row.classes}
  School: {row.school}
  Range: {row.range}
  Duration: {row.duration}
  Description: {truncated_desc}
<<< May it Serve You Well >>>'''
        formatted_spells.append(spell_str)
    
    # Join spells so each spell ends with ">>>" and next starts with "<<<"
    # This creates the pattern: ">>><<< New Spell..." 
    # After tokenization: ">>><EOS><BOS><<< New Spell..."
    # This puts EOS immediately after >>> without separating newlines
    formatted_spells_no_trail = []
    for i, spell in enumerate(formatted_spells):
        if i < len(formatted_spells) - 1:
            # For all spells except the last, remove any trailing newline and add direct connection
            spell_clean = spell.rstrip('\n')
            formatted_spells_no_trail.append(spell_clean)
        else:
            # Last spell keeps its format
            formatted_spells_no_trail.append(spell)
    
    finalSpells = "".join(formatted_spells_no_trail)
    # Writing to text file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(finalSpells)
    
    return len(formatted_spells)
