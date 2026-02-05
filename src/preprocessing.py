# Starting of preprocessing phase
# The file has features: name, classes, level, school, cast_time, range, duration, verbal, somatic, material
# Out of all of these, we will only keep name, classes, school, range, duration, description

import pandas as pd
import numpy as np
import torch
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
        # Clean description: remove newlines, handle carriage returns, convert semicolons to periods
        cleaned_desc = row.description.replace('\n', ' ').replace('\r', '').replace(';', '.')
        # Split by sentence boundaries (period followed by space and capital letter) and take first 2
        desc_sentences = '. '.join(re.split(r'\.\s+(?=[A-Z])', cleaned_desc)[:2]) + '.'
        
        spell_str = f'''<<< New Spell Forged >>>
  Name: {row.name}
  Classes: {row.classes}
  School: {row.school}
  Range: {row.range}
  Duration: {row.duration}
  Description: {desc_sentences}
<<< May it Serve You Well >>>
'''
        formatted_spells.append(spell_str)
    
    # Appending to list with two newlines between each spell block
    finalSpells = "\n\n".join(formatted_spells)
    # Writing to text file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(finalSpells)
    
    return len(formatted_spells)
