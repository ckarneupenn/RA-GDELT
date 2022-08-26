"""
=============================================================================
Clean the GDELT data, concat yearly data into one, 
filter on location for the specific country only.

# Author:   Zhaosen Guo
# Email:    zsguo@seas.upenn.edu
# Date:     06-01-2022
=============================================================================
"""

import pandas as pd

# Read the GDELT File
df = pd.read_csv("angola.csv")

# Keep only the relevant columns
df = df[['gkgrecordid',
        'documentidentifier',
        'locations',
        'organizations',
        'persons',
        'tone']]    

# Parse & clean 'locations
df['locations'].fillna('NA', inplace=True)
df['locations'] = df.locations.apply(lambda x: x.replace('{', '').replace('}', '').replace('[', '').replace(']', ''))
df['locations'] = df.locations.apply(lambda x: x.split(', '))
df['locations'] = df.locations.apply(lambda x: [s for s in x if "location_fullname=" in s])
df['locations'] = df.locations.apply(lambda x: [s.replace("location_fullname=", '') for s in x])

# Parse & clean 'organizations
df['organizations'].fillna('NA', inplace=True)
df['organizations'] = df.organizations.apply(lambda x: x.replace('{', '').replace('}', '').replace('[', '').replace(']', '').replace(' organization=', '').replace('organization=', ''))
df['organizations'] = df.organizations.apply(lambda x: x.split(','))
df['organizations'] = df.organizations.apply(lambda x: [s for s in x if " character_offset=" not in s])

# Parse & clean 'persons
df['persons'].fillna('NA', inplace=True)
df['persons'] = df.persons.apply(lambda x: x.replace('{', '').replace('}', '').replace('[', '').replace(']', '').replace(' person=', '').replace('person=', ''))
df['persons'] = df.persons.apply(lambda x: x.split(','))
df['persons'] = df.persons.apply(lambda x: [s for s in x if " character_offset=" not in s])

# Parse & clean 'tones 
df['tone'] = df.tone.apply(lambda x: x[1:-1])
df['tone'] = df.tone.apply(lambda x: x.split(', '))
df['tone'] = df.tone.apply(lambda x: [s for s in x if "tone=" in s])
df['tone'] = df.tone.apply(lambda x: float(x[0].replace("tone=", '')))

# Save the cleaned full data 
df.reset_index(drop=True, inplace=True)
df.to_pickle("AO_full_cleaned.pkl")

# Filtering for AO-only, for frequency calculation
df['location_set'] = df.locations.apply(lambda x: set(x))
df = df[df['location_set'] == {"Angola"}]
df.drop(columns=['location_set', 'locations'], inplace=True)

# Save the location filtered version
df.reset_index(drop=True, inplace=True)
df.to_pickle("AO_loc_only.pkl")
