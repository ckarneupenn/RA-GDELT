"""
=============================================================================
Based on the entity frequency data, compose a "entity of interest" list of 
related entities: top persons, top companies, and  S&P top 100 companies,
excluding some. Lately, the frequency information will be used to re-filter 
the overall GDELT dataset.

# Author:   Zhaosen Guo
# Email:    zsguo@seas.upenn.edu
# Date:     06-10-2022
=============================================================================
"""

def occ_pred(persons_df):
    """ This is a function that works on occupation prediction, and decided
    if the peron should be marked and kept. """

    
    from typing import List, Union, Any, Dict
    from abc import ABC, abstractmethod

    from tqdm import tqdm
    from pprint import pprint

    import numpy as np
    import pandas as pd

    import re
    import string
    import unicodedata

    import torch
    from datasets import Dataset
    from transformers import pipeline

    BATCH_SIZE = 128
    MODEL_CARD = 'valhalla/distilbart-mnli-12-1'
    PIPELINE = pipeline(
        'zero-shot-classification', MODEL_CARD,
        device=(0 if torch.cuda.is_available() else -1)
    )


    def batched_prediction(batch, candidate_labels: List[str], col_text: str = 'text'):
        
        out = PIPELINE(batch[col_text], candidate_labels=candidate_labels)
        ret = {'predictions': out}
        return ret


    def clean_str(s: str) -> str:
        """String pre-processing function, used to reduce noise.
            1. Convert all characters to ASCII
            2. Remove other irrelevant stuff like email address or external url
            3. Remove special symbols like newline character \\n"""
            
        # Normalize special chars
        s = str(s)
        s = (unicodedata.normalize('NFKD', s)
                .encode('ascii', 'ignore').decode())

        # Remove irrelevant info
        s = re.sub(r'\S*@\S*\s?', '', s)     # Email
        s = re.sub(r'\S*https?:\S*', '', s)  # URL (http)
        s = re.sub(r'\S*www\.\S*', '', s)    # URL (www)
        
        # Keep punctuation and words only
        pattern_keep = (string.punctuation + 
                            string.ascii_letters + 
                            string.digits + 
                            r' ')
        return re.sub(r'[^' + pattern_keep + r']+', '', s)


    # Load occupation categories
    df_occ = pd.read_csv('categories.csv')
    df_occ.loc[:, 'occupation'] = df_occ.loc[:, 'occupation'].str.lower()

    # Load textual descriptions of interested entities
    df_ent = persons_df.copy()
    df_ent.loc[:, 'text'] = df_ent.loc[:, 'description1'].map(clean_str)

    # Convert from Pandas to Huggingface dataset and predict occupations
    df_ent = (Dataset
        .from_pandas(df_ent)
        .map(
            batched_prediction, 
            batched=True,
            batch_size=BATCH_SIZE,
            fn_kwargs={'candidate_labels': df_occ.occupation.unique().tolist()}
        )
        .to_pandas())

    occ_keep = {
        'politician',
        'businessperson',
        'journalist',
        'social activist',
        'extremist',
        'judge',
        'lawyer',
        'economist',
        'critic',
        'military personnel'
    }

    df_ent.loc[:, 'top1_label'] = df_ent.predictions.map(lambda d: d['labels'][0])
    df_ent.loc[:, 'top1_score'] = df_ent.predictions.map(lambda d: d['scores'][0])
    df_ent.loc[:, 'is_kept'] = df_ent.top1_label.map(lambda o: int(o in occ_keep))
    df_ent.to_csv('occ_prediction.csv', index=False) 

    return df_ent

def main():
    import pandas as pd

    # Get frequency web search results, remove country name
    web_df = pd.read_csv("results_AO_top2k_search.csv")
    web_df["entity"] = web_df["entity"].str[7:]

    persons_df = web_df.query("org_flag == False")
    orgs_df = web_df.query("org_flag == True")

    snp = pd.read_csv("snp100.csv").NAME.to_list()

    remove_list = ['eBay', 'Meta', 
        'Twitter', 'Facebook', 'Instagram', 'Bloomberg', 
        'Youtube', 'Associated Press', 'CNN', 'SoundCloud', 
        'African News Agency', 'LinkedIn', 'Yelp', 'Reuters', 'Netflix']  
    remove_list = [x.lower() for x in remove_list]

    geo_df = pd.read_csv('major_cities.csv')
    city_list = [x.lower() for x in geo_df.name.to_list()]
    country_list = [x.lower() for x in geo_df.country.unique()]

    remove_list += city_list + country_list

    # Remove Target List in Organization
    orgs_df['remove'] = orgs_df.entity.apply(lambda x: 1 if x.lower() in remove_list else 0)
    orgs_df = orgs_df.query("remove == 0")

    orgs_df.sort_values(by='n', ascending=False, inplace=True)
    orgs_df.reset_index(drop=True, inplace=True)

    snp = [s for s in snp if s.lower() not in remove_list]

    # Run person occupation & prediction 
    persons_pred_df = occ_pred(persons_df)
    # persons_pred_df = pd.read_csv('occ_prediction.csv')
    persons_pred_df = persons_pred_df.query("is_kept == 1")

    persons_pred_df.sort_values(by='n', ascending=False, inplace=True)
    persons_pred_df.reset_index(drop=True, inplace=True)

    # Get the top entities (120 in persons & organizations each), combine them to list
    org_top_list = orgs_df.head(120).entity.to_list()
    ppl_top_list = persons_pred_df.head(120).entity.to_list()

    finalist = org_top_list + ppl_top_list + snp
    pd.DataFrame({'entity': finalist}).to_csv("entity_of_interest.csv",index=False)
    
    # Filter the original GDELT dataframe, save the filtered file
    full_gdelt_df = pd.read_pickle("AO_full_cleaned.pkl")

    full_gdelt_df['p_int'] = full_gdelt_df.persons.apply(lambda x: [p for p in finalist if p in x ])
    full_gdelt_df['o_int'] = full_gdelt_df.organizations.apply(lambda x: [c for c in finalist if c in x])
    full_gdelt_df['keep'] = full_gdelt_df.apply(lambda row: 1 if (row.p_int != [] or row.o_int != []) else 0, axis=1)#row.p_int
    
    interest_df = full_gdelt_df.query('keep == 1')

    interest_df.to_pickle("AO_interest_filtered.pkl") #137,127 entries 

if __name__ == '__main__':
    main()    
