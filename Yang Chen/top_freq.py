"""
=============================================================================
Based on the given dataset, provide proper frequency counts for 
top entities (persons & organizations), 
combine them to an entity list with frequency.

# Author:   Zhaosen Guo
# Email:    zsguo@seas.upenn.edu
# Date:     06-01-2022
=============================================================================
"""


import pandas as pd


def main():
    """"""
    
    # filter by condition
    df = pd.read_pickle("AO_loc_only.pkl")

    # ORG
    flattened_list = pd.Series([element for lst in df['organizations'] for element in lst if element != 'NA'])
    freq_df = flattened_list.value_counts().sort_index().rename_axis("entity").reset_index(name='n').sort_values('n', ascending=False).reset_index(drop=True)
    
    org_freq = freq_df.head(1000).copy()
    org_freq['org_flag'] = True    

    # PPL
    flattened_list = pd.Series([element for lst in df['persons'] for element in lst if element != 'NA'])
    freq_df = flattened_list.value_counts().sort_index().rename_axis("entity").reset_index(name='n').sort_values('n', ascending=False).reset_index(drop=True)
    
    ppl_freq = freq_df.head(1000).copy()
    ppl_freq['org_flag'] = False    

    out_df = pd.concat([org_freq, ppl_freq]).sort_values(by='n', ascending=False).reset_index(drop=True)
    
    out_df['entity'] = "Angola " + out_df["entity"]

    for e in ["url1","url2","url3","url4","url5","title1","title2","title3","title4","title5","description1","description2","description3","description4","description5"]:
        out_df[e] = ""

    out_df.to_csv('AO_top2k_search.csv', index=False)

if __name__ == '__main__':
    main()    
