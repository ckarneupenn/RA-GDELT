"""
=============================================================================
Based on filtered GDELT data frame, scrape the web and save 
text, web, and meta- data in json format.

# Author:   Zhaosen Guo
# Email:    zsguo@seas.upenn.edu
# Date:     06-16-2022
=============================================================================
"""


# import sys
# import os

# CWD = os.getcwd()
# sys.path.insert(0, CWD[:-12])

import time
import json
import numpy as np
import pandas as pd
from scrape_module import NewsScraper

def main():

    df_large = pd.read_pickle("AO_interest_filtered.pkl")
    number_json_files = 30

    splitted_dfs = np.array_split(df_large, number_json_files)

    for j in range(number_json_files):
        print("#################################")

        df_cluster = splitted_dfs[j]
        
        ids = df_cluster.gkgrecordid.tolist()
        urls = df_cluster.documentidentifier.to_list()
        persons = df_cluster.p_int.to_list()
        orgs = df_cluster.o_int.to_list()

        file = {}

        for i in range(df_cluster.shape[0]):
            scraper = NewsScraper()
            print(f"Working on cluster {j}\t serial {i}")

            url = urls[i]
            entities = persons[i] + orgs[i]
            html, text = scraper.fetch(url)[0]

            file[ids[i]] = {
                'batch' : (j + 1),
                'serial' : i,
                'url' : url,
                'entities' : entities,
                'persons' : persons[i],
                'organizations' : orgs[i],
                'text' : text,
                'html' : html
            }

        file_name = "scraped/batch" + str(j + 1) + ".json"
        with open(file_name, "w") as outfile:
            json.dump(file, outfile, indent=4)
     

if __name__ == '__main__':
    start_time = time.time()
    main()
    print("----------- scrape_dwnld.py : %.2f hours -----------" % ((time.time() - start_time) / 3600 ))
