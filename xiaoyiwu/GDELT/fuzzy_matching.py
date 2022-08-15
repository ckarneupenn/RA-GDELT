import pandas as pd
import numpy as np 
from cleanco import basename
from sklearn.feature_extraction import text
import pycountry
from fuzzywuzzy import fuzz
import py_stringsimjoin as ssj
from difflib import SequenceMatcher
from collections import Counter
import string
from cleanco import basename
import regex as re
import py_stringsimjoin as ssj
import pandas as pd
from fuzzywuzzy import fuzz
import jellyfish
import pyarrow as pa
import string
import unicodedata
import regex as re
import py_stringmatching as sm

# read data
ORBIS_INPUT = './input/NG_firm_names_lms.parquet' 
GDELT_INPUT = './input/Niger_GDELT_100.csv'
indata_orbis = pd.read_parquet(ORBIS_INPUT).head(1000)
indata_gdelt = pd.read_csv(GDELT_INPUT).head(100)

print('debug: read Orbis data parquet successfully')

# process ORBIS data
indata_orbis = indata_orbis.iloc[:,2:3].dropna()
indata_orbis['name_original'] = indata_orbis['name_internat']
indata_orbis['name'] = pd.DataFrame(indata_orbis['name_internat'].apply(str.lower))
outdata_orbis = indata_orbis[['name_original', 'name']]
outdata_orbis.head()

stop = text.ENGLISH_STOP_WORDS

def preprocess_nlp(row):
        row = row.lower()
        row = row.strip()
        row = re.sub(r'\(.*\)', '', row)
        row = row.translate(str .maketrans('', '', string.punctuation))
        row =' '.join(row.split())
        row = unicodedata.normalize('NFKD', row).encode('ASCII', 'ignore').decode()
        row =  ' '.join(word.lower() for word in row.split() if word not in stop)
        return row

outdata_orbis['name_clean']=outdata_orbis['name'].apply(preprocess_nlp)

outdata_orbis.to_csv('./output/orbis_list.csv')

indata_gdelt = indata_gdelt[['organizations']].dropna()

orgs_unextracted_gdelt = []

for index, row in indata_gdelt.iterrows():
    # row is a single-item list with a string surrounded
    # by curly braces. Extract the single item and remove
    # the surrounding curly braces.
    orgs_unextracted_gdelt.append(row[0][1:-1])

# The rows are json-like formatted strings that contain non-quoted
# information which includes company names, each of which can be extracted 
# via regex and be treated as a subrow.
orgs_extracted_gdelt = []

# The rows are json-like formatted strings that contain non-quoted
# information which includes company names, each of which can be extracted 
# via regex and be treated as a subrow.
for row in orgs_unextracted_gdelt:
    row = row.split('},')
    for subrow in row:
        match = re.findall(r'(?:n=)(.*)(?:,)', subrow)
        orgs_extracted_gdelt.append(match[0])

outdata_gdelt = pd.DataFrame(orgs_extracted_gdelt)
outdata_gdelt.rename(columns={0: 'name_gdelt'}, inplace=True)
outdata_gdelt['name_original'] = outdata_gdelt['name_gdelt']
outdata_gdelt['name_gdelt']=outdata_gdelt['name_gdelt'].apply(preprocess_nlp)

#remove country names
file = open("./input/country_spellings.txt")
lines = file.readlines()

country_ls=[]
for l in lines:
    line=l.strip()
    line_2=line.replace('"',"")
    pat = re.compile("sname(?:==|:)(.*?);")
    # print(line_2)
    for i in pat.findall(line_2):
        country_ls.append(i.lower())
    # break


for i in pycountry.countries:
    country_ls.append(i.alpha_2.lower())
    country_ls.append(i.alpha_3.lower())
    country_ls.append(i.name.lower())
len(country_ls)

country_ls.sort(reverse=True)
country_df=pd.DataFrame(data=country_ls,columns=['country_name'])
country_df.drop_duplicates(subset ='country_name',inplace=True)

outdata_gdelt=outdata_gdelt[~outdata_gdelt['name_gdelt'].isin(country_df['country_name'])]
print('debug: country names have been removed ------------------')
outdata_gdelt[outdata_gdelt['name_gdelt'].isin(country_df['country_name'])].to_csv('./output/removed_country_name.csv')

outdata_orbis.reset_index(inplace=True)
outdata_gdelt.reset_index(inplace=True)

#Join 2 tables using various similarity measure
ws = sm.WhitespaceTokenizer(return_set=True)

# distance join
output_pairs_distance_join = ssj.edit_distance_join(outdata_orbis, outdata_gdelt,
                                      'index', 'index', 
                                      'name_clean', 'name_gdelt', 
                                      50,
                                      l_out_attrs=['name_clean'], 
                                      r_out_attrs=['name_gdelt'],
                                      n_jobs =-1)

# Jaccard Join 
output_pairs_jaccard_join = ssj.jaccard_join(outdata_orbis, outdata_gdelt, 
                                             'index', 'index', 
                                             'name_clean', 'name_gdelt', 
                                             ws, 0.1, 
                                             l_out_attrs=['name_clean'], 
                                             r_out_attrs=['name_gdelt'],
                                             n_jobs=-1)
# Cosine Join 
output_pairs_cosine_join = ssj.cosine_join(outdata_orbis, outdata_gdelt, 
                                             'index', 'index', 
                                             'name_clean', 'name_gdelt', 
                                             ws, 0.1, 
                                             l_out_attrs=['name_clean'], 
                                             r_out_attrs=['name_gdelt'],
                                             n_jobs=-1)
# Dice Join 
output_pairs_dice_join = ssj.dice_join(outdata_orbis, outdata_gdelt, 
                                             'index', 'index', 
                                             'name_clean', 'name_gdelt', 
                                             ws, 0.1, 
                                             l_out_attrs=['name_clean'], 
                                             r_out_attrs=['name_gdelt'],
                                             n_jobs=-1)

 # overlap join 
output_pairs_overlap_join = ssj.overlap_join(outdata_orbis, outdata_gdelt, 
                                             'index', 'index', 
                                             'name_clean', 'name_gdelt', 
                                             ws, 0.1, 
                                             l_out_attrs=['name_clean'], 
                                             r_out_attrs=['name_gdelt'],
                                             n_jobs=-1)
# overlap coefficient join 
output_pairs_overlap_coefficient_join = ssj.overlap_coefficient_join(outdata_orbis, outdata_gdelt, 
                                             'index', 'index', 
                                             'name_clean', 'name_gdelt', 
                                             ws, 0.1, 
                                             l_out_attrs=['name_clean'], 
                                             r_out_attrs=['name_gdelt'],
                                             n_jobs=-1)
# master list
# To cross join, merge on a temporary key and then drop it.
outdata_gdelt['key'] = 1
outdata_orbis['key'] = 1

master_list = pd.merge(outdata_gdelt, outdata_orbis, on='key').drop('key', 1)
master_list.rename(columns={'name_x': 'name_gdelt', 
                             'name_original_x': 'name_original_gdelt', 
                             'name': 'name_orbis', 
                             'name_clean': 'name_clean_orbis', 
                             'name_original_y': 'name_original_orbis'}, 
                    inplace=True)
master_list.to_csv('./output/master_list.csv')

try:
    data = master_list
except:
    data = pd.read_csv('./output/master_list.csv')
    data.drop(columns='Unnamed: 0', inplace=True)
    
data = data.dropna() # To prevent errors processing matches.
# Get matches of names as well as meta information.
# This is where the heavy lifting happens.

display('Match processing will take some time...')
display(str(len(data)) + ' rows...')

# !pip install tqdm
from tqdm import tqdm
tqdm.pandas() # Introduces pd.apply_progress() for progress bars.

# Name comparisons. Run an apply() on two columns.
display('Calculating fuzz ratio for names...')
data['fuzz_ratio'] = data.progress_apply(lambda x: fuzz.ratio(x.name_gdelt, x.name_clean_orbis), axis=1)
display('Calculating fuzz partial ratio for names...')
data['fuzz_partial_ratio'] = data.progress_apply(lambda x: fuzz.partial_ratio(x.name_gdelt, x.name_clean_orbis), axis=1)
display('Calculating token sort ratio for names...')
data['fuzz_token_sort_ratio'] = data.progress_apply(lambda x: fuzz.token_sort_ratio(x.name_gdelt, x.name_clean_orbis), axis=1)
display('Calculating jaro distance for names...')
data['jaro_distance'] = data.progress_apply(lambda x: jellyfish.jaro_distance(x.name_gdelt, x.name_clean_orbis), axis=1)

# Metaphone generation.
display('Generating metaphones for uncleaned orbis names...')
data['metaphone_unclean_orbis'] = data['name_orbis'].progress_apply(jellyfish.metaphone)
display('Generating metaphones for cleaned orbis names...')
data['metaphone_clean_orbis'] = data['name_clean_orbis'].progress_apply(jellyfish.metaphone)
display('Generating metaphones for gdelt names...')
data['metaphone_gdelt'] = data['name_gdelt'].progress_apply(jellyfish.metaphone)

# Metaphone comparisons. Run an apply() on two columns.
display('Calculating fuzz ratio for metaphones...')
data['metaphone_fuzz_ratio'] = data.progress_apply(lambda x: fuzz.ratio(x.metaphone_gdelt, x.metaphone_clean_orbis), axis=1)
display('Calculating fuzz partial ratio for metaphones...')
data['metaphone_fuzz_partial_ratio'] = data.progress_apply(lambda x: fuzz.partial_ratio(x.metaphone_gdelt, x.metaphone_clean_orbis), axis=1)
display('Calculating token sort ratio for metaphones...')
data['metaphone_fuzz_token_sort_ratio'] = data.progress_apply(lambda x: fuzz.token_sort_ratio(x.metaphone_gdelt, x.metaphone_clean_orbis), axis=1)
display('Calculating jaro distance for metaphones...')
data['metaphone_jaro_distance'] = data.progress_apply(lambda x: jellyfish.jaro_distance(x.metaphone_gdelt, x.metaphone_clean_orbis), axis=1)

display('Done.')



#### py_stringsimjoin
# Edit distance join
data = pd.merge(data, 
                output_pairs_distance_join, 
                how='outer', 
                left_on=['index_x', 'index_y'], 
                right_on=['r_index', 'l_index'])

data.rename(columns={'_sim_score': 'sim_score_distance'}, inplace=True)

#### py_stringmatching
# Jaccard join
data = pd.merge(data, 
                output_pairs_jaccard_join, 
                how='outer', 
                left_on=['index_x', 'index_y'], 
                right_on=['r_index', 'l_index'])

data.rename(columns={'_sim_score': 'sim_score_jaccard'}, inplace=True)

# Cosine Join 
data = pd.merge(data, 
                output_pairs_cosine_join, 
                how='outer', 
                left_on=['index_x', 'index_y'], 
                right_on=['r_index', 'l_index'])

data.rename(columns={'_sim_score': 'sim_score_cosine'}, inplace=True)

# Dice Join 
data = pd.merge(data, 
                output_pairs_dice_join, 
                how='outer', 
                left_on=['index_x', 'index_y'], 
                right_on=['r_index', 'l_index'])

data.rename(columns={'_sim_score': 'sim_score_dice'}, inplace=True)

data = pd.merge(data, output_pairs_overlap_join, 
                how='outer', 
                left_on=['index_x', 'index_y'], 
                right_on=['r_index', 'l_index'])

data.rename(columns={'_sim_score': 'sim_score_overlap'}, inplace=True)

# Overlap coefficient join 
data = pd.merge(data, 
                output_pairs_overlap_coefficient_join, 
                how='outer', 
                left_on=['index_x', 'index_y'], 
                right_on=['r_index', 'l_index'])

data.rename(columns={'_sim_score': 'sim_score_overlap_coefficient'}, inplace=True)

data.to_csv('./output/matches_raw.csv')

try:
    indata = data
except:
    indata = pd.read_csv('./output/matches_raw.csv')
    indata.drop(columns=['Unnamed: 0'], inplace=True)


#Sort match data in a multindex and sort by name and score.
df_sorted = indata.set_index(['name_original_orbis', 'name_original_gdelt'])
df_sorted = df_sorted.sort_values(by=['name_original_orbis', 
                                      'fuzz_ratio', 
                                      'fuzz_partial_ratio', 
                                      'fuzz_token_sort_ratio'], 
                                  ascending=False)
df_sorted = df_sorted.sort_index()

df_sorted.to_csv('./output/matches_sorted.csv')

try:
    df_sorted
except:
    indata = pd.read_csv('./output/matches_sorted.csv')
    df_sorted = indata.set_index(['name_original_orbis', 'name_original_gdelt'])

df_sorted.head()


# Just in case we want to look at the df
# we should have the columns in a nice order.

df_unscored = df_sorted[[
    # 'acronym_gdelt', 
    # 'freq_gdelt', 
    'fuzz_ratio', 
    'fuzz_partial_ratio', 
    'fuzz_token_sort_ratio', 
    'jaro_distance', 
    'metaphone_unclean_orbis', 
    'metaphone_clean_orbis', 
    'metaphone_gdelt',
    'metaphone_jaro_distance',
    'metaphone_fuzz_ratio',
    'metaphone_fuzz_partial_ratio',
    'metaphone_fuzz_token_sort_ratio',
    'sim_score_distance',
    'sim_score_jaccard',
    'sim_score_cosine',
    'sim_score_dice',
    'sim_score_overlap',
    'sim_score_overlap_coefficient',
]]

df_scored = df_unscored

# An approach called "fuzz similarity"
# https://www.analyticsinsight.net/company-names-standardization-using-a-fuzzy-nlp-approach/
df_scored['fuzz_similarity'] = (2 * df_scored['fuzz_partial_ratio'] * df_scored['fuzz_token_sort_ratio']) / (df_scored['fuzz_partial_ratio'] + df_scored['fuzz_token_sort_ratio'])

# Cumulative scores.
df_scored['total_score_name'] = df_scored['fuzz_ratio'] + df_scored['fuzz_partial_ratio'] + df_scored['fuzz_token_sort_ratio']
df_scored['total_score_metaphone'] = df_scored['metaphone_fuzz_ratio'] + df_scored['metaphone_fuzz_partial_ratio'] + df_scored['metaphone_fuzz_token_sort_ratio']

# Save progress here to allow fast manipulation of matching below.
df_matches = df_scored

df_scored = df_unscored
# An approach called "fuzz similarity"
# https://www.analyticsinsight.net/company-names-standardization-using-a-fuzzy-nlp-approach/
df_scored['fuzz_similarity'] = (2 * df_scored['fuzz_partial_ratio'] * df_scored['fuzz_token_sort_ratio']) / (df_scored['fuzz_partial_ratio'] + df_scored['fuzz_token_sort_ratio'])

# Cumulative scores.
df_scored['total_score_name'] = df_scored['fuzz_ratio'] + df_scored['fuzz_partial_ratio'] + df_scored['fuzz_token_sort_ratio']
df_scored['total_score_metaphone'] = df_scored['metaphone_fuzz_ratio'] + df_scored['metaphone_fuzz_partial_ratio'] + df_scored['metaphone_fuzz_token_sort_ratio']
df_matches = df_scored
# Filter matches.
df_matches = df_matches[((df_matches['total_score_name'] > 280.0) & (df_matches['jaro_distance'] > 0.9))]
df_matches.to_csv('./output/matches_filtered.csv')
try:
    indata = df_matches
except:
    indata = pd.read_csv('./output/matches_filtered.csv')
    indata = indata.set_index(['name_original_orbis', 'name_original_gdelt'])

# Clean up the final output.
dataout = indata[['fuzz_similarity', 
                  'total_score_name', 
                  'total_score_metaphone', 
                #   'freq_gdelt', 
                  'jaro_distance', 
                  'metaphone_jaro_distance', 
                  'sim_score_distance',
                  'sim_score_jaccard',
                  'sim_score_cosine',
                  'sim_score_dice',
                  'sim_score_overlap',
                  'sim_score_overlap_coefficient',
                 ]]
dataout.to_csv('./output/OUTPUT.csv')
