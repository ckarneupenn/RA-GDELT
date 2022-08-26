# GDELT Project RA Training Material 
### ESG Lab, Wharton, UPenn
### Zhaosen Guo, zsguo@seas.upenn.edu
### Last modification: 2022/06/12

# 1. Introduction
This is a training guide for running appropriate python scripts to clean, process, and generate analytical data based on the [GDELT Project](https://www.gdeltproject.org/).  
  
This guide serves as a walk-through demo using the data collected for Angola, all the codes are reuseable, some would need slight modification in order to run for different file names. 

Codes and this markdown instruction file can be found at this GitHub [sub-folder](https://github.com/guozhaosengzs/gdelt/tree/main/io/AO_traing_demo).

System requirements:
- Python (ver > 3.8.1)
- Pandas  
--- For the above set up, please see video instructions [here](https://drive.google.com/file/d/1e7D5b2t0DrAEau-QG02gHOxhaM92Tlc7/view?usp=sharing). 


# 2. Download and Clean the Data
1. Create a new folder to contain the python scripts and data files necessary.  
1.1 If you know how to fork a git repository, please just fork & clone this repo to your local directory, therefore you do not have to manually download any scripts.
2. Go to this [Box link](https://upenn.box.com/s/fz8leo6ox41x342g01xvhgim3wl255t5) to download **angola.csv**. If you do not have access, please contact [Professor Anne Jamison](aj.egb@cbs.dk) and [Professor Tony He](tonyhe@upenn.edu).
3. Download [clean_concat_filter.py](https://github.com/guozhaosengzs/gdelt/blob/main/io/AO_traing_demo/clean_concat_filter.py) to the same folder. 
4. Open a terminal, `cd` to your folder. 
5. Run `python .\clean_concat_filter.py`, depending on your system, you might need to use different argument to invoke the right Python (e.g. Mac users might have to use `python3` instead of `python`).  
*Please note that because we are reading a large CSV file (~6GB), the ingestion and process of the data would be fairly slow, this step could take around up to 10 minutes to complete. That is why for some of the processed GDELT data in this project, we would be using [Pickle](https://pythonnumericalmethods.berkeley.edu/notebooks/chapter11.03-Pickle-Files.html#) files (**.pkl**), which is on-average 100 times more efficient compared to **.csv**. 
6. After the previous script is finished, you should be able to see 2 files created in your folder: **AO_full_cleaned.pkl** and **AO_loc_only.pkl**. The first file would be saved for later section to use, while we are going to use **AO_loc_only.pkl** in the next section to generate list of top-mentioned entities in our dataset.

# 3. Top-mentioned Entities
1. In the previous section, we ran a script to clean the GDELT data, while also producing a set of filtered data **AO_loc_only.pkl**. Since articles published on local news and website can be about the entire world, we filtered down the large dataset by keeping entries that only has "Angola" in the *locations* column. Now we can use this **.pkl** file to calculate the overall frequency of *organizations* and *persons* listed.
2. Download **[top_freq.py](top_freq.py)**, and run `python .\top_freq.py`, and the script will produce **AO_top2k_search.csv**.  
Notice that the last 15 columns are empty, that is because we  need to run them through Google Custom Search API and get the top 5 search results. 
3. Email Wharton's [Research Programming team](research-programming@wharton.upenn.edu) to request them to run a custom search through Google, also include your supervisor's name so they can bill the cost to appropriate research fund.  
***Currently, [Director Shawn Zamechek](amechek@wharton.upenn.edu) has a script ready to deploy, remember to mention or CC him on that email.
4. The finished results from the Research Computing team should be saved to the same folder as other files, alternatively, please refer to **[results_AO_top2k_search.csv](results_AO_top2k_search.csv)** that I uploaded in this repository. 

# 4. Entity Filtering and GDELT Record Filtering
1. Download the following file to the project folder:
- **[full_filter.py](full_filter.py)**, the main script of doing all the filtering.
- **[snp100.csv](snp100.csv)**, a list of the S&P 100 Companies.
- **[major_cities.csv](major_cities.csv)**, list of major cities that might be mistakenly considered as an entity in the GDELT data.
- **[categories.csv](categories.csv)**, list of occupation classification that helps to determine if the person is an entity of interest.
2. Required packages:  
Just like the previous system setup, run `pip install package-name` (non-Mac) or `pip3 install package-name` (Mac) for the following python packages.
    - typing
    - abc
    - tqdm
    - pprint
    - numpy
    - re
    - string
    - unicodedata
    - torch
    - datasets
    - transformers

3. Run `python full_filter.py`.   
****(Runtime Warning, this script takes on NLI/NLP tasks that can take more than 30min to run, depending on your device)*  
Script **[full_filter.py](full_filter.py)** accomplishes the following:  
    1. In `main()`, I first come up with a list of names of companies and cities that could potentially be distracting - that is the entities' articles we do not want to focus on in the GDELT data. Then, by checking in the 3 following subset: persons, organizations, and S&P100 companies, the script removes those names that was in the list. 
    2. In `occ_pred()`, the script takes a look of the persons subset of data. Based on the web search results acquired from the previous [Section 3.4](#3-top-mentioned-entities), now we are able to predict the occupation for each person, check if that occupation is important to our overall goal, then decide to keep or remove it from the entity list. The result is saved as **[occ_prediction.csv](occ_prediction.csv)**.
    3. Back in `main()`, the script would now merge the 3 list: top 120 person of interest, top 120 organization of interest, and S&P100 companies. All 3 lists have already been checked for noises in *step 1*. The final list are saved as **[entity_of_interest.csv](entity_of_interest.csv)**. 
    4. Filter the file **AO_full_cleaned.pkl** from [Section 2.6](#2-download-and-clean-the-data) by keeping only the entries that at least mentioned 1 entity from our entity-of-interest list. Save the filtered results as **AO_interest_filtered.pkl**.

# 5. Web Scraping and Result Storage
At this stage, since the number of entries in our GDELT data has the potential of going above 6-digits, and each entry has a unique web address, the web scraping task becomes to much to handle on our personal computer. Therefore, we are moving to [Wharton School High Performance Computing Cluster (HPCC)](https://research-it.wharton.upenn.edu/documentation/) to process it on dedicated clusters that are suitable for huge computation jobs. Here's the step that I encourage to follow for best practices, and remember, whenever you run into any problems with HPCC and cannot resolve it within an hour, it's always a good idea to reach out to Wharton's [Research Computing team](research-computing@wharton.upenn.edu) for help. 

1. Contact your supervisor to set up your Wharton account.
2. Go to [this link](https://apps.wharton.upenn.edu/research_it/apply/) to apply for your HPCC account & access. 
3. Set up the tools you need on your local computer by working with the info on this [page](https://research-it.wharton.upenn.edu/documentation/access/), so now you can access the HPCC remotely.
4. Reach out to the [Research Computing team](research-computing@wharton.upenn.edu) for help on the following 3 subjects: 
    1. Add you to your supervisor's project folder and create a folder under your name. (Please keep all the working files in that folder, so we don't lose it.) 
    2. Set up your virtual environment in the folder created above.
    3. How to submit a script using your virtual environment. This is an extremely important subject, because we need to submit our python scripts to the server for it to allocate CPUs and memories to run the job. Do not run scripts on your remote Desktop. You can also refer to [this page](https://research-it.wharton.upenn.edu/documentation/job-management/) for details. 
    4. Now that you have set up the environment, it's time to download all the files in the [hpcc folder of this repository](https://github.com/guozhaosengzs/gdelt/tree/main/io/AO_traing_demo/hpcc) to your project folder on HPCC, and upload the **AO_interest_filtered.pkl** generated in [Section 4.3.4](#4-entity-filtering-and-gdelt-record-filtering) to the same folder (this file is too big for git).  
    Alternatively, go locate the *esg_ra* folder, which should be your parent folder in step one, then proceed to go to */Zhaosen/gdelt/io/AO_traing_demo/hpcc/*. Copy the entire folder to your working directory, and you are all set. 
5. Required packages (install these on your virtual environment, you should already have all the required packages install on your HPCC environment from the previous steps):  
    - logging
    - warnings
    - multiprocessing 
    - requests
    - bs4 
    - ssl 
    - html5lib
    - trafilatura

***If you are trying to install some packages that was already included in the environment, you might get an error message. To check if your Python has the package, top up the HPCC terminal, activate your virtual environment, then start a interactive python session by using `python` and press enter. In there, try to enter `import <package names>`, if there's no error, that means you have already gotten the package. To exit the interactive python, just press *Ctrl* + *Z* on your keyboard. 

6. Locate **[web_scraping.py](https://github.com/guozhaosengzs/gdelt/blob/main/io/AO_traing_demo/hpcc/web_scraping.py)** and **[submit_web_scraping.sh](https://github.com/guozhaosengzs/gdelt/blob/main/io/AO_traing_demo/hpcc/submit_web_scraping.sh)**, the former script is the job we would like to submit, and the latter is the script that we use to submit this specific job to the HPCC server. To understand the **[submit_web_scraping.sh](https://github.com/guozhaosengzs/gdelt/blob/main/io/AO_traing_demo/hpcc/submit_web_scraping.sh)**, please read this [instruction](https://research-it.wharton.upenn.edu/documentation/job-management/). In the second file, pay attention to line 3:  
`#$ -l m_mem_free=20G`, our entire research group has 100G in total, so please adjust them according to the size of your tasks.   
Then, in line 7 and 8,   
`module load python/python-3.9.6`  
`module load gcc/gcc-11.1.0`   
make sure to load the correct python version of your python environment.   
Lastly, in line 9, `source ../../../../venv396/bin/activate`, make sure to change this path directs to your virtual environment's location, if you are having trouble, email [Research Computing team](research-computing@wharton.upenn.edu) for help.

7. With all the setup, now we can run our script. First, open up your terminal and enter `qlogin -now no`, to register your command. Then, in the prompt, `cd` to your working folder (i.e. the copy of folder *hpcc*). Lastly, run `qsub submit_web_scraping.sh`. When the job is done, if would send an email to you with the exist status. If you have just submitted your code and got an email a few minutes later with an `Exit Status = 1`, most likely you've encountered an error, go back to your folder and find the file for **AO_scrape_json.o\<Job ID Number\>**, it would contain all the output of the script. 

8. If all goes well, we should end up with a scraped folder of JSON files (I set the total number to be 30) in the *scraped* folder. 

# 6. Score Sentence-level Mentions for Entities
In this step, we will run [array jobs](https://research-it.wharton.upenn.edu/documentation/job-management/array-jobs/). Basically, we are deploying one script to 30 nodes in the hopes to process all 30 batches of our JSON file simultaneously. 
1. Required packages (in additional to all the previous requirements):
    - json
    - keybert
    - spacy
    - itertools  
    - also run this: `python3 -m spacy download en_core_web_sm`

*** If you are running into problems installing them, please check if they are already installed in your Python environment.

2. Use **[submit_score_array_job.sh](https://github.com/guozhaosengzs/gdelt/blob/main/io/AO_traing_demo/hpcc/submit_score_array_job.sh)** to submit the script **[score_array_job.py](https://github.com/guozhaosengzs/gdelt/blob/main/io/AO_traing_demo/hpcc/score_array_job.py)**. Notice that in line 7 and 12 of **submit_score_array_job.sh**  I have  
`#$ -t 1-30`  
`python -u score_array_job.py batch${SGE_TASK_ID}.json`,  
which means that I would like to submit 30 jobs to the server (array jobs), each running on a different batch number.  
Also notice on line 26 in **score_array_job.py**:  
`file_name = sys.argv[1]`, stands for accepting the first argument given to the script as the file name, so all 30 files would be processed.  
3. After all the codes are run, you should have 30 csv files in the *scored_results* folder. On your terminal, cd to that folder, and run the following argument  
`{ head -n1 batch1.csv; for f in batch*.csv; do tail -n+2 "$f"; done; } > AO_scored_edges.csv`.   
This would produce a csv file named **AO_scored_edges.csv** that combines all 30 smaller csv files. 
