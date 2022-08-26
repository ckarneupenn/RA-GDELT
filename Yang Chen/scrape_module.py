# -*- coding: utf-8 -*-
# =============================================================================
# Author: Yuxuan Wang
# Email: wangy49@seas.upenn.edu
# Date: 05-29-2022
# =============================================================================
# Modified by Zhaosen Guo for HPCC purposes 
"""
This module implements helper class that can help scraping articles from a given list of urls.
"""

# %%
import logging
from typing import List, Union, Tuple
from multiprocessing import Pool as ThreadPool

import warnings

from pprint import pprint

import re
import string
import unicodedata

import requests
from bs4 import BeautifulSoup
from ssl import SSLError

import trafilatura

# Error labels
EMPTY_TEXT = '[EMPTY_TEXT]'
EMPTY_HTML = '[EMPTY_HTML]'
ERROR_STATUS_CODE = {
    429, 499, 500, 502, 503, 504, 509, 
        520, 521, 522, 523, 524, 525, 526, 527, 530, 598
}


class NewsScraper:
    
    def __init__(
        self, 
        verbose: bool = True,
        n_cores: int = 4,
        clean_str: bool = True
    ) -> None:

        self.verbose = verbose
        self.n_cores = n_cores
        self.clean_str = clean_str
        
    def __repr__(self):
        return f'NewsScraper({self.verbose}, {self.n_cores}, {self.clean_str})'

    # @property
    # def verbose(self):
    #     return self.verbose

    # @property
    # def n_cores(self):
    #     return self.n_cores

    # @property
    # def clean_str(self):
    #     return self.clean_str

    def post_process(self, s: str) -> str:
        """String post-processing function, used to reduce noise.
            1. Convert all characters to ASCII
            2. Remove other irrelevant stuff like email address or external url
            3. Remove special symbols like newline character \\n"""
            
        # Normalize special chars
        i = s
        s = (unicodedata.normalize('NFKD', s)
                .encode('ascii', 'ignore').decode())

        # Remove irrelevant info
        s = re.sub(r'\S*@\S*\s?', '', s)     # Email
        s = re.sub(r'\S*https?:\S*', '', s)  # URL
        
        # Keep punctuation and words only
        pattern_keep = (string.punctuation + 
                            string.ascii_letters + 
                            string.digits + 
                            r' ')
        
        # Check if processed string is null
        s = re.sub(r'[^' + pattern_keep + r']+', '', s)
        if s == '':
            warnings.warn(f'@ {self.__class__.__name__}.preprocess() :: ' + 
                f'Null string after preprocessing; input: <{i}>')
            return '[NULL]'
        return s
    
    def fetch(self, urls: Union[List[str], str]) -> List[Tuple[str, str]]:
        """Main API, given a list of URLs, fetch HTML document and extract articles. Return 
            in the format of List[(HTML, extracted_article)]"""

        # Direct retrieval for single url
        if isinstance(urls, str):
            return [self._fetcher(urls)]
        
        # Multi-threading for fast-io
        elif isinstance(urls, list):
            with ThreadPool(self.n_cores) as p:
                return p.map(self._fetcher, urls)
        else:
            raise ValueError('@ NewsScraper.fetch() :: ' + 
                f'Invalid <urls> type {type(urls)}; only <str, List[str]> allowed')

    def _html2doc(self, html: str) -> str:
        """Given HTML texts, use trafilatura to extract news contents"""
        
        ret = trafilatura.extract(html, favor_precision=True, include_comments=False)
        if self.clean_str and ret is not None :
            return self.post_process(ret)
        return ret

    def _fetcher(self, url: str) -> Tuple[str, str]:

        # Simulate a user real browser 
        head = {'User-Agent': 'Mozilla/5.0'}
        try:
            if self.verbose:
                logging.info('@ NewsScraper.fetch() :: ' + 
                    f'try fetching from url <{url}>')
            resp = requests.get(url, headers=head)

        # Try no_ssl fetch
        except SSLError:
            try:
                resp = requests.get(url, headers=head, verify=False)
            except Exception as e:
                logging.warning('@ NewsScraper.fetch() :: ' + 
                    f'failed to fetch from url <{url}>: {e}')
                return (EMPTY_HTML, EMPTY_TEXT)
        
        # Other exception, such as invalid url
        except Exception as e:
            logging.warning('@ NewsScraper.fetch() :: ' + 
                f'failed to fetch from url <{url}>: {e}')
            return (EMPTY_HTML, EMPTY_TEXT)

        # Any error status
        if resp.status_code in ERROR_STATUS_CODE:
            return (EMPTY_HTML, EMPTY_TEXT)
        
        # Parse and return
        html = str(BeautifulSoup(resp.content, 'html5lib'))
        if html in {None, 'None', ''}:
            return ('', EMPTY_TEXT)
        return (html, self._html2doc(html))

# %%
# Sample usage
if __name__ == '__main__':
    
    # Init scraper and fetch
    scraper = NewsScraper()
    html, text = scraper.fetch('https://github.blog/2019-03-29-leader-spotlight-erin-spiceland/')[0]
    
    # Print results
    pprint(text)
    pprint(html)

# %%
