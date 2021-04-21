#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 16:32:47 2021

@author: lisatostrams
"""

import requests
from bs4 import BeautifulSoup

url = 'https://lokaalwijzer.nl/Search?Lng=5.109708&Lat=52.0949753&Sorting=2&Location=Utrecht%2C+Utrecht%2C+Nederland&Category='

page = requests.get(url)

soup = BeautifulSoup(page.content, 'html.parser')
results = soup.find(id='searchResults')

#%%
import pandas as pd
searchresults = results.find_all('div', class_ = "col-sm-6 mb-5 hover-animate")
ids = [tag['data-marker-id'] for tag in results.select('div[data-marker-id]')]
stands = pd.DataFrame(index=ids,columns=['name','products','address','description'])

for s_id, res in zip(ids,searchresults):
    check = 'div class="col-sm-6 mb-5 hover-animate" data-marker-id="{}"'.format(s_id)
    if check in str(res):
        name=res.find_all('h6', class_ = "card-title")[0].text.strip()
        stands.loc[s_id]['name'] = name
        products = res.find_all('p', class_ = "card-text text-muted text-sm")[0].text.strip()
        stands.loc[s_id]['products'] = products 
        
        
#%%
stand_url = 'https://lokaalwijzer.nl/Stand/Details/'

for s_id in ids:
    stand_page = requests.get(stand_url+s_id)
    soup = BeautifulSoup(stand_page.content, 'html.parser')
    adres=soup.find_all('p', class_ = "")[0].text.strip()
    stands.loc[s_id]['address']=adres
    omschrijving =soup.find_all('p', class_ = "text-muted")[0].text.strip()
    stands.loc[s_id]['description']=omschrijving

#%%

stands.to_csv('data/producenten_scraped/lokaalwijzer_punt_nl_utrecht.csv')