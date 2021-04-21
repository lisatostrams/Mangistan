#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 17:49:11 2021

@author: lisatostrams
"""


import requests
from bs4 import BeautifulSoup

url = 'https://milieuwijzerutrecht.nl/rubriek/eten-en-drinken/boerderijwinkels/'

page = requests.get(url)

soup = BeautifulSoup(page.content, 'html.parser')
results = soup.find_all('div',class_='blok resp')
#%%
import re
import pandas as pd

stands = pd.DataFrame(index=range(0,len(results)),columns=['name','products','address','description'])


for i,res in enumerate(results):
    naam = res.find_all('div',class_='totaalblok')[0].find_all('a')[0].text
    stands.loc[i]['name']=naam
    stand_url = [tag['href'] for tag in res.select('a[href]')][0]
    stand_page = requests.get(stand_url)
    stand_soup = BeautifulSoup(stand_page.content, 'html.parser')
    blok_text = stand_soup.find_all('div',class_='blokrechts')[0].text
    regels_text = [regel for regel in blok_text.split('\n') if len(regel)>0]
    adres_regel = ''
    for regel in regels_text:
        postcode=re.findall(r"\b\d{4}[\s?][a-zA-Z]{2}\b", regel)
        if len(postcode)>0:
            adres_regel=regel
            break
        if 'Nederland' in regel or 'Netherlands' in regel:
            adres_regel=regel
            break
    if len(adres_regel)==0:
        adres_regel = 'landelijk'
    stands.loc[i]['address'] = adres_regel
    stands.loc[i]['products'] = regels_text[1]
    stands.loc[i]['description'] = '\n'.join(regels_text)
    
#%%

re.findall(r"\b\d{4}[a-zA-Z]{2}[\s?]\b", test)

#%%


stands.to_csv('data/producenten_scraped/milieuwijzerutrecht_punt_nl.csv')
    
