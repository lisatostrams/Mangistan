#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 15:24:36 2021

@author: lisatostrams
"""


from owslib.wms import WebMapService
from osgeo import gdal, osr
import processing


url = 'http://afnemers.ruimtelijkeplannen.nl/vvvp-wms/NL.IMRO.9926.PV1904PRV-GC01'
folder = 
wms = WebMapService(url, version='1.1.1')
wms_layers= list(wms.contents)

wms = WebMapService('http://afnemers.ruimtelijkeplannen.nl/vvvp-wms/NL.IMRO.9926.PV1904PRV-GC01')
print("Title: ", wms.identification.title)
print("Type: ", wms.identification.type)
print("Operations: ", [op.name for op in wms.operations])
print("GetMap options: ", wms.getOperationByName('GetMap').formatOptions)
wms.contents.keys()


#%%

layers = list(wms.contents)
for i,layer in enumerate(layers):
     print(i, wms[layer].title)
     
#%%

bbox = (4.7924835, 51.8589517, 5.6263757, 52.2817361)

img = wms.getmap(   layers=[layers[55]],
                 styles=['default'],
                 srs='EPSG:4326',
                 bbox=wms[layers[55]].boundingBoxWGS84,
                 size=(300,300),
                 format='image/tiff')


#%%

with open('data.tif', 'wb') as file:
    file.write(img.read())


#%%
import georasters as gr
import geopandas as gpd


raster = gr.from_file('data.tif')
geoFrame = gr.to_geopandas(raster)

