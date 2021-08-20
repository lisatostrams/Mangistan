#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  4 14:35:18 2021

@author: lisatostrams
"""
from owslib.wms import WebMapService
from osgeo import gdal, osr

url = 'https://geodata.nationaalgeoregister.nl/cbspostcode6/wms'
folder = 'data/data_overheid_wms/postcode_stats'
wms = WebMapService(url, version='1.1.1')
wms_layers= list(wms.contents)
print("Title: ", wms.identification.title)
print("Type: ", wms.identification.type)
print("Operations: ", [op.name for op in wms.operations])
print("GetMap options: ", wms.getOperationByName('GetMap').formatOptions)

#%%


for i,layer in enumerate(wms_layers):
     print(i, wms[layer].title)
    
#%%
with open(folder+'/layer_number_names.txt', 'w+') as f:
    for i,item in enumerate(wms_layers):
        f.write("{}: {}   {}\n".format(i,wms[item].title,item))
        
#%%

import os
import time
bbox = (4.7920404, 51.8573607, 5.6273145, 52.3036184)


for layerno in range(5,len(wms_layers)):
    styles = wms[wms_layers[layerno]].styles
    for style in styles.keys():
        title = wms[wms_layers[layerno]].title
        print(title)
        title = title+'_'+style
        raw_tiff=os.path.join(folder,'_raw'+str(title)+'.tif')
        georeferenced_tiff=os.path.join(folder,'_georeferenced'+str(title)+'.tif')
        if os.path.isfile(georeferenced_tiff):
            print('already exists')
        else:
            defaultepsg=4326 
            srs_string='EPSG:'+str(defaultepsg)
            lonmin=bbox[0]
            lonmax=bbox[2]
            latmin=bbox[1]
            latmax=bbox[3]
            Xpix=4000
            Ypix=4000
            trycount = 2
            while trycount>0:
                try:
                    img = wms.getmap( layers=[wms_layers[layerno]], styles=[style], srs=srs_string, bbox=bbox, size=(Xpix, Ypix), format='image/png',timeout=60 )
                    trycount=0
                    print('success')
                except:
                    if trycount ==0:
                        print('Failed to retrieve '+title)
                    else:
                        print('fail')
                        trycount -= 1
                        # time.sleep(0.5)
            out = open(raw_tiff, 'wb')
            out.write(img.read())
            out.close()
            out = None
            
            src_ds = gdal.Open(raw_tiff)
            format = "GTiff"
            driver = gdal.GetDriverByName(format)
            dst_ds = driver.CreateCopy(georeferenced_tiff, src_ds, 0)
            gt = [lonmin, (lonmax-lonmin)/Xpix, 0, latmax, 0, -(latmax-latmin)/Ypix]
            dst_ds.SetGeoTransform(gt)
            srs = osr.SpatialReference()
            srs.ImportFromEPSG(defaultepsg)
            dest_wkt = srs.ExportToWkt()
            dst_ds.SetProjection(dest_wkt)
            dst_ds = None
            src_ds = None
            
            print('done')
            
            
#%%

import requests
from bs4 import BeautifulSoup


for style in styles.keys():

    url = styles[style]['legend']
    
    page = requests.get(url)
    
    file = open(folder+'/'+styles[style]['title']+"_legend.png", "wb")
    file.write(page.content)
    file.close()

#%%



from sklearn.cluster import KMeans

legend_rs = band.reshape(4000**2,3)

clf = KMeans(n_clusters = 3)
labels = clf.fit_predict(legend_rs)


from collections import Counter

counts = Counter(labels)

center_colors = clf.cluster_centers_
center_colors = np.clip(center_colors,0,255)
# We get ordered colors by iterating through the keys
ordered_colors = [center_colors[i] for i in counts.keys()]

#%%


def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(255*color[0]), int(255*color[1]), int(255%color[2]))


hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts.keys()]
rgb_colors = [ordered_colors[i] for i in counts.keys()]
plt.figure(figsize = (8, 6))
plt.pie(counts.values(), labels = hex_colors, colors = rgb_colors)

#%%

from skimage.morphology import closing
from skimage.morphology import disk

def mask_color_int(img,rgb):
    r,g,b=rgb
    diffr = abs(img[:,:,0] - r) < 30
    diffg = abs(img[:,:,1] - g) < 30
    diffb = abs(img[:,:,2] - b) < 30
    return np.logical_and(diffr,np.logical_and(diffg,diffb))
    

mask = labels==1
mask = mask.reshape(4000,4000)
mask = closing(mask,disk(5))
xs,ys = np.where(mask==False)
band_c = band.copy()
band_c[xs,ys]= 255
plt.imshow(band_c)

plt.imsave('test.png',band_c)
#%%

import geopandas as gpd
import rasterio
from rasterio import features
from rasterio import mask
import numpy as np

style = 'cbspostcode6:cbs_pc6_aantal_inwoners'
title = wms[wms_layers[layerno]].title
title = title+'_'+style
georeferenced_tiff=os.path.join(folder,'_georeferenced'+str(title)+'.tif')
results=[]
with rasterio.open(georeferenced_tiff) as src:
    src_meta = src.meta
    src_affine = src_meta.get("transform")

    band = src.read()
    band=np.moveaxis(band, 0, -1)


    
    gray = np.dot(band, [0.2989, 0.5870, 0.1140]).astype('uint8')

    for geometry, raster_value in features.shapes(gray, transform=src_affine):
        if raster_value > 0:
            result = {'properties': {'raster_value': raster_value}, 'geometry': geometry}
            results.append(result)

gpd_results = gpd.GeoDataFrame.from_features(results,crs=src.crs)

gpd_results["area"] = gpd_results["geometry"].to_crs({'proj':'cea'}).area

#%%



values = gpd_results.raster_value.unique()
for v in values:
    gpd_v = gpd_results[gpd_results.raster_value==v]
    gpd_v.plot()
    plt.show()

