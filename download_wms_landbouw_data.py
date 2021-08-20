#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 15:24:36 2021

@author: lisatostrams
"""


from owslib.wms import WebMapService
from osgeo import gdal, osr
# import processing


url = 'http://afnemers.ruimtelijkeplannen.nl/vvvp-wms/NL.IMRO.9926.PV1904PRV-GC01'
# url = 'http://afnemers.ruimtelijkeplannen.nl/vvvp-wms/NL.IMRO.9926.SV1904PRS-GC01'
folder = 'data/data_overheid_wms/landbouw'
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
with open(folder+'/layer_number_names.txt', 'w') as f:
    for i,item in enumerate(wms_layers):
        f.write("{}: {}   {}\n".format(i,wms[item].title,item))
        
     
#%%
import os
import time
bbox = (4.7924835, 51.8589517, 5.6263757, 52.3817361)

for layerno in range(0,len(wms_layers)):
    title = wms[wms_layers[layerno]].title
    print(title)
    raw_tiff=os.path.join(folder,'_raw'+str(layerno)+'.tif')
    georeferenced_tiff=os.path.join(folder,'_georeferenced'+str(layerno)+'.tif')
    if os.path.isfile(georeferenced_tiff):
        print('already exists')
    else:
        defaultepsg=4326 
        srs_string='EPSG:'+str(defaultepsg)
        lonmin=bbox[0]
        lonmax=bbox[2]
        latmin=bbox[1]
        latmax=bbox[3]
        Xpix=2000
        Ypix=2000
        trycount = 2
        while trycount>0:
            try:
                img = wms.getmap( layers=[wms_layers[layerno]], styles=['default'], srs=srs_string, bbox=bbox, size=(Xpix, Ypix), format='image/tiff',timeout=60 )
                trycount=0
            except:
                if trycount ==0:
                    print('Failed to retrieve '+title)
                else:
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
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np

tiff = os.path.join(folder,'_georeferenced'+str(layerno)+'.tif')
legend = Image.open(tiff)
legend_small = legend.resize((1000,1000), Image.ANTIALIAS)

def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(255*color[0]), int(255*color[1]), int(255%color[2]))


def get_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def mask_colors(img,redthresh=0.0,greenthresh=0.0,bluethresh=0.0):
    green = img[:,:,1]>= greenthresh
    not_red = img[:,:,0] >= redthresh
    not_blue = img[:,:,2] >= bluethresh
    diffrg = abs(img[:,:,0] - img[:,:,1]) > 0
    diffgb = abs(img[:,:,1] - img[:,:,2]) >= 0
    diffrb = abs(img[:,:,0] - img[:,:,2]) > 0
    color_less  = np.logical_and(diffrb,np.logical_and(diffrg,diffgb))
    return np.logical_and(np.logical_and(np.logical_and(green,not_red),not_blue),color_less)
   

legend_small = np.array(legend_small)
legend_rs = (legend_small.reshape(1000**2,3))


from sklearn.cluster import KMeans
clf = KMeans(n_clusters = 4)
labels = clf.fit_predict(legend_rs)

from collections import Counter

counts = Counter(labels)

center_colors = clf.cluster_centers_
center_colors = np.clip(center_colors,0,255)
# We get ordered colors by iterating through the keys
ordered_colors = [center_colors[i] for i in counts.keys()]
hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts.keys()]
rgb_colors = [ordered_colors[i]/255 for i in counts.keys()]
plt.figure(figsize = (8, 6))
plt.pie(counts.values(), labels = hex_colors, colors = rgb_colors)


#%%

f,ax=plt.subplots(figsize=(10,10))
ax.imshow(legend)

colors= [rgb_colors[i] for i in [1,4,5]]
names = ['Landbouwontwikkelingsgebied','Landbouwstabiliseringsgebied','Agrarische bedrijven']
markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in colors]

plt.legend(markers, names, loc=(1.04,0),numpoints=1)
plt.tight_layout()
plt.savefig('landbouw.png')


#%%

import geopandas as gpd
import rasterio
from rasterio import features
from rasterio import mask
import numpy as np
import os
import matplotlib.pyplot as plt
# for layerno in range(54,60):
layerno=55
tiff = os.path.join(folder,'_georeferenced'+str(layerno)+'.tif')

results=[]



with rasterio.open(tiff) as src:
    src_meta = src.meta
    src_affine = src_meta.get("transform")

    band = src.read()
    
    band=np.moveaxis(band, 0, -1)
    mask = (band==[255,255,255]).sum(axis=2)
    xs,ys = np.where(mask==3)
    band[xs,ys,:] = 0
    gray = np.dot(band, [0.2989, 0.5870, 0.1140]).astype('uint8')
    
    for geometry, raster_value in features.shapes(gray, transform=src_affine):
        if raster_value > 0:
            result = {'properties': {'raster_value': raster_value}, 'geometry': geometry}
            results.append(result)

gpd_results = gpd.GeoDataFrame.from_features(results,crs=src.crs)

gpd_results["area"] = gpd_results["geometry"].to_crs({'proj':'cea'}).area/10**6

print('{:.2f}km^2'.format(gpd_results['area'].sum()))


