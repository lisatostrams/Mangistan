#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  3 11:20:17 2021

@author: lisatostrams
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 15:24:36 2021

@author: lisatostrams
"""


from owslib.wfs import WebFeatureService
from osgeo import gdal, osr
# import processing


url = 'https://service.pdok.nl/cbs/pd/wfs/v1_0'
folder = 'data/data_overheid_wfs/populatie'
wms = WebFeatureService(url, version='1.1.0')
wms_layers= list(wms.contents)
print("Title: ", wms.identification.title)
print("Type: ", wms.identification.type)
print("Operations: ", [op.name for op in wms.operations])
print("Operation options: ", wms.getOperationByName('DescribeFeatureType'))


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
    if not os.path.isfile(georeferenced_tiff):
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