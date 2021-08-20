#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  3 11:31:19 2021

@author: lisatostrams
"""




from owslib.wms import WebMapService
from osgeo import gdal, osr
# import processing


url = 'https://service.pdok.nl/cbs/pd/wms/v1_0'
folder = 'data/data_overheid_wms/populatie'
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


for layerno in range(0,1):
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
        Xpix=4000
        Ypix=4000
        trycount = 2
        while trycount>0:
            try:
                img = wms.getmap( layers=[wms_layers[layerno]], styles=['default'], srs=srs_string, bbox=bbox, size=(Xpix, Ypix), format='image/png',timeout=60 )
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
import matplotlib.pyplot as plt
import numpy as np

legend = plt.imread(folder+'/v1_0.png')

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
   



# from sklearn.cluster import KMeans
xs,ys = np.where(mask_colors(legend) == True)

# legend_c = legend.copy()
# legend_c[xs,ys]=1
# plt.imshow(legend_c)


legend_rs = (legend[xs,ys]*255).astype('uint8')

clf = KMeans(n_clusters = 18)
labels = clf.fit_predict(legend_rs)

# band_rs = (band.reshape(4000**2,3))
# band_labels = clf.predict(band_rs)


#%%
from collections import Counter

counts = Counter(labels)

center_colors = clf.cluster_centers_
center_colors = np.clip(center_colors,0,255)
# We get ordered colors by iterating through the keys
ordered_colors = [center_colors[i] for i in counts.keys()]
hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts.keys()]
rgb_colors = [ordered_colors[i] for i in counts.keys()]


def mask_color(img,rgb):
    r,g,b=rgb
    diffr = abs(img[:,:,0] - r) < 0.05
    diffg = abs(img[:,:,1] - g) < 0.05
    diffb = abs(img[:,:,2] - b) < 0.01
    return np.logical_and(diffr,np.logical_and(diffg,diffb))
legend_colors = []
for i in range(0,len(rgb_colors)):
    mask = mask_color(legend,rgb_colors[i]/255)
    xs,ys = np.where(mask == False)
    if mask.sum()>104:
        legend_colors.append(rgb_colors[i])
        legend_c = legend.copy()
        legend_c[xs,ys]=1
        plt.imshow(legend_c)
        plt.show()
#%%
# legend_colors.append(np.array([255,255,255],dtype='float32'))
# legend_sort = sorted(legend_colors,key=lambda l: l[0]+l[1],reverse=True)
masks = []
for i in range(0,len(legend_sort)):
    mask = mask_color_int(band,legend_sort[i])
    masks.append(mask)
    xs,ys = np.where(mask == False)
    band_c = band.copy()
    band_c[xs,ys]=200
    plt.imshow(band_c)
    plt.show()
    
#%%


kmeans = KMeans(n_clusters=7, init=np.array(legend_sort), max_iter=1,n_init=1) # just run one k-Means iteration so that the centroids are not updated
band_rs = (band.reshape(4000**2,3))
band_labels = kmeans.fit_predict(band_rs)
kmeans.labels_

# band_labels = clf.predict(band_rs)
#%%


cluster_mask = band_labels.reshape(4000,4000)
band_c = band.copy()
for i in range(0,6):
    xs,ys = np.where(cluster_mask==i)
    
    band_c[xs,ys]=legend_sort[i]
    plt.imshow(band_c)
    plt.show()

#%%
m=np.zeros_like(mask)*0
for i in range(0,len(masks)):
    m+= 1*masks[i]
    
m = (m>=1)*1
plt.imshow(m,cmap='gray')
#%%
import geopandas as gpd
import rasterio
from rasterio import features
from rasterio import mask
import numpy as np

def combine_masks(list_masks):
    mask0 = list_masks[0]
    
    for mask in list_masks:
        mask0 = np.logical_or(mask,mask0)
    return mask0

def mask_color_int(img,rgb):
    r,g,b=rgb
    diffr = abs(img[:,:,0] - r) < 5
    diffg = abs(img[:,:,1] - g) < 3
    diffb = abs(img[:,:,2] - b) < 3
    return np.logical_and(diffr,np.logical_and(diffg,diffb))
    
results = []

src=rasterio.open(georeferenced_tiff) 
with rasterio.open(georeferenced_tiff) as src:
    src_meta = src.meta
    src_affine = src_meta.get("transform")

    band = src.read()
    band=np.moveaxis(band, 0, -1)
    masks = []
    colors = []
    for color in legend_sort:
        mask=mask_color_int(band,(color*255).astype('uint8'))
        masks.append(mask)
        # band_c = band.copy()
        xs,ys = np.where(mask == True)
        band[xs,ys]= (color*255).astype('uint8')
        colors.append((color*255).astype('uint8'))


    mask_c = combine_masks(masks)
    xs,ys=np.where(mask_c==False)
    band[xs,ys]=0
        
    
    
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



#%%

gpd_results.to_csv(folder+'/population_shapes_gpd.csv')
gpd_results.to_file(folder+"/populatie.geojson", driver='GeoJSON')
#%%


f=open(folder+'/colors_to_gray.txt','w')
for ele in colors:
    gray = ((ele[0]*0.2989) + (ele[1]* 0.5870) +(ele[2]* 0.1140)).astype('uint8') 
    print(gray)
    f.write(str(ele)+', '+str(gray)+'\n')

f.close()



#%%

f=open('feedback.txt','w')
for i in range(1,29):
    f.write('{}:\n'.format(i))

f.close()




  