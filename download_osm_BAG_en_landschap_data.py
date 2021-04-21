# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
import time
import osmnx as ox
import matplotlib.pyplot as plt
import pandas as pd


from shapely import geometry

# postcodes = pd.read_excel('data/Consumenten_data/Postcodes Provincie Utrecht.xlsx') 
# plaats = postcodes['Plaats'].unique()
plaats = ['Abcoude', 'Nigtevecht', 'Baambrugge', 'de Hoef', 'Amstelhoek',
       'IJsselstein', 'Benschop', 'Lopik', 'Lopikerkapel', 'Jaarsveld',
       'Polsbroek', 'Montfoort', 'Oudewater', 'Snelrewaard', 'Nieuwegein',
       'Woerden', 'Vleuten', 'De Meern', 'Linschoten', 'Papekop',
       'Hekendorp', 'Kamerik', 'Zegveld', 'Harmelen', 'Utrecht',
       'Maarssen', 'Oud Zuilen', 'Tienhoven', 'Westbroek', 'Breukelen',
       'Nieuwer Ter Aa', 'Kockengen', 'Nieuwersluis',
       'Loenen aan de Vecht', 'Vreeland', 'Loenersloot', 'Mijdrecht',
       'Vinkeveen', 'Waverveen', 'Wilnis', 'Zeist', 'Austerlitz',
       'Huis ter Heide', 'Bilthoven', 'De Bilt', 'Den Dolder',
       'Bosch en Duin', 'Groenekan', 'Maartensdijk', 'Hollandsche Rading',
       'Baarn', 'Lage Vuursche', 'Bunschoten-Spakenburg', 'Eemdijk',
       'Eemnes', 'Soest', 'Soesterberg', 'Achterveld', 'Amersfoort',
       'Hoogland', 'Hooglanderveen', 'Leusden', 'Stoutenburg',
       'Stoutenburg Noord', 'Veenendaal', 'Rhenen', 'Elst Ut',
       'Renswoude', 'Woudenberg', 'Doorn', 'Cothen', 'Langbroek', 'Maarn',
       'Maarsbergen', 'Leersum', 'Amerongen', 'Overberg',
       'Wijk bij Duurstede', 'Driebergen-Rijsenburg', 'Bunnik', 'Odijk',
       'Werkhoven', 'Ossenwaard', 'Houten', "'t Goy", 'Schalkwijk',
       "Tull en 't Waal", 'Everdingen', 'Zijderveld', 'Hagestein',
       'Hoef en Haag', 'Hei- en Boeicop', 'Lexmond', 'Vianen', 'Leerdam',
       'Schoonrewoerd', 'Oosterwijk', 'Meerkerk', 'Ameide',
       'Tienhoven aan de Lek', 'Nieuwland', 'Leerbroek', 'Kedichem']
query = {'state': 'Utrecht'}

# get the boundaries of the place
utrecht_boundary = ox.geocode_to_gdf(query)
utrecht_boundary.plot()

gemeente_boundaries = {}
fig,ax = plt.subplots()
utrecht_boundary.plot(ax=ax,color='b')
for gemeente in plaats:
    query= gemeente+",Utrecht, The Netherlands"
    gdf = ox.geocode_to_gdf( query)
    # gdf.plot(ax=ax,color='r',alpha=0.8)
    gemeente_boundaries[gemeente] = gdf

    gdf = gdf['geometry'].to_crs({'proj':'cea'}) 
    print(gemeente+' area: {:.2f} square km'.format(gdf.area[0]/10**6))



# plt.show()


ox.config(timeout=10000)

cwd = os.getcwd()
caches = ['/data/cache/footprints_building/Utrecht_{}.pickle','/data/cache/footprints_landuse/Utrecht_{}.pickle','/data/cache/graph_raw/Utrecht.pickle']
# nx.write_gpickle(G,cwd+caches[2])
from shapely.ops import split
import geopandas as gpd


def split_in_half(geom,vert=True):
    ct = geom.centroid   
    xmin,ymin,xmax,ymax=geom.bounds
    ctx=ct.xy[0][0]
    cty=ct.xy[1][0]
    if vert:
        c = [(ctx,ymin),(ctx,ymax)]
    else:
        c = [(xmin,cty),(xmax,cty)]
    line = geometry.LineString(c)
    geom_collect = split(geom,line)
    return geom_collect

for gemeente in plaats:
    print(gemeente)
    
    if os.path.isfile(cwd+caches[1].format(gemeente)) or os.path.isfile(cwd+caches[1].format('{}_pt{}'.format(gemeente,0))):
        print(gemeente + ' already cached')
    else:
                                                                            
        gdf = gemeente_boundaries[gemeente]
        gdf_cea = gdf['geometry'].to_crs({'proj':'cea'}) 
        area = gdf_cea.area[0]/10**6
        if area > 25:
            small_geometries = []
            print('splitting '+gemeente)
            collect = split_in_half(gdf.geometry[0])
            g = []
            for gc in collect:
                g.append(gc)
            i=1
            while len(g)>0:
                gd = gpd.GeoDataFrame(['g[0]'],geometry=[g[0]],crs="EPSG:4326").to_crs({'proj':'cea'}) 
                area=gd.geometry[0].area/10**6
                if area > 25:
                    vert=True
                    if i%2==1:
                        vert=False
                    collect = split_in_half(g[0],vert=vert)
                    for gc in collect:
                        g.append(gc)
    
                else:
                    small_geometries.append(g[0])
                g=g[1:]
                i+=1
            print('split '+gemeente+' into '+str(len(small_geometries))+' parts')
            st = time.time()
            for i,geom in enumerate(small_geometries):
                building = ox.geometries.geometries_from_polygon(geom,{'building':['residential','house','apartments']})
                building.to_pickle(cwd+caches[0].format('{}_pt{}'.format(gemeente,i)))
            print('for {} it took {:.2f} seconds to download and cache buildings'.format(gemeente,time.time()-st))
            st = time.time()
            for i,geom in enumerate(small_geometries):
                building = ox.geometries.geometries_from_polygon(geom,{'landuse':['farmland','meadow','orchard']})
                building.to_pickle(cwd+caches[1].format('{}_pt{}'.format(gemeente,i)))
            print('for {} it took {:.2f} seconds to download and cache landuse'.format(gemeente,time.time()-st))
        else:
            st = time.time()
            building = ox.geometries.geometries_from_polygon(gemeente_boundaries[gemeente].geometry[0],{'building':['residential','house','apartments']})
            building.to_pickle(cwd+caches[0].format(gemeente))
            print('for {} it took {:.2f} seconds to download and cache buildings'.format(gemeente,time.time()-st))
            st=time.time()
            footpr = ox.geometries.geometries_from_polygon(gemeente_boundaries[gemeente].geometry[0],{'landuse':['farmland','meadow','orchard']})
            footpr.to_pickle(cwd+caches[1].format(gemeente))
            print('for {} it took {:.2f} seconds to download and cache landuse'.format(gemeente,time.time()-st))
            