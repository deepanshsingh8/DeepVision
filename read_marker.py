# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 18:10:29 2020

@author: Deepansh
"""

import pandas as pd
import collections


markers = pd.read_csv("markers.csv")
markers.drop(markers.columns[[0]],axis=1,inplace=True)
a = collections.defaultdict(list)
b = collections.defaultdict(list)

for i,row in markers.iterrows():
    for j, val in enumerate(row):
        a[val].append([i,j])
    #break
        
        #a[markers.iloc[i,j]].append([i,j])

for key in a:
    if key == 0:
        continue
    else:
        if len(a[key])%2 == 0:
            b[key] = a[key][int(len(a[key])/2)-1]
        else:
            b[key] = a[key][int((len(a[key])+1)/2)-1]
    
centrelist = list()
for key,value in b.items():
    centrelist.append(value)    