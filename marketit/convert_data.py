import cv2
import pandas as pd
import numpy as np
import requests
import os
import shutil
import urllib
import re


idx = 0
rootDir = '/home/ubuntu/jupyter-home/gang/apperance/input'
targetDir = '/home/ubuntu/jupyter-home/gang/apperance/input2'
label_csv = pd.read_csv("/home/ubuntu/jupyter-home/gang/apperance/apperance_score.csv")
limit = 10000

for dirName, subdirList, fileList in os.walk(rootDir):
    
    uid = dirName.split("/")[-1]
    for fname in fileList:
        x_newrow = np.array([])
        if re.search('\.jpg$', fname) :
            
            # load image
            imgPath = '%s/%s' % (dirName,fname)
            x,b =fname.split('.')
            id,_=x.split('_')
            d =  label_csv.loc[label_csv['id'] == float(id)]
            newrow = d['value'].values

            set_dir = ''
            if not newrow:
                set_dir='B'
            elif newrow[0].count('하') > 0:
                set_dir='D'
            elif newrow[0].count('중') > 0:
                set_dir='C'
            elif newrow[0].count('중상') > 0:
                set_dir='B'
            elif newrow[0].count('상') > 0:    
                set_dir='A'
            elif newrow[0].count('최상') > 0:
                set_dir='S'
            else:
                set_dir='B'
            
            destImgPath = "%s/%s/%s.%s" % (targetDir, set_dir, x,b)
            # print(destImgPath)
            shutil.copyfile(imgPath, destImgPath)
        
        limit = limit-1
    
        if limit % 100 == 0 :
            print("remain %d " % limit)
           
