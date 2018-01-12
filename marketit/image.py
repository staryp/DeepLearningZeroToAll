from darkflow.net.build import TFNet
import cv2
from matplotlib import pyplot as plt
from PIL import Image
import pandas as pd
import numpy as np
import requests
import os
import shutil
import urllib
import re
import cuter
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import numpy as np
import copy

os.getcwd()
os.chdir("/home/ubuntu/darkflow")



options = {"model": "./cfg/yolo.cfg", "load": "./bin/yolo.weights", "threshold": 0.6, "gpu": 1.0}
tfnet = TFNet(options)

rootDir = '/home/ubuntu/jupyter-home/appearance_ml/input'
targetDir = '/home/ubuntu/jupyter-home/gang/apperance/input'
        

limit = 50000 # 최대 검사할 이미지 숫자

for dirName, subdirList, fileList in os.walk(rootDir):
    
    uid = dirName.split("/")[-1]
    idx = 0
    for fname in fileList:
        if re.search('\.jpg$', fname) :
            
            # load image
            imgPath = '%s/%s' % (dirName,fname)
            img = cv2.imread(imgPath)
            
            # find person
            result = tfnet.return_predict(img)
            person_found_list = list(filter(lambda x: x["label"]=="person", result))
            
            # copy image
            if len(person_found_list) > 0:
                idx = idx + 1
                destImgPath = "%s/%s_%d.%s" % (targetDir, uid, idx, fname.split('.')[-1])
                # print(destImgPath)
                shutil.copyfile(imgPath, destImgPath)
        limit = limit-1    
        
        if limit % 5000 == 0 :
            print("remain %d " % limit)
        
    if limit <= 0:
        break

size = 300,300
idx = 0
for dirName, subdirList, fileList in os.walk(targetDir):
    
    uid = dirName.split("/")[-1]
    
    for fname in fileList:

        if re.search('\.jpg$', fname) :

            imgPath = '%s/%s' % (dirName,fname)
            img = Image.open(imgPath)
            resize_and_crop(imgPath,imgPath,size)
            idx = idx+1


       
xdata = np.array([])
for dirName, subdirList, fileList in os.walk(rootDir):
    
    uid = dirName.split("/")[-1]
    idx = 0
    for fname in fileList:
        if re.search('\.jpg$', fname) :
            
            # load image
            imgPath = '%s/%s' % (dirName,fname)
            img = cv2.imread(imgPath)

            for y in img:
                for x in y:
                    xdata.append(x)

        break


x_data = np.array([])
idx = 0
targetDir = '/home/ubuntu/jupyter-home/gang/apperance/input'
limit = 10000

for dirName, subdirList, fileList in os.walk(targetDir):
    
    uid = dirName.split("/")[-1]
    for fname in fileList:
        x_newrow = np.array([])
        if re.search('\.jpg$', fname) :
            
            # load image
            imgPath = '%s/%s' % (dirName,fname)
            img = cv2.imread(imgPath)
            
            for y in img:
                for x in y:
                    x_newrow=np.append(x_newrow,x)
        
        if idx == 0 :
            x_data = np.append(x_data, x_newrow)
            idx=1
        else:
            x_data = np.vstack([x_data, x_newrow])

        limit = limit-1
        if limit % 100 == 0 :
            print("remain %d " % limit)
           
np.savetxt("x_data.csv", x_data, delimiter=",")


data = np.array([])
idx = 0
train = pd.read_csv("/home/ubuntu/jupyter-home/gang/apperance/apperance_score.csv")

targetDir = '/home/ubuntu/jupyter-home/gang/apperance/input'
for dirName, subdirList, fileList in os.walk(targetDir):
    
    uid = dirName.split("/")[-1]
    for fname in fileList:
        newrow = np.array([])
        if re.search('\.jpg$', fname) :
            x,b =fname.split('.')
            id,_=x.split('_')
            d =  train.loc[train['id'] == float(id)]
            newrow = d['value'].values
            if newrow[0].count('하') > 0:
                newrow = np.array([[1,0,0,0,0]])
            elif newrow[0].count('중') > 0:
                newrow = np.array([[0,1,0,0,0]])
            elif newrow[0].count('중상') > 0:
                newrow = np.array([[0,0,1,0,0]])
            elif newrow[0].count('상') > 0:    
                newrow = np.array([[0,0,0,1,0]])
            elif newrow[0].count('최상') > 0:
                newrow = np.array([[0,0,0,0,1]])
            else:
                newrow = np.array([[0,0,1,0,0]])
        if idx == 0 :
            data = np.append(data, newrow)
            idx=1
        else:
            
            data = np.vstack([data, newrow])


    image_batch = tf.train.batch([image_list, label_list], batch_size=FLAGS.batch_size)