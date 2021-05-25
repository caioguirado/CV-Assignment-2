import pandas as pd
import torch
import numpy as np
from PIL import Image
import time
import cv2
from matplotlib import pyplot as plt

'''
Script to generate both kinds of data
'''

def handleFaceCrop(img):
  rects = inferenceGetFaceCrop(img)
  # print(type(rects))
  
  if len(rects) == 0:
    data = cv2.resize(img, (48, 48))
    datanp = data.flatten()
    return datanp
        
  l = rects[0].left()
  r = rects[0].right()
  t = rects[0].top()
  b = rects[0].bottom()

  # plt.imshow(img)
  # plt.show()
  # print(l, r,t,b)
  if (l>=0 and r>=0 and t>=0 and b>=0):
    img_resize = img[t:b, l:r]
    if (img_resize.shape[0]>0 and img_resize.shape[1]>0):
      data = cv2.resize(img_resize, (48, 48))
      datanp = data.flatten()
      
    else:
      datanp = img.flatten()  
  else:
    datanp = img.flatten()
  # print(img.shape)
  
  # plt.imshow(img)
  # plt.show()

  # data = np.asarray(img.reshape(48, 48, 3), dtype="int32")
  # print(data)
  # print(data.shape)
  return datanp

def handleMouthCrop(img):
  marks = inferenceGetMouth(img)

  # print(marks)
  # print((marks[0][0]))
  if len(marks) == 0:
    data = cv2.resize(img, (48, 48))
    datanp = data.flatten()
    return datanp


  x = marks[4][0]
  y = marks[3][1]
  h = marks[6][1] - y
  w = marks[12][0] - x

  # plt.imshow(img)
  # plt.show()

  # print(x,y,h,w)
  # print(img.shape)
  if (x>=0 and y>=0 and h>=0 and w>=0):
    img_resize = img[y:y+h, x:x+w]
  # print(img_resize.shape)
    if (img_resize.shape[0] > 0 and img_resize.shape[1]>0):
      data = cv2.resize(img_resize, (48, 48))
      datanp = data.flatten()
    else:
      datanp = img.flatten()
  else:
    datanp = img.flatten()
  return datanp

  # plt.imshow(data)
  # plt.show()

def handleEyeCrop(img):
  marks = inferenceGetMouth(img)

  print(marks)

  print((marks[0][0]))
  x = marks[0][0]
  y = marks[19][1]
  h = marks[29][1] - y
  w = marks[16][0] - x

  # plt.imshow(img)
  # plt.show()

  # print(l,r,t,b)
  img = img[y:y+h, x:x+w]
  data = cv2.resize(img, (48, 48))
  datanp = data.flatten()
  return datanp
  # plt.imshow(data)
  # plt.show()  



df = pd.read_csv('/content/gdrive/MyDrive/cv/Facial-Landmarks-Detection-Pytorch/fer2013.csv')
pixels = df['pixels']

pixels_face = []
pixels_mouth = []
pixels_eyes = []

face_str = ''
for index, row in tqdm(df.iterrows()):
  print(index)
  pix = row['pixels']
  X = np.array(pix.split()).reshape(48, 48, 1).astype('int32')
  cv2.imwrite('X.png', X)
  # time.sleep(1)
  # print(X.shape)
  image = cv2.imread('/content/gdrive/MyDrive/cv/X.png')
  # print('xxxx '+str(image.shape))
  face_data = handleFaceCrop(image)
  mouth_data = handleMouthCrop(image)
  pixels_face.append(face_data)
  pixels_mouth.append(mouth_data)
  # break

df['pixels_face'] = pixels_face
df['pixels_mouth'] = pixels_mouth