import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms.functional as torchFunc
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision import transforms
from torchsummary import summary

'''
Class for Face Aug
'''

class AugFace:
  def __init__(self, dim, bright, contrast, saturation, hue, face_offset, crop_offset):
    self.face_offset = face_offset
    self.dim = dim
    self.crop_offset = crop_offset
    self.transform = transforms.ColorJitter(bright, contrast, saturation, hue)

  def img_crop_offset(self, img, landmarks, crop_coord):
    l = int(crop_coord['left']) - self.face_offset
    t = int(crop_coord['top']) - self.face_offset
    w = int(crop_coord['width']) + (2 * self.face_offset)
    h = int(crop_coord['height']) + (2*self.face_offset)

    img = torchFunc.crop(img, t, l, h,w)
    landmarks = landmarks - np.array([[l, t]])

    new_dimen = self.dim + self.crop_offset

    img = torchFunc.resize(img, (new_dimen, new_dimen))
    
    landmarks[:, 0] *= new_dimen/w
    landmarks[:, 1] *= new_dimen/h

    return img, landmarks

  def img_random_face_crop(self, img, landmarks):
    img = np.array(img)
    h, w = img.shape[:2]
    t = np.random.randint(0, h-self.dim)
    l = np.random.randint(0, w - self.dim)

    img = img[t: t + self.dim, l: l+self.dim]
    landmarks = landmarks - np.array([[l, t]])

    return torchFunc.to_pil_image(img), landmarks

  def __call__(self, img, landmarks, crop_coord):
    img, landmarks = self.img_crop_offset(img, landmarks, crop_coord)
    img, landmarks = self.img_random_face_crop(img, landmarks)
    return img, landmarks