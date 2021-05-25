import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms.functional as torchFunc
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision import transforms
from torchsummary import summary
from face_aug import *
from landmarks_aug import *
'''
Class for Data Augmentation - Combining face aug and landmarks aug
'''

class DataAugPreprocessor:
  def __init__(self, dim, bright, contrast, saturation, hue, angle, face_offset, crop_offset):
    self.dim = dim
    self.landmarks_aug = AugLandmarks(dim, angle)
    self.face_aug = AugFace(dim, bright, contrast, saturation, hue, face_offset, crop_offset)
  
  def __call__(self, img, landmarks, crop_coord):
    img = torchFunc.to_pil_image(img)
    img, landmarks = self.face_aug(img, landmarks, crop_coord)
    landmarks = landmarks / np.array([*img.size])
    img, landmarks = self.landmarks_aug(img, landmarks)
    img = torchFunc.to_grayscale(img)
    img = torchFunc.to_tensor(img)
    img = (img - img.min())/(img.max()-img.min())
    img = (2*img)-1

    return img, torch.FloatTensor(landmarks.reshape(-1)-0.5)