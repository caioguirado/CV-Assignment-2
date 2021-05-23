import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms.functional as torchFunc
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision import transforms
from torchsummary import summary

class AugLandmarks:
  def __init__(self, dim, angle):
    self.dim = dim
    self.angle = angle
  
  def rotate_random(self, img, landmarks):
    angle = self.angle
    transform_landmarks_factor = np.array([[+np.cos(np.radians(angle)), -np.sin(np.radians(angle))],
                                    [+np.sin(np.radians(angle)), +np.cos(np.radians(angle))]])
    image = torchFunc.rotate(img, angle)
    landmarks = landmarks - 0.5
    trans_landmarks = np.matmul(landmarks, transform_landmarks_factor)
    trans_landmarks = trans_landmarks + 0.5

    return img,trans_landmarks

  def __call__(self, img, landmarks):
    img, landmarks = self.rotate_random(img, landmarks)
    return img, landmarks