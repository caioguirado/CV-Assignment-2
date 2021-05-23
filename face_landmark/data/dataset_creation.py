import os
import numpy as np
from xml.etree import ElementTree

class DatasetLandmarks(Dataset):
  def __init__(self, augProcessor, train):
    self.root_dir = 'ibug_300W_large_face_landmark_dataset'
    self.img_paths = []
    self.landmarks = []
    self.crops = []
    self.augProcessor = augProcessor
    self.train = train

    elementTree = ElementTree.parse(os.path.join(self.root_dir, f'labels_ibug_300W_{"train" if train else "test"}.xml'))
    root = elementTree.getroot()

    for name in root[2]:
      self.img_paths.append(os.path.join(self.root_dir, name.attrib['file']))
      self.crops.append(name[0].attrib)
      landmark = []
      for mark_num in range(68):
        x = int(name[0][mark_num].attrib['x'])
        y = int(name[0][mark_num].attrib['y'])
        landmark.append([x, y])
      self.landmarks.append(landmark)

    self.landmarks = np.array(self.landmarks).astype('float32')
    assert len(self.img_paths) == len(self.landmarks)
    
  def __len__(self):
    return len(self.img_paths)
    
  def __getitem__(self, index):
    img = io.imread(self.img_paths[index], as_gray=False)
    landmarks = self.landmarks[index]

    img, landmarks = self.augProcessor(img, landmarks, self.crops[index])
    return img, landmarks
