import torch
import numpy as np

'''
Data Loader class
'''
class Dataset(torch.utils.data.Dataset):

  def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

  def __len__(self):
        return len(self.data)

  def __getitem__(self, index):

        # Running experiment with simple vanilla data
        pixels_values_flat = self.data['pixels'].iloc[index]
        X = np.array(pixels_values_flat.split()).reshape(48, 48, 1).astype('float32')
        
        # Running experiment with cropped face data
        pixels_face_values_flat = self.data['pixels_face']
        X = np.array(pixels_face_values_flat).reshape(48, 48, 3).astype('int32')

        # Running experiment with mouth data
        pixels_face_values_flat = self.data['pixels_face']
        X = np.array(pixels_face_values_flat).reshape(48, 48, 3).astype('int32')

        y = self.data['emotion'].iloc[index].item()
      #   y = torch.eye(7, dtype=torch.int8)[y]

        if self.transform:
            X = self.transform(X)

        if X.shape[0] < 3:
          X = torch.cat([X, X, X], axis=0) # Simulating 3 channels

        return X, y