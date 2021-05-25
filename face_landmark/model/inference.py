from XceptionNet import *
import dlib
from imutils import resize, face_utils
import cv2

'''
Perform inference on the trained Xception net model to find facial landmarks
'''


# xModel = XceptionNetModule()
# xModel.cuda()
# xModel.load_state_dict(torch.load('/Users/dhruvrathi/narnia/masters/period5/CV/ass2/CV-Assignment-2/face_landmark/face_landmark_model.pt'))
face_detector = dlib.get_frontal_face_detector()
landmark_detector = dlib.shape_predictor("/Users/dhruvrathi/narnia/masters/period5/CV/ass2/CV-Assignment-2/face_landmark/shape_predictor_68_face_landmarks.dat")

def inference(path):
  img = dlib.load_rgb_image(path)
  gray_out = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
  faces = face_detector(img, 1)
  marks = []

  for k, d in enumerate(faces):
    points_landmark = landmark_detector(img, d)
    for n in range(0, 67):
        x = points_landmark.part(n).x
        y = points_landmark.part(n).y
        marks.append((x, y))
        cv2.circle(img, (x, y), 5, (255, 255, 0), -1)

  plt.imshow(img)
  plt.show()
  return marks

path = "/Users/dhruvrathi/narnia/masters/period5/CV/ass2/CV-Assignment-2/face_landmark/monica_bellucci.png"

inference(path)

"""# INFERENCE"""

import dlib
from imutils import resize, face_utils
import cv2

# preprocessing before inference
def prep_img(img):
  img = torchFunc.to_pil_image(img)
  img = torchFunc.resize(img, (128, 128))
  img = torchFunc.to_tensor(img)
  img = (img - img.min())/(img.max()-img.min())
  img = (2*img)-1
  return img.unsqueeze(0)

def draw_facial_landmarks(img, landmarks):
  img = cv2.resize(img, (128, 128)) 

  img = img.copy()
  for landmarks, (l,t,h,w) in landmarks:
    landmarks = landmarks.view(-1,2)
    landmarks = landmarks+0.5
    landmarks = landmarks.numpy()

    for i, (x,y) in enumerate(landmarks, 1):
      try:
        # cv2.circle(img, (int((x * w) + l), int((y * h) + t)), 5, (255, 255, 0), -1)
        # cv2.circle(img, x,y , 5, (255, 255, 0), -1)
        
        cv2.circle(img, (int((x * w) + l), int((y * h) + t)), 5, [40, 117, 255], -1)
      except:
        pass
  return img

detector = dlib.get_frontal_face_detector()

@torch.no_grad()
def inference_mo(img):
  gray_out = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
  faces_detected = detector(gray_out, 1)

  face_landmarks = []

  for(ind, face) in enumerate(faces_detected):
    (x,y,w,h) = face_utils.rect_to_bb(face)
    crop_img = gray_out[y:y+h, x:x+w]
    final_img = prep_img(crop_img)
    landmarks = xModel(final_img.cuda())
    face_landmarks.append((landmarks.cpu(), (x,y,h,w)))
  
  return face_landmarks

def read_inference_img(path):
  img = io.imread(path)
  return img
