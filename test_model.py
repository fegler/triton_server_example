import os 
import sys 
import cv2 
import numpy as np 
from PIL import Image 

import torch 

## model load
now_path = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(now_path, 'triton/core/1')
MODEL_PATH = os.path.join(MODEL_DIR, 'model.pt')

model = torch.jit.load(MODEL_PATH)

## customize here for using your image 
# default: zero array input 
# im = cv2.imread(IMAGE_PATH) 
im = torch.tensor(np.zeros((1,3,224,224)), dtype=torch.float32)

## inference 
try:
    model(im)
except Exception as e: 
    print('error: ', e)
finally: 
    print('test finish')