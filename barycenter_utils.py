
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import sys
import glob
import cv2


#sys.path.append('drive/MyDrive/projected_sinkhorn/')
 
 
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
import argparse
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from matplotlib import pyplot as plt

#print(device)


import cv2

def wasserstein_cost( X , p=2, kernel_size= 784 , n = 28 ):
    C = X.new_zeros(kernel_size,kernel_size)
    for i in range(kernel_size):
      for j in range(kernel_size):
        tix = i/n
        tiy = i%n
        tjx = j/n
        tjy = j%n
        C[i,j] = ( abs(tix - tjx)**p + abs(tiy - tjy)**p)**(1/p)

    
    return C
  
def old_wasserstein_cost(X, p=2, kernel_size=5):
    if kernel_size % 2 != 1: 
        raise ValueError("Need odd kernel size")
        
    center = kernel_size // 2
    C = X.new_zeros(kernel_size,kernel_size)
    for i in range(kernel_size): 
        for j in range(kernel_size): 
            C[i,j] = (abs(i-center)**2 + abs(j-center)**2)**(p/2)
    return C


def any_nan(X): 
    return (X != X).any().item()

def print_cifar( pixels ):
  if pixels.shape[0] == 1:
    plt.axis('off')
    plt.imshow( pixels[0] , cmap = 'gray' )
    plt.show()
    return
  #print('Pixel shape' , pixels.shape)
  pixels = (pixels.transpose(1,2,0)) #.astype(np.uint8)
  
  fig = plt.figure(figsize=(4,4)) 
  ax = fig.add_subplot(141) 
  ax.imshow(pixels,interpolation='nearest')
  plt.axis('off')
  plt.show()
  


def print_temp( pixels  ):
  pixels = torch.clamp(pixels , min = 0.0 , max = 1.0)
  pixels = pixels[0].cpu().detach().numpy()
  print_cifar(pixels)



def rotate_image(image, angle):
  ishape = image.shape
  image = ( image.transpose(1,2,0) )*255.
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags = cv2.INTER_LINEAR) #, flags=cv2.INTER_LINEAR
  if ishape[0] == 1:
    return (result/255.).reshape(ishape)
  return (result/255.).transpose(2,0,1)

def expandd(X, shape): 
    return X.view(*X.size()[:-1], *shape)

def unflatten2(X):
    n = X.size(-1)
    k = int(math.sqrt(n))
    return expandd(X,(k,k))

def unsqueeze3(X):
    return X.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

def expand_filter(X, nfilters , index ): 
    sizes = list(-1 for _ in range(X.dim()))
    sizes[-index] = nfilters
    return X.expand(*sizes)


def unfoldd(x, kernel_size, padding=None): 
    size = x.size()
    if len(size) > 4: 
        x = x.contiguous().view(-1, *size[-3:])
    out = F.unfold(x, kernel_size, padding=kernel_size//2)
    if len(size) > 4: 
        out = out.view(*size[:-3], *out.size()[1:])
    return out

def collapse2(X): 
    return X.view(*X.size()[:-2], -1)


def translate(image , pix = 1 ):
  ishape = image.shape
  image = ( image.transpose(1,2,0)*255 )
  M = np.float64([[1, 0, pix], [0, 1, pix]])
  shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
  if ishape[0] > 1:
    return (shifted/255.).transpose(2,0,1)
  return shifted.reshape(ishape)/255.
