
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import sys
import glob
import cv2

import sys

 
 
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
import argparse
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from matplotlib import pyplot as plt

sys.path.append('drive/MyDrive/projected_sinkhorn/')
os.system('python barycenter_utils.py')

#import barycenter_utils

from barycenter_utils import *





# _mm() perfoms matrix multiplication between K and alpha convolutionally.

def _mm(A,x, shape): 
    kernel_size = A.size(-1) 
    nfilters = shape[-3]  
    unfolded = unfoldd(x, kernel_size, padding=kernel_size//2).transpose(-1,-2) 
    unfolded = expandd(unfolded, (A.size(-3),A.size(-2)*A.size(-1))).transpose(-2,-3) 
    out = torch.matmul(unfolded, collapse2(A.contiguous()).unsqueeze(-1)).squeeze(-1) 
    return unflatten2(out)  


#function to calculate ∇H∗qk(uk) 
def del_H( q , alpha , K  ):
  q1 = q/( _mm(K , alpha , alpha.size()) + 1e-15 )
  q1 = alpha* _mm(K , q1 , q1.size())
  return q1 


def alph( u , gamma):
  return torch.exp(u/gamma )



def vectorized_barycentre_attack( marginals  ,  beta  , C , gamma  , theeta , T , thold = 2000  , vbose = 5000 , pbose = False , nrml = None ):
  shape = marginals.size()
  N = shape[0]
  B = shape[1]
  
  n = marginals.size(3)

  with torch.no_grad():
    X = marginals.new_ones(shape)        # X -> ( u(k's) , v)  initialized with ones #( N , B , 3 , 32 , 32 )

    un = marginals.new_ones( shape[1:])

    K = expand_filter( torch.exp(-C/gamma)*(unsqueeze3(X.new_ones(shape[0])).unsqueeze(-1)) , shape[-3] , 3 )
    K = expand_filter( K , shape[-4] , 4 )   # K = e-C/gamma . C was of size (7*7), Expanding K from (7*7) to (N*B*C*7*7) so that _mm() can perform matrix multiplication with qk.
    

    un = -X[N-1]/beta[N-1] - torch.sum( ((X*beta.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))/beta[N-1])[:-1] ,  0 )  # uN as per definition
    del_h_u = del_H( marginals , alph( torch.cat( (X[:-1] , torch.unsqueeze(un,0) ) , 0 )  ,gamma) , K  ) # Calculating  ∇H∗qk(uk) for all k from 1 to n at once
    
    p = torch.sum( del_h_u , 0  )/N  #calculating p
    co = 0
    
    while co < thold :
      del_h_un = del_h_u[N-1]
      del_F_x =  ((del_h_u - del_h_un.unsqueeze(0) )*beta.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)) # calculating ∇F(x)
      del_F_x[N-1] = -del_h_un 
      X = (X - T*del_F_x  )
      X = X - ( torch.matmul( torch.matmul( X.reshape(N,-1).t() , beta.unsqueeze(-1)) , beta.unsqueeze(-1).t() )/torch.sum(beta*beta) ).t().reshape(shape)
      X[N-1] = X[N-1]/(1 - T/theeta)  # Updating X
      
      un = -X[N-1]/beta[N-1] - torch.sum( ((X*beta.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))/beta[N-1])[:-1] ,  0 )
      del_h_u = del_H( marginals , alph( torch.cat( (X[:-1] , torch.unsqueeze(un,0) ) , 0 )  ,gamma)  , K  )
      
      
      p = torch.sum( del_h_u , 0  )/N 

      if pbose:
        if co%vbose == 0:
          print('At iteration ' , co , ' |inputimage - p|∞ max value = ' , torch.max(torch.abs(p - marginals[0])).item() , 'Max un ' ,torch.max(un).item() , 'Max ux value ', torch.max(X).item() , 'Min ux value' , torch.min(X).item() , ' MAx p' , torch.max(p).item() , ' Min p ' , torch.min(p).item()   )
          
        if co%(3*vbose) == 0:
          print('printing p at iteration ' , co)
          print_temp(p*nrml)

      if co>thold:
        break 
      co+=1
  if pbose:
    print('It took ' , co , 'iterations to converge ')
  return p  #returning p



import imutils
import scipy.ndimage as nd

def attack(X ,y , net1 , beta , p = 2 , alpha=0.01, xmin=0, xmax=1 , normalize=lambda x: x, verbose=0, 
             norm='l2' ,   gamma = 0.05  , theeta = 1 , T = 0.1  , thold = 2000 , vbose = 5000 , Cf = None , pbose = False , show = False , alp = 0.7 , angle1 = 3 , angle2 = 3 , pix = 1 , N = 4 ): 
    B = X.size(0)
    c = X.size(1)
    marginals = []

    C = Cf  #Cf is the old wasserstein cost 

    
    A1 = X.clone()
    with torch.no_grad():
      Xright = A1.new_zeros(X.size())
      Xright1 = A1.new_zeros(X.size())
      Xleft1 = A1.new_zeros(X.size())
      Xleft  = A1.new_zeros(X.size())
      Xtranslate = A1.new_zeros(X.size())
      for i in range(B):
        r_rot = rotate_image(A1[i].cpu().numpy() , angle1) #Right roated LINF attack
        l_rot = rotate_image(A1[i].cpu().numpy() , -angle1 ) #Left rotated LINF attack
        r_rot1 = rotate_image(A1[i].cpu().numpy() , angle2) #Right roated LINF attack
        l_rot1 = rotate_image(A1[i].cpu().numpy() , -angle2 )
        trans_img = translate( A1[i].cpu().numpy() , pix )
        Xright[i] = torch.tensor( r_rot ).to(device).type(X.dtype)
        Xleft[i] = torch.tensor( l_rot ).to(device).type(X.dtype)
        Xright1[i] = torch.tensor( r_rot1 ).to(device).type(X.dtype)
        Xleft1[i] = torch.tensor( l_rot1 ).to(device).type(X.dtype)
        Xtranslate[i] = torch.tensor( trans_img ).to(device).type(X.dtype)

    

    with torch.no_grad():
      if show:
        print('Showing Linf attack ')
        pixels = A1[0].cpu().detach().numpy()
        print_cifar(pixels)

        print('Showing right_rot ')
        pixels = Xright[0].cpu().detach().numpy()
        print_cifar(pixels)

        print('Showing left_rot ')
        pixels = Xleft[0].cpu().detach().numpy()
        print_cifar(pixels)
  
        print('Showing translated ')
        pixels = Xtranslate[0].cpu().detach().numpy()
        print_cifar(pixels)
    
    ############################## Preparing inputs for vectorized barycenter attack, marginals - size after stacking - ( N * B * 3 * 32 * 32)

    marginals.append(A1)
    marginals.append(Xright)
    marginals.append(Xleft)
    if N > 4 :
      marginals.append(Xright1)
      marginals.append(Xleft1)

    marginals.append(Xtranslate)


    marginals = torch.stack(marginals , 0) 


    ################################

    with torch.no_grad():

      normalization = marginals.view(N,B,-1).sum(-1).view(N,B,1,1,1)  #normalization , Sum of all pixels of image 
      n1 = A1.view(B,-1).sum(-1).view(B,1,1,1)  
      
      X_ = ( vectorized_barycentre_attack( marginals.clone()/normalization  ,  beta ,  C , gamma  , theeta , T , thold  , vbose , pbose , n1 )*n1 ) # Calling vectorized barycenter attack
        
            
      X_ = torch.clamp(X_, min=xmin, max=xmax)

      if show:
        for i in range(B):
          if i%20 == 0:
            print('Showing orig image')
            pixels = X[i].cpu().detach().numpy()
            print_cifar(pixels)              
            print('Showing adversarial image')
            pixels = X_[i].cpu().detach().numpy()
            print_cifar(pixels)
            
        e1 =  (net1(normalize(marginals[0])).max(1)[1] != y)
        print("Error count of Attack 2 " , e1.sum().item() )

      err = (net1(normalize(X_)).max(1)[1] != y)
      err_best = err.sum().item()
      if show:
        print('printing final error')
        print(err_best)

    return X_.detach().clone(), y ,  A1

