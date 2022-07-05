#Implementation of Projected Gradient Descent to get Linfinity attacks

import torch
import torch.nn as nn
import torch.optim as optim
import sys
sys.path.append('drive/MyDrive/projected_sinkhorn/')


def LINF( X , y , net , alpha ,  normalize , norm ,  epsilon = 0.01 , epsilon_iters=10, epsilon_factor=1.1 , maxiters= 600 , verbose = False , xmin = 0.0 , xmax = 1.0 , ball = 'linfinity'):
    batch_size = X.size(0)
    epsilon = X.new_ones(batch_size)*epsilon
    epsilon_plan = X.new_ones(batch_size)*epsilon

    normalization = X.view(batch_size,-1).sum(-1).view(batch_size,1,1,1)
    X_ = X.clone()

    X_best = X.clone()
    err_best = err = net(normalize(X)).max(1)[1] != y
    epsilon_best = epsilon.clone()

    t = 0
    while True: 
        X_.requires_grad = True
        opt = optim.SGD([X_], lr=0.1)
        loss = nn.CrossEntropyLoss()(net(normalize(X_)),y)
        opt.zero_grad()
        loss.backward()

        with torch.no_grad(): 
            # take a step
            if norm == 'linfinity': 
                X_[~err] += alpha*torch.sign(X_.grad[~err])
            elif norm == 'l2': 
                X_[~err] += (alpha*X_.grad/(X_.grad.view(X.size(0),-1).norm(dim=1).view(X.size(0),1,1,1)   ))[~err]


            

            if ball == 'linfinity':
              X_ = torch.min(X_, X + epsilon.view(X.size(0), 1, 1,1))
              X_ = torch.max(X_, X - epsilon.view(X.size(0), 1, 1,1))
            elif ball == 'l2':
              X_[~err] = X[~err] + (X_[~err] - X[~err]).renorm(p=2, dim=0, maxnorm=epsilon[~err][0].item() )

            X_ = torch.clamp(X_, min=xmin, max=xmax)

            
            
            err = (net(normalize(X_)).max(1)[1] != y)
            err_rate = err.sum().item()/batch_size
            if err_rate > err_best.sum().item()/batch_size:
                X_best = X_.clone() 
                err_best = err
                epsilon_best = epsilon.clone()

            if verbose and t % verbose == 0:
                print('Iteration= ', t, 'loss= ', loss.item(), 'epsilon_mean= ' , epsilon.mean().item(), 'err_rate= ' , err_rate )
            
            t += 1
            if err.all() or t == maxiters:
                if verbose:
                  print("Breaking attack - " , ball , " norm -  " , norm , ' at iteration ' , t , ' with epsilon =  ' , epsilon.mean().item()) 
                break

            if t > 0 and t % epsilon_iters == 0: 
                epsilon[~err] *= epsilon_factor

    return X_best