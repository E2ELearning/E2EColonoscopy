from tqdm import tqdm
from constant import *
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from bayes_opt import BayesianOptimization

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from models import *
from utils import *
from opt import parse_args
from models.cnn import CNN
from model import gen_model

class Manager:
    def __init__(self, args):
        self.args = args

        self.pbounds = {
            'lr' : lr,
        }

        self.bayes_optimizer = BayesianOptimization(
            f=self.train,
            pbounds=self.pbounds,
            random_state= 0 
        )

    def train(self, lr):
        self.args.lr = lr

        train_loader, valid_loader = load_data(self.args)

        model = gen_model(args, device)

        os.makedirs(ckpt_dir, exist_ok=True)

        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        best_loss = 1e5
        train_loss = 0
        
        for epoch in range(self.args.epochs):
            model.train()

            for image, target in train_loader:
                image = image.to(device)
                target = target.to(device)  
                
                output = model(image)  # (B, class_num)
                
                loss = F.mse_loss(output, target)
                
                train_loss += loss.item()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            train_loss /= len(train_loader.dataset)

            valid_loss = self.validate(model,valid_loader)

            if valid_loss < best_loss:
                best_file = os.path.join(ckpt_dir,'loss_{:.4e}_lr_{:.4e}.pth'
                           .format(best_loss, args.lr))
                
                if os.path.isfile(best_file):
                    os.remove(best_file)
                    
                best_file = os.path.join(ckpt_dir,'loss_{:.4e}_lr_{:.4e}.pth'
                           .format(valid_loss, args.lr))
                
                torch.save(model.state_dict(), best_file)
                
                best_loss = valid_loss
                        
                
        return -best_loss

    def validate(self, model, valid_loader):
        model.eval()
        valid_loss = 0

        with torch.no_grad():
            for image, target in valid_loader:
                image = image.to(device)
                target = target.to(device)
                
                output = model(image)
                
                loss = F.mse_loss(output, target)
                
                valid_loss += loss.item()    

        valid_loss /= len(valid_loader.dataset)

        return valid_loss

if __name__=='__main__':
    args = parse_args()
    manager = Manager(args)
    
    print("Training starts.")
    manager.bayes_optimizer.maximize(init_points=init_points, n_iter=n_iter, acq='ei', xi=0.01)

    print("\nBest optimization option")
    print(manager.bayes_optimizer.max)

