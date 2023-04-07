import os
import numpy as np
import cv2
import torch
from torch.utils import data
import random

class RC_Dataset(data.Dataset): 
    def __init__(self, dataset, transform, args):        
        self.dataset = dataset
        self.transform = transform
        self.mode = False
        self.args = args
        self.seq_len = args.seq_len
        self.target_len = args.target_len
        self.add_targets = {'image{}'.format(i) : 'image' for i in range(self.seq_len-1)}
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        
        dataset = self.dataset.iloc[idx]
        #print("-----dataset--",dataset)
        
        image_list = dataset[:self.seq_len].values
        target_coordinate = []
        #print("----finish dataset----")
        
        for i in range(self.target_len):
            #print(dataset[self.seq_len+i])
            #print(dataset[self.seq_len+self.target_len+i])
            target_coordinate.append([dataset[self.seq_len+i], dataset[self.seq_len+self.target_len+i],dataset[self.seq_len+self.target_len+i+1],dataset[self.seq_len+self.target_len+i+2]])
            #print("----target_coordinate.append----")
            #print(target_coordinate)
        
        trans_params = {}
        for i, frame in enumerate(image_list):
            frame = os.sep.join(frame.split("/")[-2:])
            image = cv2.imread(self.args.dataset + '/images/' + frame)
            # img = cv2.resize(image,(self.args.input_size,self.args.input_size))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # self.img_resize = cv2.resize(image, (64,64))
            
            if i == 0:
                trans_params['image'] = image
            else:
                trans_params['image{}'.format(i-1)] = image
                
                
            
        # trans_params['keypoints'] = target_coordinate
                
        transformed = self.transform(**trans_params)
        
        # target_coordinate = list(transformed['keypoints'])
        # target_coordinate = [list(coord) for coord in target_coordinate]
        
        
        image_seq = []
        for i in range(self.seq_len):
            if i == 0:
                image = transformed['image']   
            else:
                image = transformed['image{}'.format(i-1)]
            
            image_seq.append(image)
          
        
        image = torch.stack(image_seq)
             
        
        image=torch.FloatTensor(image)
        #print("image",type(image))
        target_coordinate=torch.FloatTensor(target_coordinate)
        #print("target",type( target_coordinate),target_coordinate)
        
        return image, target_coordinate
    