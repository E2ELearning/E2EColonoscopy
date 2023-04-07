import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


# 2D CNN encoder using ResNet-18 pretrained
class CNN(nn.Module):
    def __init__(self,args):

        super(CNN, self).__init__()
        self.fc_hidden1 = args.hidden1
        self.fc_hidden2 = args.hidden2
        self.drop_p = args.dropout
        self.output_size = args.output_size

        resnet = models.resnet34(pretrained=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)

        # self.regressor = nn.Sequential(
        #     nn.Linear(resnet.fc.in_features, self.fc_hidden1, bias = True),
        #     nn.BatchNorm1d(self.fc_hidden1),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(p = self.drop_p),
        #     nn.Linear(self.fc_hidden1, self.fc_hidden2),
        #     nn.BatchNorm1d(self.fc_hidden2),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(p = self.drop_p),
        #     nn.Linear(self.fc_hidden2, self.output_size)
        # )
        
        # self.classifier = nn.Sequential(
        #     nn.Linear(resnet.fc.in_features, self.fc_hidden1, bias = True),
        #     nn.BatchNorm1d(self.fc_hidden1),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(p = self.drop_p),
        #     nn.Linear(self.fc_hidden1, self.fc_hidden2),
        #     nn.BatchNorm1d(self.fc_hidden2),
        #     nn.ReLU(inplace=True), 
        #     nn.Dropout(p = self.drop_p),
        #     nn.Linear(self.fc_hidden2, 2)
            
        # )
        
        self.drop = nn.Dropout(p = self.drop_p)
        self.regressor = nn.Linear(resnet.fc.in_features, self.output_size)
        self.classifier = nn.Linear(resnet.fc.in_features, 1)
        
    def forward(self, x):
        x = self.resnet(x[:,0,...])
        x = x.view(x.size(0),-1)
        x = F.relu(x, inplace = True)
        
        x = self.drop(x)
        class_num = self.classifier(x)
        coordinate = self.regressor(x)
        
        return coordinate.unsqueeze(1), class_num
    
## ---------------------- end of CRNN module ---------------------- ##
# Total params: 21,286,211
# Trainable params: 21,286,211
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 0.57
# Forward/backward pass size (MB): 96.29
# Params size (MB): 81.20
# Estimated Total Size (MB): 178.06

class DensNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        # self.fc_hidden1 = args.hidden1
        # self.fc_hidden2 = args.hidden2
        self.drop_p = args.dropout
        self.output_size = args.output_size
        self.hidden=128
        
        dense121_p = models.densenet121(pretrained=True)

        #print(preloaded)
        #print(preloaded )
        modules = list(dense121_p.children())[:-1]      # delete the last fc layer.
        #print("---------------------------")
        #print(modules)
        self.den121 = nn.Sequential(*modules)
        

        ### Resnet uses the name fc for its last layer 
        # ##while Densenet uses the name classifier for its last layer
        self.drop = nn.Dropout(p = self.drop_p)
        # self.regressor = nn.Linear(dense121_p.classifier.in_features, self.output_size)
        # self.classifier = nn.Linear(dense121_p.classifier.in_features, 1)
        
        self.regressorF = nn.Linear(50176, self.hidden)
        self.regressor = nn.Linear(self.hidden, self.output_size)
        
        self.classifierF = nn.Linear(50176,self.hidden)
        self.classifier = nn.Linear(self.hidden, 1)
        
    def forward(self, x):
        x = self.den121(x) 
        #print(x.shape)
        x = x.view(x.size(0),-1)
        x = F.relu(x)

        x = self.drop(x)
        #print("check model shape",x.shape)
        x_class=self.classifierF(x)
        x_class = F.relu(x_class)
        class_num = self.classifier(x_class)
        
        x_pixel=self.regressorF(x)
        x_pixel = F.relu(x_pixel)
        coordinate = self.regressor(x_pixel)
        
        
        return coordinate.unsqueeze(1), class_num
    
    

class MOBILEV2(nn.Module):
    def __init__(self, args):
        super().__init__()
        # self.fc_hidden1 = args.hidden1
        # self.fc_hidden2 = args.hidden2
        self.drop_p = args.dropout
        self.output_size = args.output_size
        self.hidden=128
        
        self.MoV =  models.mobilenet_v2(pretrained=True)

        #print(preloaded)
        #print(preloaded )
        modules = list(self.MoV.children())[:-1]      # delete the last fc layer.
        #print("---------------------------")
        #print(modules)
        self.MoV2 = nn.Sequential(*modules)
        

        self.drop = nn.Dropout(p = self.drop_p)
        
        self.regressorF = nn.Linear(62720, self.hidden)
        self.regressor = nn.Linear(self.hidden, self.output_size)
        
        self.classifierF = nn.Linear(62720,self.hidden)
        self.classifier = nn.Linear(self.hidden, 1)
        
    def forward(self, x):
        x = self.MoV2(x) 
        #print(x.shape)
        x = x.view(x.size(0),-1)
        x = F.relu(x)

        x = self.drop(x)
        #print("check model shape",x.shape)
        x_class=self.classifierF(x)
        x_class = F.relu(x_class)
        class_num = self.classifier(x_class)
        
        x_pixel=self.regressorF(x)
        x_pixel = F.relu(x_pixel)
        coordinate = self.regressor(x_pixel)
        
        
        return coordinate.unsqueeze(1), class_num