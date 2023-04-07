import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout

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

# 2D CNN encoder using ResNet-34 pretrained
class CNNregression(nn.Module):
    def __init__(self,args):

        super(CNNregression, self).__init__()
        self.fc_hidden1 = args.hidden1
        self.fc_hidden2 = args.hidden2
        self.drop_p = args.dropout
        self.output_size = args.output_size

        resnet = models.resnet34(pretrained=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        
        
        self.drop = nn.Dropout(p = self.drop_p)
        self.regressor = nn.Linear(resnet.fc.in_features, self.fc_hidden1)
        self.drop1 = nn.Dropout(p = self.drop_p)
        self.regressor1 = nn.Linear(self.fc_hidden1, self.output_size)
        
    def forward(self, x):
        x = self.resnet(x[:,0,...])
        x = x.view(x.size(0),-1)
        ### x.view(x.size(0), -1) is flattening the tensor, this is because the Linear layer only accepts a vector (1d array)
        x = F.relu(x, inplace = True)
        
        x = self.drop(x)
        x = self.regressor(x)
        x = self.drop1(x)
        coordinate = self.regressor1(x)
        
        return coordinate.unsqueeze(1)
 
class CNNregression1(nn.Module):
    def __init__(self,args):

        super(CNNregression1, self).__init__()
        self.fc_hidden1 = args.hidden1
        self.fc_hidden2 = args.hidden2
        self.drop_p = args.dropout
        self.output_size = args.output_size

        resnet = models.resnet34(pretrained=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        
        
        self.drop = nn.Dropout(p = self.drop_p)
        self.regressor = nn.Linear(resnet.fc.in_features, self.fc_hidden1)
        self.regressor0 = nn.Linear(self.fc_hidden1, self.fc_hidden2)
        self.drop1 = nn.Dropout(p = self.drop_p)
        self.regressor1 = nn.Linear(self.fc_hidden2, self.output_size)
        
    def forward(self, x):
        x = self.resnet(x[:,0,...])
        x = x.view(x.size(0),-1)
        ### x.view(x.size(0), -1) is flattening the tensor, this is because the Linear layer only accepts a vector (1d array)
        x = F.relu(x, inplace = True)
        
        x = self.drop(x)
        x = self.regressor(x)
        x = self.drop1(x)
        
        x = self.regressor0(x)
        x = self.drop1(x)
        
        coordinate = self.regressor1(x)
        
        return coordinate.unsqueeze(1)
    
# 2D CNN encoder using ResNet-34 pretrained
class Res18(nn.Module):
    def __init__(self,args):

        super(Res18, self).__init__()
        self.fc_hidden1 = args.hidden1
        self.fc_hidden2 = args.hidden2
        self.drop_p = args.dropout
        self.output_size = args.output_size

        resnet = models.resnet18(pretrained=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
 
        self.drop = nn.Dropout(p = self.drop_p)
        self.regressor = nn.Linear(resnet.fc.in_features, self.fc_hidden1)
        self.drop1 = nn.Dropout(p = self.drop_p)
        self.regressor1 = nn.Linear(self.fc_hidden1, self.output_size)
        
    def forward(self, x):
        x = self.resnet(x[:,0,...])
        x = x.view(x.size(0),-1)
        ### x.view(x.size(0), -1) is flattening the tensor, this is because the Linear layer only accepts a vector (1d array)
        x = F.relu(x, inplace = True)
        
        x = self.drop(x)
        x = self.regressor(x)
        x = self.drop1(x)
        coordinate = self.regressor1(x)
        
        return coordinate.unsqueeze(1)
    

class MOBILEV2(nn.Module):
    def __init__(self,args):
        super(MOBILEV2, self).__init__()
        self.fc_hidden1 = args.hidden1
        self.fc_hidden2 = args.hidden2
        self.drop_p = args.dropout
        self.output_size = args.output_size

        # https://medium.com/nerd-for-tech/know-about-inception-v2-and-v3-implementation-using-pytorch-b1d96b2c1aa5
        ## https://github.com/pytorch/vision/blob/662373f6057bb0d39eaf6e5fde3083639ed93af3/torchvision/models/inception.py#L168-L171
        
        self.MoV =  models.mobilenet_v2(pretrained=True)
        #num_ftrs = self.MoV.classifier[1].in_features
        modules = list(self.MoV.children())[:-1]      # delete the last fc layer.
        self.MoV2 = nn.Sequential(*modules)
        
        self.drop = nn.Dropout(p = self.drop_p)
        #self.regressor = nn.Linear(num_ftrs, self.fc_hidden1)
        self.regressor = nn.Linear(62720, self.fc_hidden1)
        self.regressor1 = nn.Linear(self.fc_hidden1, self.output_size)
    
                 
        
    def forward(self, x):
        x = self.MoV2(x[:,0,...])
        #print(x.shape)
        x = x.view(x.size(0),-1)
        #print(x.shape)
        ### x.view(x.size(0), -1) is flattening the tensor, this is because the Linear layer only accepts a vector (1d array)
        #x = F.relu(x, inplace = True)
        x = F.relu(x)
        x = self.drop(x)
        x = self.regressor(x)
        x = self.drop(x)
        coordinate = self.regressor1(x)
        
        return coordinate.unsqueeze(1)
    
       
## ---------------------- end of CRNN module ---------------------- ##
# class CNNregression(nn.Module):
#     def __init__(self,args):

#         super(CNNregression, self).__init__()
#         self.fc_hidden1 = args.hidden1
#         self.fc_hidden2 = args.hidden2
#         self.drop_p = args.dropout
#         self.output_size = args.output_size


#         Incep = models.inception_v3(pretrained=True)
#         modules = list(Incep.children())[:-1]      # delete the last fc layer.
#         self.Incep = nn.Sequential(*modules)
        
#         # self.regressor = nn.Sequential(
#         #     nn.Linear(resnet.fc.in_features, self.fc_hidden1, bias = True),
#         #     nn.BatchNorm1d(self.fc_hidden1),
#         #     nn.ReLU(inplace=True),
#         #     nn.Dropout(p = self.drop_p),
#         #     nn.Linear(self.fc_hidden1, self.fc_hidden2),
#         #     nn.BatchNorm1d(self.fc_hidden2),
#         #     nn.ReLU(inplace=True),
#         #     nn.Dropout(p = self.drop_p),
#         #     nn.Linear(self.fc_hidden2, self.output_size)
#         # )
        
#         # self.classifier = nn.Sequential(
#         #     nn.Linear(resnet.fc.in_features, self.fc_hidden1, bias = True),
#         #     nn.BatchNorm1d(self.fc_hidden1),
#         #     nn.ReLU(inplace=True),
#         #     nn.Dropout(p = self.drop_p),
#         #     nn.Linear(self.fc_hidden1, self.fc_hidden2),
#         #     nn.BatchNorm1d(self.fc_hidden2),
#         #     nn.ReLU(inplace=True), 
#         #     nn.Dropout(p = self.drop_p),
#         #     nn.Linear(self.fc_hidden2, 2)
            
#         # )
        
#         # self.drop = nn.Dropout(p = self.drop_p)
#         # self.regressor = nn.Linear(resnet.fc.in_features, self.output_size)
#         # self.classifier = nn.Linear(resnet.fc.in_features, 1)
        
        
#         self.drop = nn.Dropout(p = self.drop_p)
#         self.regressor = nn.Linear(Incep.fc.in_features, self.fc_hidden1)
#         self.regressor1 = nn.Linear(self.fc_hidden1, self.output_size)
        
#     def forward(self, x):
#         x = self.Incep(x[:,0,...])
#         x = x.view(x.size(0),-1)
#         ### x.view(x.size(0), -1) is flattening the tensor, this is because the Linear layer only accepts a vector (1d array)
        
#         x = F.relu(x, inplace = True)
        
#         x = self.drop(x)
#         x = self.regressor(x)
#         coordinate = self.regressor1(x)
        
#         return coordinate.unsqueeze(1)

class CUSTOMNet(Module):   
    def __init__(self):
        super(CUSTOMNet, self).__init__()

        self.cnn_layers = Sequential(
        nn.Conv2d(3,32,kernel_size=3,padding=0,stride=1),
        nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.Conv2d(32,32,kernel_size=3,stride=1,padding=0),
        nn.ReLU(),
        nn.MaxPool2d(2,2),
        Dropout (p = 0.5),
    
        nn.Conv2d(32,64,kernel_size=3,padding=0,stride=1),
        nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.Conv2d(64,64,kernel_size=3,stride=1,padding=0),
        nn.ReLU(),
        nn.MaxPool2d(2,2),
        Dropout (p = 0.5),
        
        nn.Conv2d(64,128,kernel_size=3,padding=0,stride=1),
        nn.ReLU(),
        nn.BatchNorm2d(128),
        nn.Conv2d(128,128,kernel_size=3,padding=0,stride=1),
        nn.ReLU(),
        nn.MaxPool2d(2,2),
        Dropout (p = 0.5))

        self.linear_layers = Sequential(
                Linear(128 * 24 * 24, 500))

        
        self.linear_layers1=Sequential(       
                nn.Linear(500, 100),
                nn.ReLU(),
                nn.Linear(100, 50),
                nn.ReLU(),
                Dropout (p = 0.5),
                nn.Linear(50 , 2),
                nn.Tanh())

    # Defining the forward pass    
    def forward(self, x):
        x = self.cnn_layers(x)
        print(x.size())
        x = x.reshape(x.size(0), -1)
        #x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        x = self.linear_layers1(x)
        return x
    
# model = CUSTOMNet()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
# model = model.to(device)
# print(model)
# summary(model, (3, 224, 224))
