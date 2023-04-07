import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from opt import parse_args
from torchsummary import summary

args = parse_args()

class DensNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.fc_hidden1 = args.hidden1
        self.fc_hidden2 = args.hidden2
        self.drop_p = args.dropout
        self.output_size = args.output_size
        
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
        self.regressor = nn.Linear(dense121_p.classifier.in_features, self.output_size)
        self.classifier = nn.Linear(dense121_p.classifier.in_features, 1)
        
    def forward(self, x):
        x = self.den121(x) 
        #print(x.shape)
        x = x.view(x.size(0),-1)
        x = F.relu(x)

        x = self.drop(x)
        class_num = self.classifier(x)
        coordinate = self.regressor(x)
        
        return coordinate.unsqueeze(1), class_num
    
model=DensNet(args)
# for parameter in model.parameters():
#     print(parameter)

pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(pytorch_total_params)
# device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
# x = torch.randn(1,3, 224, 224).to(device)
# summary(model,x) # resnet
