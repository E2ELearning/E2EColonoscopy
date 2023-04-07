import torch
import torch.nn.functional as F
from custom_loss import compute_loss
from sklearn.metrics import f1_score

def run(model, device, train_loader, optimizer, args):
    # set model as training mode
    model.train()
    train_loss = 0
    train_reg_loss = 0
    #train_class_loss = 0
    train_f1 = 0
    
    all_y = []
    all_y_pred = []
    
    for image, target in train_loader:
        image = image.to(device)
        target = target.to(device)
        #target_class = target_class.to(device)

        
        #print("target at train",target)
        optimizer.zero_grad()
        
        #out_p, out_c = model(image)
        out_p= model(image)
       
        #y_pred_tags = torch.round(torch.sigmoid(out_c))

        # y_pred_tag = torch.softmax(out_c, dim = 1)
        # _, y_pred_tags = torch.max(y_pred_tag, dim = 1)
        
        #all_y.extend(target_class.cpu().detach().numpy().tolist())
        #all_y_pred.extend(y_pred_tags.cpu().detach().numpy().tolist())
        
        loss = compute_loss(target, out_p, args)
        
        train_loss += loss.item() * image.size(0)
        #train_reg_loss += reg_loss.item() * image.size(0)
        #train_class_loss += class_loss.item() * image.size(0)

        
        loss.backward()
        optimizer.step()
        
    train_loss /= len(train_loader.dataset)
    #train_reg_loss/= len(train_loader.dataset)
    #train_class_loss/= len(train_loader.dataset)

    #train_f1 = f1_score(all_y, all_y_pred)
    
    return train_loss
