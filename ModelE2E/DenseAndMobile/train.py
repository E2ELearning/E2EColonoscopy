import torch
import torch.nn.functional as F
from custom_loss import compute_loss
from sklearn.metrics import f1_score

def run(model, device, train_loader, optimizer, args):
    # set model as training mode
    model.train()
    train_loss = 0
    train_reg_loss = 0
    train_class_loss = 0
    train_f1 = 0
    
    all_y = []
    all_y_pred = []
    
    for image, target in train_loader:
        image = image.to(device)
        target = target.to(device)
        #target_class = target_class.to(device)
        
        #print("input shape",image.shape)
        #image=torch.squeeze(image)
        
        image=image.squeeze(1)
        #print("input shape",image.shape)
        
        
        #print("target",target.shape)
        #print("target_class",target_class.shape)
        
        optimizer.zero_grad()
        
        out_p= model(image)
        
        #print("out_p",out_p.shape)
        #print("out_c_class",out_c.shape)
        
       
        #y_pred_tags = torch.round(torch.sigmoid(out_c))

        # y_pred_tag = torch.softmax(out_c, dim = 1)
        # _, y_pred_tags = torch.max(y_pred_tag, dim = 1)
        
        #all_y.extend(target_class.cpu().detach().numpy().tolist())
        #all_y_pred.extend(y_pred_tags.cpu().detach().numpy().tolist())
        
        loss = compute_loss(target, out_p, args)
        
        train_loss += loss.item() * image.size(0)
        # train_reg_loss += reg_loss.item() * image.size(0)
        # train_class_loss += class_loss.item() * image.size(0)

        
        loss.backward()
        optimizer.step()
        
    train_loss /= len(train_loader.dataset)
    # train_reg_loss/= len(train_loader.dataset)
    # train_class_loss/= len(train_loader.dataset)

    # train_f1 = f1_score(all_y, all_y_pred)
    
    return train_loss






















# def run(model, device, train_loader, optimizer, args):
#     # set model as training mode
#     model.train()
#     train_loss = 0
#     train_f1 = 0
    
#     all_y = []
#     all_y_pred = []
    
#     for image, target_class in train_loader:
#         image = image.to(device)
#         target_class = target_class.to(device)

#         optimizer.zero_grad()
        
#         image=image.squeeze(1)
#         #print("input shape",image.shape)
        
#         out_c = model(image)
#         #print("out_c",out_c.shape)
#         #print("out_c",out_c)
#         #target_class = torch.tensor(target_class, dtype=torch.long)
#         target_class=target_class.long()
#         target_class=target_class.squeeze(1)
#         #print("target_class",target_class.shape)
#         #print("target_class",target_class)
        
#         y_pred_tags = torch.round(torch.sigmoid(out_c))
#         #y_pred_tags=F.softmax(out_c)
#         y_pred_tags=torch.softmax(out_c, dim = 1)
#         _, predict = torch.max( y_pred_tags,dim = 1)
#         # print("y_pred_tags",y_pred_tags)
#         #print("y_pre",predict)
#         y_pred_tags=predict
        
#         #print("target_class",target_class)
#         #print("y_pred_tags",y_pred_tags)
        
#         all_y.extend(target_class.cpu().detach().numpy().tolist())
#         #print("all_y",all_y)
#         all_y_pred.extend(y_pred_tags.cpu().detach().numpy().tolist())
#         #print("all_y",all_y_pred)
        
#         loss = compute_loss(target_class,out_c, args)
        
#         train_loss += loss.item() * image.size(0)
        
#         loss.backward()
#         optimizer.step()
        
#     train_loss /= len(train_loader.dataset)

#     train_f1 = f1_score(all_y, all_y_pred)
    
#     return train_loss, train_f1