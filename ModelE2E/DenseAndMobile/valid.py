import torch
import torch.nn.functional as F
from custom_loss import compute_loss
from sklearn.metrics import f1_score

def run(model, device, valid_loader, args):
    # set model as validing mode
    model.eval()
    valid_loss = 0
    valid_reg_loss = 0
    valid_class_loss = 0
    
    all_y = []
    all_y_pred = []
    
    with torch.no_grad():
        for image, target in valid_loader:
            image = image.to(device)
            target = target.to(device)
            #target_class = target_class.to(device)
            
            image=image.squeeze(1)
            
            
            out_p= model(image)

            loss= compute_loss(target,  out_p, args)
            
        
            valid_loss += loss.item() * image.size(0)
          
    valid_loss /= len(valid_loader.dataset)

    
    return valid_loss
