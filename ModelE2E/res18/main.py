import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from utils import *
from opt import parse_args
from model import gen_model
import train
import valid
from plot_prediction import predict
from scheduler import CosineAnnealingWarmUpRestarts
from contextlib import redirect_stdout
from torch.optim.lr_scheduler import ReduceLROnPlateau
#from torchsummary import summary

torch.cuda.empty_cache()
def set_seed(seed):
    # random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
set_seed(42)

def main(args):
    
    # Check GPU to use CUDA
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")   # use CPU or GPU
    
    model = gen_model(args, device)
    #print(model)
    #summary(model, (1,3, 224, 224))
    
    
    # Define optimizer      
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) 
    # optimizer = torch.optim.SGD(model.parameters(), lr = args.lr)
    # Define schedular
    # scheduler = ReduceLROnPlateau(optimizer, 'min',patience=3)
    scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=args.epochs, T_mult=1, eta_max=1e-4,  T_up=1, gamma=0.5)
    
    # Load Data
    train_loader, valid_loader = load_data(args)
    # for batch_idx, sample in enumerate(train_loader):
    #     # print("image type", type(sample[0][0]))
    #     # print("pixel type", type(sample[1][0]))
    #     # print("class type", type(sample[2][0]))
    #     IMcheck=sample[0][0]
    #     tensor_image=torch.squeeze(IMcheck)
    #     new=tensor_image.permute(1, 2, 0)
    #     new=new.detach().cpu().numpy()
        
    #     lablecheck=sample[2][0].detach().cpu().numpy()
    #     PIx=sample[1][0].detach().cpu().numpy()
        
    #     # print("image 0",new.shape)
    #     # print("lable ",lablecheck.shape,"  ", lablecheck, type(lablecheck))
    #     # print("pix ",PIx.shape,"    ",PIx)
        
    #     a = int(PIx[0,0]*224)
    #     b   =int(PIx[0,1]*224)
    
    #     print(" x ",a,"       ",PIx[0,0])
    #     print(" y ",b,"         ",PIx[0,1])
        
    #     print("-----------------") 
    #     plt.scatter([a], [b],c='red', s=50)
    #     plt.imshow(new)
    #     plt.show()  
    #     pass

    # print("----------------------------------------------------------------------------------")
    
    # record training process
    epoch_train_losses = []
    epoch_valid_losses = []

    # Evaluate Best loss
    best_loss = 1e5
    
    # start training
    
    print("\nStart Training")
    for epoch in range(args.epochs):
        # train, valid model

        train_loss = train.run(model, device, train_loader, optimizer, args)
        valid_loss = valid.run(model, device, valid_loader, args)
        try:
            lr = optimizer.param_groups[0]['lr']
            scheduler.step()
        except:
            pass
        
        best_loss = min(valid_loss, best_loss)
        
        if valid_loss == best_loss:
            print("bestloss at epoch",epoch, "and validationloss is",valid_loss)
            os.makedirs(args.model_name, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.model_name,'best_model.pth'))  # save best model
            best_train_loss   = train_loss
            best_valid_loss  = valid_loss 
        
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(args.model_name,'model_epoch_{}.pth'.format(epoch+1)))  # save best model

        print('Epoch: {} loss: {:.4e},  / val_loss: {:.4e},  / lr : {:.4e}'
            .format(epoch+1, train_loss,  valid_loss, lr))

        # save results
        epoch_train_losses.append(train_loss)
        epoch_valid_losses.append(valid_loss)

        # save all train valid results
        T = np.array(epoch_train_losses)
        V = np.array(epoch_valid_losses)

        save_loss_path = os.path.join(args.model_name, 'loss')
        os.makedirs(save_loss_path, exist_ok=True)
        np.save(os.path.join(save_loss_path,'CRNN_epoch_training_loss.npy'), T)
        np.save(os.path.join(save_loss_path,'CRNN_epoch_valid_loss.npy'), V)

    with open('result{}.csv','a') as file:
        file.write('{},{},{},{},{:.4e},{:.4e}'\
            .format(args.model, args.epochs, args.alpha, args.seq_len, best_train_loss, best_valid_loss))

    # Plot Loss
    fig = plt.figure(figsize=(8, 4))
    plt.plot(np.arange(1, args.epochs + 1), T)  # train loss (on epoch end)
    plt.plot(np.arange(1, args.epochs + 1), V)  # valid loss (on epoch end)
    plt.title("model loss")
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend(['train', 'valid'], loc="upper left")
    
    title = os.path.join(save_loss_path,"fig_Loss_RCnet.png")
    plt.savefig(title, dpi=600)

 
if __name__=='__main__':
    
    args = parse_args()
    print('\nTraining Conditions')
    for arg in vars(args):
        print (arg, ' : ', getattr(args, arg))

    main(args)
    predict(args)
