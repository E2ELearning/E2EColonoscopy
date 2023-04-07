import os
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, mean_absolute_percentage_error, accuracy_score, classification_report

import torch
import torch.utils.data as data
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import transforms

from model import gen_model
from opt import parse_args
from utils import *
from custom_loss import compute_loss

def CRNN_final_prediction(model, device, loader, args):

    model.eval()
    test_loss = 0
    all_y = []
    all_y_pred = []
    
    with torch.no_grad():
        for image, target in loader:
            image = image.to(device)
            target = target.to(device)
            #print(target)    
            image=image.squeeze(1)
            out_p= model(image)

            loss = compute_loss(target, out_p,  args)
            
            all_y.extend(target.cpu().numpy().tolist())
            all_y_pred.extend(out_p.cpu().numpy().tolist())

            test_loss += loss.item() * image.size(0)
    
    test_loss /= len(loader.dataset)               
        
    return np.array(all_y_pred), np.array(all_y), test_loss

def predict(args):
    # Check GPU to use CUDA
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")   # use CPU or GPU
    
    # Create model
    model = gen_model(args, device)
    
    # Load model
    print('\nLoading best model from {}'.format(args.model_name))
    model.load_state_dict(torch.load(os.path.join(args.model_name,\
            'best_model.pth')))

    # Read and list iamge folders
    image_folder = os.path.join(args.dataset, 'images')
    
    sequence_list = sorted(os.listdir(image_folder))

    # train, valid split
    train_image_list, test_image_list = \
        train_test_split(sequence_list, test_size=0.2, random_state = 0)
        
    train_dataset = read_data(train_image_list, args)
    
    # Mean and standard deviation of training images
    print('\nCalculating mean and std of training dataset')
    mean, std = mean_std(args, train_dataset)
    
    # 2 dataset 

    res_size = args.input_size
    add_targets = {'image{}'.format(i) : 'image' for i in range(args.seq_len)}
    # Transform inputs
    transform = A.Compose([ A.Resize(res_size,res_size),
                            A.Normalize(mean = mean, std= std),
                            transforms.ToTensorV2()])

    
    # Data loading parameters
    params = {'batch_size': 1, 'shuffle': False, 'num_workers': 4, 'pin_memory': True, 'drop_last': False} \
        if not args.no_cuda else {}

    test_loss = 0
    reg = 0
    total_len_data = 0
    reg_true = []
    reg_pred = []

    for exp in test_image_list:
        
        print(exp)
        # read images and targets from a valid experiment
        dataset = read_data([exp], args)

        # Dataset
        dataset = RC_Dataset(dataset, transform, args)

        # DataLoader
        data_loader = data.DataLoader(dataset, **params)

        # Prediction
        # print("Predicting {}".format(exp))
        all_y_pred, all_y, loss= CRNN_final_prediction(model, device, data_loader, args)
        #np.save('Y_pre{}'.format(exp),all_y_pred)
        #np.save('Y_target{}'.format(exp),all_y)
        print(exp, 'rmse : ', loss**0.5) 

        test_loss += loss * len(data_loader.dataset)
        
        reg_true.extend(all_y.tolist())
        reg_pred.extend(all_y_pred.tolist()) 
        total_len_data += len(data_loader.dataset)
        
        fig = plt.figure(figsize=(10, 6))
        plt.title(exp)
        
        numberofpoint= all_y[...,0]
        print()
        plt.subplot(5, 1, 1)
        plt.plot(all_y[...,0])
        plt.plot(all_y_pred[...,0])
        plt.legend(['target', 'pred'], loc="upper right")
        plt.xlim([-100,len(numberofpoint) + 200])
        plt.xlabel('frame')
        plt.ylabel('steer X')

        plt.subplot(5, 1, 2)   
        plt.plot(all_y[...,1])
        plt.plot(all_y_pred[...,1])
        plt.legend(['target', 'pred'], loc="upper right")
        plt.xlim([-100,len(numberofpoint) + 200])
        plt.xlabel('frame')
        plt.ylabel('steer Y')
        
        plt.subplot(5, 1, 3)   
        plt.plot(all_y[...,2])
        plt.plot(all_y_pred[...,2])
        plt.legend(['target', 'pred'], loc="upper right")
        plt.xlim([-100,len(numberofpoint) + 200])
        plt.xlabel('frame')
        plt.ylabel('Fedding velocity')
        
        plt.subplot(5, 1, 4)   
        plt.plot(all_y[...,3])
        plt.plot(all_y_pred[...,3])
        plt.legend(['target', 'pred'], loc="upper right")
        plt.xlim([-100,len(numberofpoint) + 200])
        plt.xlabel('frame')
        plt.ylabel('Collision')
        
        perror = np.sqrt((all_y[...,0]- all_y_pred[...,0])**2 + (all_y[...,1]-all_y_pred[...,1])**2)
        plt.subplot(5, 1, 5)   
        plt.plot(perror)
        plt.legend(['steering error'], loc="upper right")
        
        plt.xlim([-100,len(numberofpoint) + 200])
        plt.ylim([-0.2,1.5])
        plt.xlabel('frame')
        plt.ylabel('All regression error')
        


        plt.tight_layout()
        title = "./Prediction_{}.png".format(exp)
        save_figure_path = os.path.join(args.model_name, 'predictions', 'figures')
        os.makedirs(save_figure_path, exist_ok=True)
        plt.savefig(os.path.join(save_figure_path,title), dpi=600)
        
        # fig2 = plt.figure(figsize = (8,4))
        # plt.subplot(1,2,1)
        # plt.scatter(all_y[...,0],all_y_pred[...,0], s = 3)
        
        # plt.subplot(1,2,2)
        # plt.scatter(all_y[...,1],all_y_pred[...,1], s = 3)
        # plt.show()
        
    plt.close('all')
    
    np.save('./reg_true.npy', reg_true)
    np.save('./reg_pred.npy', reg_pred)
   
    print('Test Loss : {:.4e} '\
        .format(test_loss / total_len_data))
    
    with open('result{}.csv'.format('nvs'),'a') as file:
        file.write('{:.4e}\n'\
        .format(test_loss / total_len_data))


if __name__=='__main__':

    args = parse_args()
    print('\nPrediction Conditions')
    for arg in vars(args):
        print (arg, ' : ', getattr(args, arg))
    predict(args)
    