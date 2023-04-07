import os
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

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

    all_y = []
    all_y_pred = []
    test_loss = 0
    test_reg_loss = 0
    test_cls_loss = 0

    all_class = []
    all_class_pred = []
    
    with torch.no_grad():
        for image, target, target_class in loader:
            image = image.to(device)
            target = target.to(device)
            target_class = target_class.to(device)
            
            out_p, out_c, h, c = model(image, h, c)

            loss, reg_loss, cls_loss = compute_loss(target, target_class, out_p, out_c, args)

            test_loss += loss.item() * image.size(0)
            test_reg_loss += reg_loss.item() * image.size(0)
            test_cls_loss += cls_loss.item() * image.size(0)
        
            all_y.extend(target.cpu().numpy().tolist())
            all_y_pred.extend(out_p.cpu().numpy().tolist())
            
            y_pred_tag = torch.softmax(out_c, dim = 1)
            _, y_pred_tags = torch.max(y_pred_tag, dim = 1)
            
            all_class.extend(target_class.cpu().numpy().tolist())
            all_class_pred.extend(y_pred_tags.cpu().numpy().tolist())
    
    test_loss /= len(loader.dataset)            
    test_reg_loss /= len(loader.dataset)            
    test_cls_loss /= len(loader.dataset)            
        
    return np.array(all_y_pred), np.array(all_y), np.array(all_class_pred), np.array(all_class),\
        test_loss, test_reg_loss, test_cls_loss

def predict(args):
    # Check GPU to use CUDA
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")   # use CPU or GPU
    
    # Create model
    model = gen_model(args, device)
    
    # Load model
    print('\nLoading best model from {}'.format(args.save_model_path))
    model.load_state_dict(torch.load(os.path.join(args.save_model_path,\
            'best_model.pth')))

    # Read and list iamge folders
    image_folder = os.path.join(args.data_path, 'images')
    all_X_list = sorted(os.listdir(image_folder))

    # train, valid split
    train_image_list, test_image_list = \
        train_test_split(all_X_list, test_size=0.2, random_state = 0)

    train_dataset = read_data(train_image_list, args)
    
    # Mean and standard deviation of training images
    print('\nCalculating mean and std of training dataset')
    mean, std = mean_std(args, train_dataset)
    # mean = [0.567,0.390,0.388]
    # std = [0.159,0.155,0.168]

    res_size = args.input_size
    add_targets = {'image{}'.format(i) : 'image' for i in range(args.seq_len)}
    # Transform inputs
    transform = A.Compose([ A.Resize(res_size,res_size),
                            A.Normalize(mean = mean, std= std),
                            transforms.ToTensorV2()],
                            additional_targets= add_targets,
                            keypoint_params= A.KeypointParams(format = 'xy', remove_invisible = False))
    
    # Data loading parameters
    params = {'batch_size': 256, 'shuffle': False, 'num_workers': 4, 'pin_memory': True, 'drop_last': False} \
        if not args.no_cuda else {}

    y_true = []
    y_pred = []
    
    test_loss = 0
    reg = 0
    cls = 0
    total_len_data = 0
    
    for exp in test_image_list:
        
        # read images and targets from a valid experiment
        dataset = read_data([exp], args)
        
        # Dataset
        dataset = RC_Dataset(dataset, transform, args)

        # DataLoader
        data_loader = data.DataLoader(dataset, **params)

        # Prediction
        # print("Predicting {}".format(exp))
        all_y_pred, all_y, all_class_pred, all_class, loss, reg_loss, cls_loss \
            = CRNN_final_prediction(model, device, data_loader, args)
        
        test_loss += loss * len(data_loader.dataset)
        reg += reg_loss * len(data_loader.dataset)
        cls += cls_loss * len(data_loader.dataset)
        
        y_true.extend(all_class.tolist())
        y_pred.extend(all_class_pred.tolist())
                
        total_len_data += len(data_loader.dataset)
        
        # Plot predictions
        fig = plt.figure(figsize=(10, 5))
        plt.title(exp)
        
        plt.subplot(3, 1, 1)
        plt.plot(all_y[...,0])
        plt.plot(all_y_pred[...,0])
        plt.legend(['target', 'pred'], loc="upper left")
        plt.xlabel('frame')
        plt.ylabel('steer X')

        plt.subplot(3, 1, 2)   
        plt.plot(all_y[...,1])
        plt.plot(all_y_pred[...,1])
        plt.legend(['target', 'pred'], loc="upper left")
        plt.xlabel('frame')
        plt.ylabel('steer Y')
        
        plt.subplot(3, 1, 3)   
        plt.plot(all_class)
        plt.plot(all_class_pred)
        plt.legend(['target', 'pred'], loc="upper left")
        plt.xlabel('frame')
        plt.ylabel('class')

        title = "./Prediction_{}.png".format(exp)
        save_figure_path = os.path.join(args.save_model_path, 'predictions', 'figures')
        os.makedirs(save_figure_path, exist_ok=True)
        plt.savefig(os.path.join(save_figure_path,title), dpi=600)
    plt.close('all')
        
    f1 = f1_score(y_true, y_pred)
    print('Test Loss : {:.4e} / reg : {:.4e} / cls : {:.4e} / F1 : {:.3f}'\
        .format(test_loss / total_len_data, reg / total_len_data, cls / total_len_data, f1))
    
    with open('result.txt','a') as file:
        file.write('Test Loss : {:.4e} / reg : {:.4e} / cls : {:.4e} / F1 : {:.3f}\n'\
        .format(test_loss / total_len_data, reg / total_len_data, cls / total_len_data, f1))
        
        
if __name__=='__main__':

    args = parse_args()
    print('\nPrediction Conditions')
    for arg in vars(args):
        print (arg, ' : ', getattr(args, arg))
    predict(args)
    