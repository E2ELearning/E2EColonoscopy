
import os
import numpy as np
import torch
from torch.utils import data
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import transforms
from RCDataset import RC_Dataset
import pickle
## ---------------------- end of Dataloaders ---------------------- ##

def read_data(X_list, args):
    
    label_path = args.dataset + '/labels'

    
    df = pd.DataFrame()
    
    for exp in X_list:
        
        data = pd.read_csv(os.path.join(label_path, exp) + '.csv')
        data = data.replace('None', np.nan)
        data = data.dropna(axis=0)
        
        num_seq = len(data) - (args.seq_len-1)*args.step - args.target_len + 1
        
        #make sequences of images
        image_seq = [data[i:i+ (args.seq_len-1)*args.step+1 :args.step]['Path'].values for i in range(num_seq)]
        image_df = pd.DataFrame(data = image_seq)
        image_df.columns = ['image{}'.format(i) for i in range(args.seq_len)]
        

        
        
        target_x_seq = [data[i+ (args.seq_len-1)*args.step: i+ (args.seq_len-1)*args.step + args.target_len]['VeX'].values for i in range(num_seq)]
        target_x_df = pd.DataFrame(data = target_x_seq)
        target_x_df.columns = ['VeX{}'.format(i) for i in range(args.target_len)]
        
        #make sequences of data
        target_y_seq = [data[i+ (args.seq_len-1)*args.step: i+ (args.seq_len-1)*args.step + args.target_len]['VeY'].values for i in range(num_seq)]
        target_y_df = pd.DataFrame(data = target_y_seq)
        target_y_df.columns = ['VeY{}'.format(i) for i in range(args.target_len)]
        
        target_F_seq = [data[i+ (args.seq_len-1)*args.step: i+ (args.seq_len-1)*args.step + args.target_len]['Feedv'].values for i in range(num_seq)]
        target_F_df = pd.DataFrame(data = target_F_seq)
        target_F_df.columns = ['Feedv{}'.format(i) for i in range(args.target_len)]
        
        #make sequences of data
        target_Coll_seq = [data[i+ (args.seq_len-1)*args.step: i+ (args.seq_len-1)*args.step + args.target_len]['coll'].values for i in range(num_seq)]
        target_Coll_df = pd.DataFrame(data = target_Coll_seq)
        target_Coll_df.columns = ['coll{}'.format(i) for i in range(args.target_len)]
        
        
        
        #make sequences of class
        
        # target_class_seq = [data[i+ (args.seq_len-1)*args.step: i+ (args.seq_len-1)*args.step + args.target_len]['Class'].values for i in range(num_seq)]
        # target_class_df = pd.DataFrame(data = target_class_seq)
        # target_class_df.columns = ['Class{}'.format(i) for i in range(args.target_len)]
        
        # target_x_df /= (640-1)
        # target_y_df /= (480-1)
        target_F_df=(target_F_df-0.)/(63.-0.)
        
        #dataset = pd.concat([image_df, target_x_df, target_y_df, target_class_df], axis = 1)
        
        #dataset = pd.concat([image_df,target_F_df,target_Coll_df], axis = 1)
        dataset = pd.concat([image_df, target_x_df, target_y_df,target_F_df,target_Coll_df], axis = 1)
        
        df = pd.concat([df, dataset])
        
    df.index = np.arange(0, len(df))
    #print("read data function")
    #print(df)

    return df


def load_data(args):
    
    # path to images
    print("args.dataset",args.dataset)
    image_folder = os.path.join(args.dataset, 'images')

    # read all image folders and sort
    all_X_list = sorted(os.listdir(image_folder))
    print(all_X_list)
    # train, test split
    train_image_list, test_image_list = \
        train_test_split(all_X_list, test_size=0.2, random_state = 0)

    # train_image_list, test_image_list = \
    #     train_test_split(train_image_list, test_size=0.2, random_state = 0)

    print(train_image_list, test_image_list)
    # make 
    train_dataset = read_data(train_image_list, args)
    test_dataset = read_data(test_image_list, args)
    
    print('Train_set : ', train_image_list)
    print('Test_set : ', test_image_list)
    print(len(train_dataset), len(test_dataset))
    # train_dataset, test_dataset = train_test_split(train_dataset, test_size = 0.3, stratify= train_dataset['Class0'])

    # mean and standard deviation to normalize images
    mean, std = mean_std(args, train_dataset)
    
    
    ## 2 dataset 
    #mean = [ 0.553,0.383,0.426]
    #std = [0.151,0.140,0.159]

    # size of inputs(images)
    
    res_size = args.input_size
    
    add_targets = {'image{}'.format(i) : 'image' for i in range(args.seq_len)}
    # train_transform = A.Compose([
    #                             A.Resize(res_size,res_size),
    #                             A.VerticalFlip(p=0.5),
    #                             A.HorizontalFlip(p=0.5),
    #                             A.Rotate(45, p = 0.5, border_mode= 1),
    #                             A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5, p=0.5),
    #                             A.Normalize(mean = mean, std= std),
    #                             transforms.ToTensorV2(),
    #                             ],
    #                             additional_targets= add_targets,
    #                             keypoint_params= A.KeypointParams(format = 'xy', remove_invisible = False))
        
    # test_transform = A.Compose([ A.Resize(res_size,res_size),
    #                         A.Normalize(mean = mean, std= std),
    #                         transforms.ToTensorV2()],
    #                         additional_targets= add_targets,
    #                         keypoint_params= A.KeypointParams(format = 'xy', remove_invisible = False))

    train_transform = A.Compose([
                                A.Resize(res_size,res_size),
                                A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5, p=0.5),
                                A.Normalize(mean = mean, std= std),
                                transforms.ToTensorV2()
                                ])

        
    test_transform = A.Compose([ A.Resize(res_size,res_size),
                            A.Normalize(mean = mean, std= std),
                            transforms.ToTensorV2()])


    train_params = {'batch_size': args.batch_size, 'shuffle': True, 'num_workers': 4, 'pin_memory': True, 'drop_last': True} \
        if not args.no_cuda else {}

    test_params = {'batch_size': args.batch_size, 'shuffle': True, 'num_workers': 4, 'pin_memory': True, 'drop_last': False} \
        if not args.no_cuda else {}
        
    # Datasets
    train_set = RC_Dataset(train_dataset, train_transform, args)
    test_set = RC_Dataset(test_dataset, test_transform, args)

    # Dataloaders
    train_loader = data.DataLoader(train_set, **train_params)
    test_loader = data.DataLoader(test_set, **test_params)
    
    return train_loader, test_loader


# extract mean and std for images
def mean_std(args, dataset):
    
    rgb_mean_std_filename = os.path.join(args.dataset, 'rgb_mean_std.pkl')
    
    if os.path.isfile(rgb_mean_std_filename ):
        print("\nRGB data found!")
        with open(rgb_mean_std_filename, 'rb') as fr:
            data = pickle.load(fr)
            
        mean = data['mean']
        std = data['std']
        
        print('mean : {:.3f},{:.3f},{:.3f}'.format(mean[0],mean[1],mean[2]))
        print('std : {:.3f},{:.3f},{:.3f}\n'.format(std[0],std[1],std[2]))

        
    else:
        print("\nNo RGB data found!")
        mean = 0
        std = 0
        num = 0
        
        image_list = dataset['image0'].values
        num = len(image_list)
            
        for frame in dataset['image0']:
            frame = os.sep.join(frame.split("/")[-2:])
            image = cv2.imread(args.dataset + '/images/' + frame)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
            img_mean, img_std = cv2.meanStdDev(image)
            
            mean += img_mean / 255
            std += img_std / 255
                
        mean /= num
        std /= num
        
        mean = mean.reshape(-1).tolist()
        std = std.reshape(-1).tolist()
        
        data = {"mean" : mean, "std" : std}
        
        with open(rgb_mean_std_filename, 'wb') as fr:
            pickle.dump(data, fr)
        
        print('mean : {:.3f},{:.3f},{:.3f}'.format(mean[0],mean[1],mean[2]))
        print('std : {:.3f},{:.3f},{:.3f}\n'.format(std[0],std[1],std[2]))

    return mean, std
